"""
Prioritized Experience Replay (PER) + N-step return buffer.

PER (Schaul et al., 2016): Sample transitions proportional to their
TD-error magnitude. Higher-error transitions are sampled more often,
accelerating learning on surprising experiences.

N-step (Sutton, 1988): Compute multi-step bootstrapped returns:
    R_n = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n * Q(s_{t+n})
Reduces bias at the cost of variance.
"""

import numpy as np


class SumTree:
    """Binary sum tree for O(log n) priority sampling."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(tree_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s):
        """Sample a leaf node by cumulative priority sum s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with sum-tree sampling.

    Stores transitions: (market_state, position_info, action, reward,
                         next_market_state, next_position_info, done,
                         pair_idx, action_mask)

    Supports importance sampling (IS) weight correction with
    annealing beta from beta_start → 1.0.
    """

    def __init__(self, capacity=500_000, alpha=0.6, beta_start=0.4,
                 beta_end=1.0, beta_steps=500_000, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.epsilon = epsilon
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self._step = 0

    def add(self, market_state, position_info, action, reward,
            next_market_state, next_position_info, done,
            pair_idx=0, action_mask=None):
        """Add transition with max priority (ensures new samples get replayed)."""
        transition = (
            market_state, position_info, action, reward,
            next_market_state, next_position_info, done,
            pair_idx, action_mask,
        )
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size):
        """Sample a batch proportional to priority.

        Returns:
            batch: dict with numpy arrays for each field
            indices: tree indices for priority updates
            is_weights: importance sampling weights (B,)
        """
        self._step += 1
        beta = min(
            self.beta_end,
            self.beta_start + (self.beta_end - self.beta_start) *
            self._step / self.beta_steps,
        )

        indices = []
        priorities = []
        transitions = []

        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            if data is None:
                # Fallback: resample
                s = np.random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
            if data is None:
                continue
            indices.append(idx)
            priorities.append(priority)
            transitions.append(data)

        if len(transitions) < batch_size:
            # Pad with random samples if tree has gaps
            while len(transitions) < batch_size:
                s = np.random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
                if data is not None:
                    indices.append(idx)
                    priorities.append(priority)
                    transitions.append(data)

        priorities = np.array(priorities, dtype=np.float64)

        # Importance sampling weights
        n = self.tree.size
        min_prob = np.min(priorities) / (self.tree.total() + 1e-10)
        max_weight = (n * min_prob + 1e-10) ** (-beta)
        probs = priorities / (self.tree.total() + 1e-10)
        is_weights = (n * probs + 1e-10) ** (-beta) / (max_weight + 1e-10)
        is_weights = np.array(is_weights, dtype=np.float32)

        # Unpack transitions
        ms = np.array([t[0] for t in transitions], dtype=np.float32)
        pi = np.array([t[1] for t in transitions], dtype=np.float32)
        actions = np.array([t[2] for t in transitions], dtype=np.int64)
        rewards = np.array([t[3] for t in transitions], dtype=np.float32)
        next_ms = np.array([t[4] for t in transitions], dtype=np.float32)
        next_pi = np.array([t[5] for t in transitions], dtype=np.float32)
        dones = np.array([t[6] for t in transitions], dtype=np.float32)
        pair_ids = np.array([t[7] for t in transitions], dtype=np.int64)
        # Action masks: list of arrays (may be None)
        masks_list = [t[8] for t in transitions]
        if masks_list[0] is not None:
            action_masks = np.array(masks_list, dtype=bool)
        else:
            action_masks = None

        batch = {
            "market_state": ms,
            "position_info": pi,
            "actions": actions,
            "rewards": rewards,
            "next_market_state": next_ms,
            "next_position_info": next_pi,
            "dones": dones,
            "pair_ids": pair_ids,
            "action_masks": action_masks,
        }
        return batch, indices, is_weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size


class NStepBuffer:
    """N-step return accumulator.

    Collects transitions and computes n-step returns:
        R_n = r_0 + γ*r_1 + ... + γ^{n-1}*r_{n-1}
    Then yields (s_0, a_0, R_n, s_n, done_n) for the replay buffer.
    """

    def __init__(self, n_step=3, gamma=0.95):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = []

    def add(self, market_state, position_info, action, reward,
            next_market_state, next_position_info, done,
            pair_idx=0, action_mask=None):
        """Add a single-step transition. Returns n-step transition if ready."""
        self.buffer.append((
            market_state, position_info, action, reward,
            next_market_state, next_position_info, done,
            pair_idx, action_mask,
        ))

        if done:
            # Flush all remaining transitions on episode end
            return self._flush()

        if len(self.buffer) >= self.n_step:
            return [self._compute_nstep()]

        return []

    def _compute_nstep(self):
        """Compute n-step return from the first n entries and pop the first."""
        n = min(len(self.buffer), self.n_step)
        # Discounted return
        R = 0.0
        for i in reversed(range(n)):
            R = self.buffer[i][3] + self.gamma * R * (1.0 - self.buffer[i][6])

        first = self.buffer[0]
        last = self.buffer[n - 1]

        result = (
            first[0],   # market_state (s_0)
            first[1],   # position_info (s_0)
            first[2],   # action (a_0)
            R,           # n-step return
            last[4],    # next_market_state (s_n)
            last[5],    # next_position_info (s_n)
            last[6],    # done (at step n)
            first[7],   # pair_idx
            first[8],   # action_mask
        )
        self.buffer.pop(0)
        return result

    def _flush(self):
        """Flush all remaining transitions at episode end."""
        results = []
        while len(self.buffer) > 0:
            results.append(self._compute_nstep())
        return results

    def reset(self):
        self.buffer = []
