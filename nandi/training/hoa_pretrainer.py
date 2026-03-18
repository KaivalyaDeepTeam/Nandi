"""
Phase 1: Hindsight Optimal Action (HOA) Pre-training.

Computes the perfect action at each bar using future prices, then trains
the agent via supervised classification (weighted cross-entropy).

Two modes:
  position_aware=True  (DQN default): Simulates position tracking, produces
      HOLD/LONG/SHORT/CLOSE labels. ~85-90% HOLD, ~1-2% entries, ~2% CLOSE.
  position_aware=False (PPO default): Flat labels only — each bar is labeled
      independently as HOLD/LONG/SHORT. No CLOSE labels. ~50-70% HOLD,
      ~15-25% LONG, ~15-25% SHORT. Gives PPO much more entry signal.

Optional price-flip augmentation doubles training data by mirroring prices
(1/price) and swapping LONG↔SHORT, preventing directional bias.
"""

import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nandi.config import HOA_CONFIG, HOA_CONFIG_H1, DQN_CONFIG, PAIR_TO_IDX, TRANSACTION_COST_BPS

logger = logging.getLogger(__name__)

# Action constants
HOLD = 0
LONG = 1
SHORT = 2
CLOSE = 3


def compute_hoa_labels(prices, features, lookback, pair_name="unknown",
                       horizon=None, cost_threshold_mult=None,
                       timeframe="M5", position_aware=True,
                       flat_hold_pct=0.60):
    """Compute Hindsight Optimal Action labels for a price series.

    Args:
        prices: numpy array of prices
        features: numpy array (n_bars, n_features), pre-scaled
        lookback: int, lookback window for market state
        pair_name: str
        horizon: int, look-ahead bars (default from HOA_CONFIG)
        cost_threshold_mult: float (default from HOA_CONFIG)
        timeframe: str
        position_aware: bool — if True, simulate position tracking with CLOSE
            labels (DQN mode). If False, flat labels only: each bar independently
            labeled HOLD/LONG/SHORT (PPO mode — more entry signal).
        flat_hold_pct: float — target HOLD percentage for flat mode (default 0.60).
            Uses percentile-based thresholding since with long horizons, simple
            cost-threshold gives 0% HOLD.

    Returns:
        market_states: (N, lookback, n_features)
        position_infos: (N, 8)
        labels: (N,) int — action labels
        pair_indices: (N,) int — pair index for embedding
    """
    cfg = HOA_CONFIG
    horizon = horizon or cfg["horizon"]
    cost_mult = cost_threshold_mult or cfg["cost_threshold_mult"]

    # Transaction cost in return space
    cost_bps = TRANSACTION_COST_BPS
    cost_return = cost_bps / 10000.0

    n_bars = len(prices)
    n_features = features.shape[1]
    pair_idx = PAIR_TO_IDX.get(pair_name, 0)

    market_states = []
    position_infos = []
    labels = []
    pair_indices = []

    # First pass: compute edges for all valid bars
    bar_edges = []  # (bar_idx, best_long, best_short)

    for t in range(lookback, n_bars - horizon):
        price_t = prices[t]
        if price_t <= 0:
            continue

        future = prices[t + 1: t + 1 + horizon]
        if len(future) < 2:
            continue

        # Long PnL curve: (future - current) / current
        long_pnl = (future - price_t) / price_t - cost_return
        # Short PnL curve: (current - future) / current
        short_pnl = (price_t - future) / price_t - cost_return

        best_long = np.max(long_pnl)
        best_short = np.max(short_pnl)

        bar_edges.append((t, best_long, best_short))

    # Second pass: assign flat labels
    flat_labels = np.zeros(n_bars, dtype=np.int64)  # default HOLD

    if not position_aware and len(bar_edges) > 0:
        # ── Percentile-based thresholding for flat mode ──
        # With long horizons, every bar has profitable moves in both directions.
        # Use directional edge strength to select top entries.
        # net_edge = |best_long - best_short| measures how "one-sided" the move is.
        net_edges = np.array([abs(bl - bs) for _, bl, bs in bar_edges])
        edge_threshold = np.percentile(net_edges, flat_hold_pct * 100)
        logger.info(f"  {pair_name} flat-mode edge threshold: {edge_threshold * 10000:.1f} bps "
                    f"(p{flat_hold_pct * 100:.0f})")

        for t, best_long, best_short in bar_edges:
            net_edge = abs(best_long - best_short)
            if net_edge > edge_threshold:
                if best_long >= best_short:
                    flat_labels[t] = LONG
                else:
                    flat_labels[t] = SHORT
    else:
        # ── Cost-threshold labeling (position-aware mode / DQN) ──
        threshold = cost_mult * cost_return
        for t, best_long, best_short in bar_edges:
            if best_long > threshold and best_long >= best_short:
                flat_labels[t] = LONG
            elif best_short > threshold and best_short > best_long:
                flat_labels[t] = SHORT

    if not position_aware:
        # ── Flat-only mode (PPO): use flat_labels directly, no CLOSE ──
        for t in range(lookback, n_bars - horizon):
            price_t = prices[t]
            if price_t <= 0:
                continue

            ms = features[t - lookback: t]
            if ms.shape[0] < lookback:
                continue

            pos_info = np.array([
                0.0,    # position (always flat in flat-label mode)
                0.0,    # equity_return
                0.0,    # drawdown
                float(np.std(features[max(0, t - 10):t, 0]) if t > 10 else 1.0),
                0.0,    # bars_in_trade
                0.0,    # time_of_day
                0.0,    # unrealized_pnl
                0.0,    # mfe
            ], dtype=np.float32)

            market_states.append(ms.astype(np.float32))
            position_infos.append(pos_info)
            labels.append(flat_labels[t])
            pair_indices.append(pair_idx)
    else:
        # ── Position-aware mode (DQN): simulate trades, add CLOSE labels ──
        simulated_pos = 0  # -1, 0, +1
        entry_bar = 0

        for t in range(lookback, n_bars - horizon):
            price_t = prices[t]
            if price_t <= 0:
                continue

            ms = features[t - lookback: t]
            if ms.shape[0] < lookback:
                continue

            label = HOLD  # default

            if simulated_pos == 0:
                label = flat_labels[t]
                if label in (LONG, SHORT):
                    simulated_pos = 1 if label == LONG else -1
                    entry_bar = t
            else:
                future = prices[t + 1: t + 1 + horizon]
                if len(future) < 2:
                    label = HOLD
                else:
                    if simulated_pos == 1:
                        current_pnl = (price_t - prices[entry_bar]) / prices[entry_bar]
                        future_pnl = (future - prices[entry_bar]) / prices[entry_bar]
                        future_max = np.max(future_pnl)
                        if current_pnl > cost_return and future_max < current_pnl * 0.8:
                            label = CLOSE
                            simulated_pos = 0
                        elif t - entry_bar > horizon:
                            label = CLOSE
                            simulated_pos = 0
                        else:
                            label = HOLD
                    else:
                        current_pnl = (prices[entry_bar] - price_t) / prices[entry_bar]
                        future_pnl = (prices[entry_bar] - future) / prices[entry_bar]
                        future_max = np.max(future_pnl)
                        if current_pnl > cost_return and future_max < current_pnl * 0.8:
                            label = CLOSE
                            simulated_pos = 0
                        elif t - entry_bar > horizon:
                            label = CLOSE
                            simulated_pos = 0
                        else:
                            label = HOLD

            # Build position info (8-dim, V4)
            bars_in_trade = t - entry_bar if simulated_pos != 0 else 0
            bars_norm = bars_in_trade / 100.0

            unrealized_pnl = 0.0
            trade_mfe = 0.0
            if simulated_pos != 0 and prices[entry_bar] > 0:
                exc = (price_t - prices[entry_bar]) / prices[entry_bar]
                if simulated_pos == -1:
                    exc = -exc
                unrealized_pnl = exc
                for k in range(entry_bar + 1, t + 1):
                    k_exc = (prices[k] - prices[entry_bar]) / prices[entry_bar]
                    if simulated_pos == -1:
                        k_exc = -k_exc
                    trade_mfe = max(trade_mfe, k_exc)

            unrealized_norm = float(np.clip(unrealized_pnl * 100.0, -2.0, 2.0))
            mfe_norm = float(np.clip(trade_mfe * 100.0, 0.0, 2.0))

            pos_info = np.array([
                float(simulated_pos),
                0.0,    # equity_return
                0.0,    # drawdown
                float(np.std(features[max(0, t - 10):t, 0]) if t > 10 else 1.0),
                bars_norm,
                0.0,    # time_of_day
                unrealized_norm,
                mfe_norm,
            ], dtype=np.float32)

            market_states.append(ms.astype(np.float32))
            position_infos.append(pos_info)
            labels.append(label)
            pair_indices.append(pair_idx)

    market_states = np.array(market_states)
    position_infos = np.array(position_infos)
    labels = np.array(labels, dtype=np.int64)
    pair_indices = np.array(pair_indices, dtype=np.int64)

    # Log distribution
    label_names = [("HOLD", 0), ("LONG", 1), ("SHORT", 2)]
    if position_aware:
        label_names.append(("CLOSE", 3))
    for action_name, action_id in label_names:
        count = np.sum(labels == action_id)
        pct = 100.0 * count / max(1, len(labels))
        logger.info(f"  {pair_name} HOA {action_name}: {count:,} ({pct:.1f}%)")

    return market_states, position_infos, labels, pair_indices


def compute_spin_hoa_labels(prices, features, lookback, pair_name="unknown",
                            horizon=12, cost_threshold_mult=2.0,
                            flat_hold_pct=0.55,
                            atr_series=None, h1_trend_series=None,
                            risk_config=None, max_samples=30_000):
    """Compute HOA labels for SPIN — stop-loss-aware labeling.

    Memory-efficient: computes lightweight labels first, subsamples,
    then builds the expensive market_state windows only for kept indices.

    Unlike standard HOA which uses unconstrained future prices, SPIN HOA
    simulates the actual risk gates: would a LONG/SHORT trade have been
    profitable AFTER accounting for SL hit, max hold, trend filter?

    Args:
        prices: numpy array of close prices
        features: (n_bars, n_features) pre-scaled features
        lookback: int
        pair_name: str
        horizon: int, look-ahead bars
        cost_threshold_mult: float
        flat_hold_pct: float, target HOLD percentage
        atr_series: (n_bars,) ATR values
        h1_trend_series: (n_bars,) H1 trend direction
        risk_config: SPIN_RISK_CONFIG override
        max_samples: int, max samples to return (subsamples if more)

    Returns:
        market_states, position_infos, labels, pair_indices
    """
    from nandi.config import SPIN_RISK_CONFIG, SPIN_CONFIG

    cfg_risk = risk_config or SPIN_RISK_CONFIG
    pair_idx_val = PAIR_TO_IDX.get(pair_name, 0)
    cost_return = TRANSACTION_COST_BPS / 10000.0
    sl_mult = cfg_risk["stop_loss_atr_mult"]
    max_hold = cfg_risk["max_hold_bars"]
    position_dim = SPIN_CONFIG["position_dim"]

    n_bars = len(prices)

    if atr_series is None:
        atr_series = np.ones(n_bars) * 0.001
    if h1_trend_series is None:
        h1_trend_series = np.zeros(n_bars)

    # ── Pass 1: compute lightweight labels (just bar index + label) ──
    bar_edges = []

    for t in range(lookback, n_bars - horizon):
        price_t = prices[t]
        if price_t <= 0:
            continue

        atr_t = atr_series[t]
        h1_trend_t = h1_trend_series[t]

        future = prices[t + 1: t + 1 + min(horizon, max_hold)]
        if len(future) < 2:
            continue

        # Simulate LONG trade with stop-loss
        long_edge = 0.0
        if h1_trend_t >= 0 or not cfg_risk.get("trend_filter", True):
            sl_price = price_t - atr_t * sl_mult
            long_pnl = -cost_return * 2
            for k, p in enumerate(future):
                if p <= sl_price:
                    long_pnl = (sl_price - price_t) / price_t - cost_return * 2
                    break
                pnl = (p - price_t) / price_t - cost_return * 2
                long_pnl = max(long_pnl, pnl)
            long_edge = long_pnl

        # Simulate SHORT trade with stop-loss
        short_edge = 0.0
        if h1_trend_t <= 0 or not cfg_risk.get("trend_filter", True):
            sl_price = price_t + atr_t * sl_mult
            short_pnl = -cost_return * 2
            for k, p in enumerate(future):
                if p >= sl_price:
                    short_pnl = (price_t - sl_price) / price_t - cost_return * 2
                    break
                pnl = (price_t - p) / price_t - cost_return * 2
                short_pnl = max(short_pnl, pnl)
            short_edge = short_pnl

        bar_edges.append((t, long_edge, short_edge))

    if not bar_edges:
        return (np.array([]).reshape(0, lookback, features.shape[1]),
                np.zeros((0, position_dim), dtype=np.float32),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64))

    # Percentile-based thresholding to assign labels
    net_edges = np.array([abs(le - se) for _, le, se in bar_edges])
    edge_threshold = np.percentile(net_edges, flat_hold_pct * 100)
    logger.info(f"  {pair_name} SPIN-HOA edge threshold: {edge_threshold * 10000:.1f} bps "
                f"(p{flat_hold_pct * 100:.0f})")

    bar_indices = []
    bar_labels = []
    for t, long_edge, short_edge in bar_edges:
        net_edge = abs(long_edge - short_edge)
        label = HOLD
        if net_edge > edge_threshold:
            if long_edge > short_edge and long_edge > 0:
                label = LONG
            elif short_edge > long_edge and short_edge > 0:
                label = SHORT
        bar_indices.append(t)
        bar_labels.append(label)

    bar_indices = np.array(bar_indices)
    bar_labels = np.array(bar_labels, dtype=np.int64)

    total_before = len(bar_labels)
    for action_name, action_id in [("HOLD", 0), ("LONG", 1), ("SHORT", 2)]:
        count = np.sum(bar_labels == action_id)
        pct = 100.0 * count / max(1, len(bar_labels))
        logger.info(f"  {pair_name} SPIN-HOA {action_name}: {count:,} ({pct:.1f}%)")

    # ── Pass 2: subsample if needed, THEN build market_state windows ──
    if len(bar_indices) > max_samples:
        keep = np.random.permutation(len(bar_indices))[:max_samples]
        keep.sort()
        bar_indices = bar_indices[keep]
        bar_labels = bar_labels[keep]
        logger.info(f"  {pair_name} subsampled: {max_samples:,} / {total_before:,}")

    n_keep = len(bar_indices)
    n_feat = features.shape[1]
    market_states = np.empty((n_keep, lookback, n_feat), dtype=np.float32)
    position_infos = np.zeros((n_keep, position_dim), dtype=np.float32)
    pair_indices = np.full(n_keep, pair_idx_val, dtype=np.int64)

    for i, t in enumerate(bar_indices):
        market_states[i] = features[t - lookback: t]

    return market_states, position_infos, bar_labels, pair_indices


class HOADataset(Dataset):
    """Dataset of HOA-labeled transitions for supervised pre-training."""

    def __init__(self, market_states, position_infos, labels, pair_indices):
        self.market_states = torch.tensor(market_states, dtype=torch.float32)
        self.position_infos = torch.tensor(position_infos, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.pair_indices = torch.tensor(pair_indices, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.market_states[idx], self.position_infos[idx],
                self.labels[idx], self.pair_indices[idx])


class HOAPretrainer:
    """Phase 1: Supervised pre-training with Hindsight Optimal Actions.

    Trains the DQN agent using weighted cross-entropy on HOA labels.
    Inverse class frequency weighting handles the ~85% HOLD imbalance.
    """

    def __init__(self, agent, pair_data, lookback=120, timeframe="M5",
                 hoa_config=None, device=None,
                 position_aware=True, price_flip_augment=False):
        """
        Args:
            agent: NandiDQNAgent or NandiPPOAgent instance
            pair_data: dict of pair_name → {train_features, train_prices, ...}
            lookback: int
            timeframe: str
            hoa_config: override HOA_CONFIG
            device: torch.device
            position_aware: bool — True for DQN (CLOSE labels), False for PPO
                (flat labels only, more entry signal)
            price_flip_augment: bool — if True, augment with mirrored prices
                (1/price, LONG↔SHORT swapped) to prevent directional bias
        """
        self.agent = agent
        self.pair_data = pair_data
        self.lookback = lookback
        self.timeframe = timeframe
        self.cfg = hoa_config or HOA_CONFIG
        self.position_aware = position_aware
        self.price_flip_augment = price_flip_augment

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

    def compute_all_labels(self):
        """Compute HOA labels for all pairs, with optional price-flip augmentation."""
        all_ms, all_pi, all_labels, all_pairs = [], [], [], []

        mode = "flat" if not self.position_aware else "position-aware"
        logger.info(f"HOA mode: {mode} | price-flip augment: {self.price_flip_augment}")

        for pair_name, data in self.pair_data.items():
            logger.info(f"Computing HOA labels for {pair_name.upper()}...")
            flat_hold_pct = self.cfg.get("flat_hold_pct", 0.60)
            ms, pi, labels, pair_idx = compute_hoa_labels(
                prices=data["train_prices"],
                features=data["train_features"],
                lookback=self.lookback,
                pair_name=pair_name,
                horizon=self.cfg["horizon"],
                cost_threshold_mult=self.cfg["cost_threshold_mult"],
                timeframe=self.timeframe,
                position_aware=self.position_aware,
                flat_hold_pct=flat_hold_pct,
            )
            if len(labels) > 0:
                all_ms.append(ms)
                all_pi.append(pi)
                all_labels.append(labels)
                all_pairs.append(pair_idx)

            # Price-flip augmentation: mirror prices, swap LONG↔SHORT
            if self.price_flip_augment and len(labels) > 0:
                logger.info(f"Computing price-flip augmentation for {pair_name.upper()}...")
                # Invert prices: 1/price (e.g., EURUSD 1.10 → 0.909)
                flipped_prices = 1.0 / data["train_prices"]
                ms_flip, pi_flip, labels_flip, pair_idx_flip = compute_hoa_labels(
                    prices=flipped_prices,
                    features=data["train_features"],
                    lookback=self.lookback,
                    pair_name=pair_name,
                    horizon=self.cfg["horizon"],
                    cost_threshold_mult=self.cfg["cost_threshold_mult"],
                    timeframe=self.timeframe,
                    position_aware=self.position_aware,
                    flat_hold_pct=flat_hold_pct,
                )
                if len(labels_flip) > 0:
                    # Swap LONG↔SHORT labels (HOLD stays HOLD, CLOSE stays CLOSE)
                    swap_map = np.array([HOLD, SHORT, LONG, CLOSE], dtype=np.int64)
                    labels_flip = swap_map[labels_flip]
                    all_ms.append(ms_flip)
                    all_pi.append(pi_flip)
                    all_labels.append(labels_flip)
                    all_pairs.append(pair_idx_flip)
                    logger.info(f"  {pair_name} flip LONG: {np.sum(labels_flip == LONG):,} "
                                f"SHORT: {np.sum(labels_flip == SHORT):,}")

        if not all_ms:
            raise ValueError("No HOA labels computed — check data")

        return (
            np.concatenate(all_ms),
            np.concatenate(all_pi),
            np.concatenate(all_labels),
            np.concatenate(all_pairs),
        )

    def train(self):
        """Run HOA pre-training.

        Returns:
            metrics: dict with accuracy metrics
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Phase 1: HOA Pre-training")
        logger.info(f"{'=' * 60}")

        start_time = time.time()

        # Compute labels
        ms, pi, labels, pairs = self.compute_all_labels()
        logger.info(f"Total HOA samples: {len(labels):,}")

        # Compute class weights: sqrt-inverse-frequency with cap
        n_classes = 4 if self.position_aware else 3
        class_counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
        if not self.position_aware:
            # Pad to 4 classes for the model (CLOSE gets zero weight)
            class_counts = np.concatenate([class_counts, [1.0]])
        class_counts = np.maximum(class_counts, 1.0)
        class_weights = 1.0 / np.sqrt(class_counts)
        class_weights = class_weights / class_weights.min()
        class_weights = np.minimum(class_weights, 10.0)
        if not self.position_aware:
            # Zero out CLOSE class weight — no CLOSE labels exist
            class_weights[CLOSE] = 0.0
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        logger.info(f"Class weights: {class_weights} (sqrt-inv-freq, capped 10:1)")

        # Create dataset and dataloader
        dataset = HOADataset(ms, pi, labels, pairs)
        dataloader = DataLoader(
            dataset, batch_size=self.cfg["batch_size"],
            shuffle=True, drop_last=True,
            num_workers=0,
        )

        # Setup
        self.agent.to(self.device)
        self.agent.train()
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.cfg["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg["epochs"],
        )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights_t,
            label_smoothing=self.cfg["label_smoothing"],
        )

        best_weighted_acc = 0.0
        best_state = None

        for epoch in range(self.cfg["epochs"]):
            total_loss = 0.0
            correct = np.zeros(4)
            total = np.zeros(4)
            n_batches = 0

            for batch_ms, batch_pi, batch_labels, batch_pairs in dataloader:
                batch_ms = batch_ms.to(self.device)
                batch_pi = batch_pi.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_pairs = batch_pairs.to(self.device)

                # Get classification logits
                logits = self.agent.get_classification_logits(
                    batch_ms, batch_pi, batch_pairs,
                )

                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                # Track per-class accuracy
                preds = logits.argmax(dim=-1)
                for c in range(4):
                    mask = batch_labels == c
                    total[c] += mask.sum().item()
                    correct[c] += (preds[mask] == c).sum().item()

            scheduler.step()

            # Compute metrics
            avg_loss = total_loss / max(1, n_batches)
            per_class_acc = np.where(total > 0, correct / total, 0.0)
            overall_acc = np.sum(correct) / max(1, np.sum(total))

            # Balanced macro accuracy: equal weight per class (not biased by HOLD count)
            active_classes = total > 0
            if active_classes.sum() > 0:
                w_acc = np.mean(per_class_acc[active_classes])
            else:
                w_acc = 0.0

            logger.info(
                f"Epoch {epoch + 1:>2}/{self.cfg['epochs']} | "
                f"Loss: {avg_loss:.4f} | "
                f"HOLD={per_class_acc[0]:.3f} LONG={per_class_acc[1]:.3f} "
                f"SHORT={per_class_acc[2]:.3f} CLOSE={per_class_acc[3]:.3f} | "
                f"Overall={overall_acc:.3f} Macro={w_acc:.3f}"
            )

            if w_acc > best_weighted_acc:
                best_weighted_acc = w_acc
                best_state = {k: v.cpu().clone() for k, v in
                              self.agent.state_dict().items()}

        # Restore best checkpoint
        if best_state is not None:
            self.agent.load_state_dict(best_state)
            self.agent.to(self.device)

        elapsed = time.time() - start_time
        logger.info(f"\nHOA Pre-training complete in {elapsed / 60:.1f} minutes")
        logger.info(f"Best weighted accuracy: {best_weighted_acc:.3f}")

        # Check success criteria
        metrics = {
            "hold_acc": per_class_acc[0],
            "long_acc": per_class_acc[1],
            "short_acc": per_class_acc[2],
            "close_acc": per_class_acc[3],
            "weighted_acc": best_weighted_acc,
            "loss": avg_loss,
        }

        passed = (
            per_class_acc[0] >= self.cfg.get("min_hold_acc", 0.85) * 0.9 and
            (per_class_acc[1] >= self.cfg.get("min_trade_acc", 0.40) * 0.8 or
             per_class_acc[2] >= self.cfg.get("min_trade_acc", 0.40) * 0.8) and
            best_weighted_acc >= self.cfg.get("min_weighted_acc", 0.60) * 0.8
        )

        if passed:
            logger.info("Phase 1 PASSED — proceeding to RL fine-tuning")
        else:
            logger.warning(
                "Phase 1 accuracy below targets — RL fine-tuning may still work "
                "but convergence will be slower"
            )

        metrics["passed"] = passed
        return metrics
