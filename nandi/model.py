"""
Nandi Model — Multi-Scale Fractal Attention Network (MSFAN) + PPO Agent.

Architecture (novel):
  1. Feature Projection → projects raw features into d_model space
  2. Multi-Scale Causal Convolutions → captures patterns at 3 temporal scales
     - Micro (kernel=3, dilation=1): short-term momentum
     - Meso (kernel=7, dilation=4): medium-term patterns
     - Macro (kernel=15, dilation=16): long-term regime
  3. Cross-Scale Attention → each scale attends to all scales
  4. Position Context → embeds current position/drawdown/P&L
  5. Actor Head → continuous position output [-1, 1] with learned std
  6. Critic Head → state value estimate
  7. Uncertainty Head → epistemic uncertainty for trade gating
"""

import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from nandi.config import ENCODER_CONFIG, PPO_CONFIG, MODEL_DIR


class CausalConvBlock(layers.Layer):
    """Dilated causal convolution block with residual connection."""

    def __init__(self, d_model, kernel_size, dilation_rate, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.conv1 = layers.Conv1D(
            d_model, kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation=None,
        )
        self.conv2 = layers.Conv1D(d_model, 1, activation=None)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)
        self.activation = layers.Activation("gelu")
        self.residual_proj = None  # Built on first call if needed

    def call(self, x, training=False):
        residual = x
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        h = self.dropout(h, training=training)
        h = self.conv2(h)
        h = self.norm2(h)
        # Residual connection (project if shapes differ)
        if residual.shape[-1] != h.shape[-1]:
            if self.residual_proj is None:
                self.residual_proj = layers.Dense(self.d_model)
            residual = self.residual_proj(residual)
        return self.activation(h + residual)


class MultiScaleEncoder(keras.Model):
    """Multi-Scale Fractal Attention Network (MSFAN).

    Processes market data at multiple temporal scales using dilated causal
    convolutions, then uses cross-scale multi-head attention to capture
    inter-scale relationships. This is novel because it explicitly models
    the fractal nature of financial markets — patterns that repeat at
    different timescales.
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or ENCODER_CONFIG
        d = config["d_model"]

        # Scale-specific causal conv blocks
        self.scale_blocks = []
        for i in range(config["n_scales"]):
            block = CausalConvBlock(
                d, config["kernel_sizes"][i],
                config["dilations"][i],
                config["dropout"],
            )
            self.scale_blocks.append(block)

        # Cross-scale attention
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=config["n_heads"],
            key_dim=d // config["n_heads"],
            dropout=config["dropout"],
        )
        self.cross_norm = layers.LayerNormalization()

        # Output
        self.output_dense = layers.Dense(d, activation="gelu")
        self.output_norm = layers.LayerNormalization()

    def call(self, x, training=False):
        """
        Args:
            x: (batch, lookback, d_model) projected features
        Returns:
            (batch, d_model) encoded market state
        """
        scale_outputs = []
        for block in self.scale_blocks:
            h = block(x, training=training)
            # Take last timestep from each scale
            scale_outputs.append(h[:, -1:, :])

        # Stack: (batch, n_scales, d_model)
        multi_scale = tf.concat(scale_outputs, axis=1)

        # Cross-scale attention: each scale attends to all scales
        attended = self.cross_attn(
            multi_scale, multi_scale,
            training=training,
        )
        attended = self.cross_norm(attended + multi_scale)

        # Pool across scales → (batch, d_model)
        pooled = tf.reduce_mean(attended, axis=1)

        return self.output_norm(self.output_dense(pooled))


class NandiAgent(keras.Model):
    """PPO Actor-Critic agent with MSFAN encoder.

    Outputs:
        - action_mean: target position [-1, 1]
        - action_std: learned exploration noise
        - value: state value estimate
        - uncertainty: epistemic uncertainty (for trade gating)
    """

    def __init__(self, n_features, encoder_config=None):
        super().__init__()
        config = encoder_config or ENCODER_CONFIG
        d = config["d_model"]

        # Feature projection
        self.feature_proj = keras.Sequential([
            layers.Dense(d, activation="gelu"),
            layers.LayerNormalization(),
        ])

        # Market state encoder
        self.encoder = MultiScaleEncoder(config)

        # Position context embedding
        self.position_embed = keras.Sequential([
            layers.Dense(32, activation="gelu"),
            layers.Dense(32, activation="gelu"),
        ])

        # Combined processing
        self.trunk = keras.Sequential([
            layers.Dense(d, activation="gelu"),
            layers.LayerNormalization(),
            layers.Dense(d // 2, activation="gelu"),
        ])

        # Actor head (continuous action)
        self.actor_mean = layers.Dense(1, activation="tanh")
        self.log_std = tf.Variable(
            tf.constant(-0.5, shape=(1,)), trainable=True, name="log_std"
        )

        # Critic head
        self.critic_head = layers.Dense(1)

    def call(self, market_state, position_info, training=False):
        """Forward pass.

        Args:
            market_state: (batch, lookback, n_features)
            position_info: (batch, position_info_dim)
        """
        # Project features to d_model
        projected = self.feature_proj(market_state)

        # Encode multi-scale market state
        encoded = self.encoder(projected, training=training)

        # Add position context
        pos_emb = self.position_embed(position_info)
        combined = tf.concat([encoded, pos_emb], axis=-1)

        # Shared trunk
        h = self.trunk(combined)

        # Actor
        action_mean = self.actor_mean(h)
        action_std = tf.exp(tf.clip_by_value(self.log_std, -3.0, 0.5))

        # Critic
        value = self.critic_head(h)

        return action_mean, action_std, value

    def get_action(self, market_state, position_info, deterministic=False):
        """Sample action from policy.

        Returns: action, log_prob, value, uncertainty
        """
        # Ensure batch dimension
        if market_state.ndim == 2:
            market_state = market_state[np.newaxis]
        if position_info.ndim == 1:
            position_info = position_info[np.newaxis]

        market_state = tf.cast(market_state, tf.float32)
        position_info = tf.cast(position_info, tf.float32)

        mean, std, value = self(
            market_state, position_info, training=not deterministic
        )

        if deterministic:
            action = mean
        else:
            noise = tf.random.normal(tf.shape(mean))
            action = mean + noise * std

        action = tf.clip_by_value(action, -1.0, 1.0)

        # Log probability of action under Gaussian policy
        log_prob = -0.5 * (
            tf.square((action - mean) / (std + 1e-8))
            + 2.0 * tf.math.log(std + 1e-8)
            + tf.math.log(2.0 * np.pi)
        )
        log_prob = tf.reduce_sum(log_prob, axis=-1)

        # MC dropout uncertainty: variance across multiple forward passes
        unc = 0.0
        if deterministic:
            preds = []
            for _ in range(5):
                m, _, _ = self(market_state, position_info, training=True)
                preds.append(m.numpy())
            unc = float(np.std(preds))

        return (
            action.numpy().flatten()[0],
            log_prob.numpy().flatten()[0],
            value.numpy().flatten()[0],
            unc,
        )

    def evaluate_actions(self, market_states, position_infos, actions):
        """Evaluate actions for PPO update.

        Returns: log_probs, values, entropy
        """
        mean, std, values = self(
            market_states, position_infos, training=True
        )

        # Log probability
        log_probs = -0.5 * (
            tf.square((actions - mean) / (std + 1e-8))
            + 2.0 * tf.math.log(std + 1e-8)
            + tf.math.log(2.0 * np.pi)
        )
        log_probs = tf.reduce_sum(log_probs, axis=-1)

        # Entropy of Gaussian
        entropy = 0.5 * (1.0 + tf.math.log(2.0 * np.pi * tf.square(std) + 1e-8))
        entropy = tf.reduce_sum(entropy, axis=-1)

        return log_probs, tf.squeeze(values, -1), entropy

    def get_uncertainty(self, market_state, position_info, n_samples=10):
        """Monte Carlo dropout uncertainty estimation."""
        if market_state.ndim == 2:
            market_state = market_state[np.newaxis]
        if position_info.ndim == 1:
            position_info = position_info[np.newaxis]

        market_state = tf.cast(market_state, tf.float32)
        position_info = tf.cast(position_info, tf.float32)

        predictions = []
        for _ in range(n_samples):
            mean, _, _ = self(market_state, position_info, training=True)
            predictions.append(mean.numpy())

        preds = np.array(predictions)
        return float(np.std(preds))  # Higher std = more uncertain

    def save_agent(self, path=None):
        """Save model weights."""
        path = path or os.path.join(MODEL_DIR, "nandi_agent")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.save_weights(path)

    def load_agent(self, path=None):
        """Load model weights."""
        path = path or os.path.join(MODEL_DIR, "nandi_agent")
        if os.path.exists(path + ".index"):
            self.load_weights(path)
            return True
        return False
