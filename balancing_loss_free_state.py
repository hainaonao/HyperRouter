r"""
Shared state for Balancing-Loss-Free Gate.

This module manages the EMA memory buffers (RC per-layer, RL cross-layer)
that are shared across all BalancingLossFreeGate instances (one per MoE layer).

Key design decisions:
- 1 step = 1 optimizer.step(), NOT 1 forward()
- Micro-batch accumulation: sum + count, EMA applied once per step
- Bias correction (Adam-style) for cold start
- RL uses finalized value from previous step for all layers (positional fairness)
- Gradient checkpointing safe: deterministic linear → double-count cancels via mean
"""

import torch


class BalancingLossFreeState:
    """Shared state across all BalancingLossFreeGate instances.

    Manages per-layer EMA (RC) and cross-layer EMA (RL) memory buffers,
    micro-batch accumulation, bias correction, and step lifecycle.

    Usage:
        state = BalancingLossFreeState(num_experts=8, decay=0.999)
        # Gates register themselves:
        gate_0 = BalancingLossFreeGate(..., shared_state=state, layer_idx=0)
        gate_1 = BalancingLossFreeGate(..., shared_state=state, layer_idx=1)
        # Training loop:
        for step in range(num_steps):
            for micro_batch in micro_batches:
                loss = model(micro_batch)
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            state.finalize_step()  # MUST call after optimizer.step()
    """

    def __init__(self, num_experts: int, decay: float = 0.999):
        """
        Args:
            num_experts: Total number of experts E (= num_expert * world_size).
            decay: EMA decay factor d. Higher = smoother memory.
        """
        self.num_experts = num_experts
        self.decay = decay

        # --- Per-layer RC EMA ---
        # Registered layers: set of layer indices
        self._registered_layers: set = set()
        # EMA buffers: layer_idx -> Tensor(E,)
        self._rc_ema: dict[int, torch.Tensor] = {}
        # Step count per layer (for bias correction)
        self._rc_step_count: dict[int, int] = {}

        # --- Cross-layer RL EMA ---
        self._rl_ema: torch.Tensor | None = None  # Tensor(E,)
        self._rl_step_count: int = 0
        # Finalized RL from previous step — used by all layers in current step
        self._rl_finalized: torch.Tensor | None = None  # Tensor(E,)

        # --- Micro-batch accumulation (within one optimizer step) ---
        # layer_idx -> Tensor(E,): running sum of logit means
        self._accum_sum: dict[int, torch.Tensor] = {}
        # layer_idx -> int: number of forward calls accumulated
        self._accum_count: dict[int, int] = {}

        # --- Step tracking ---
        self._current_step: int = 0

    @property
    def num_layers(self) -> int:
        return len(self._registered_layers)

    def register_layer(self, layer_idx: int) -> None:
        """Register a layer index. Called once per gate during __init__."""
        if layer_idx in self._registered_layers:
            return  # idempotent
        self._registered_layers.add(layer_idx)
        self._rc_step_count[layer_idx] = 0

    def accumulate(self, layer_idx: int, logits_mean: torch.Tensor) -> None:
        """Accumulate router logit mean for a micro-batch forward pass.

        Args:
            layer_idx: Index of the layer calling this.
            logits_mean: Mean of router logits over tokens, shape (E,).
                         Must be detached (no gradient).
        """
        assert layer_idx in self._registered_layers, (
            f"Layer {layer_idx} not registered. Call register_layer() first."
        )
        logits_mean = logits_mean.detach()

        if layer_idx not in self._accum_sum:
            self._accum_sum[layer_idx] = logits_mean.clone()
            self._accum_count[layer_idx] = 1
        else:
            self._accum_sum[layer_idx] += logits_mean
            self._accum_count[layer_idx] += 1

    def get_rc(self, layer_idx: int, device: torch.device) -> torch.Tensor:
        """Get bias-corrected RC_i for a layer. Returns zeros if not yet available.

        Args:
            layer_idx: Layer index.
            device: Device to place the tensor on.

        Returns:
            Tensor of shape (E,), detached, no gradient.
        """
        step_count = self._rc_step_count.get(layer_idx, 0)
        if step_count == 0:
            # Cold start: no EMA yet → zero vector
            return torch.zeros(self.num_experts, device=device)

        rc_raw = self._rc_ema[layer_idx].to(device)
        # Bias correction: RC / (1 - d^t)
        correction = 1.0 - self.decay ** step_count
        return rc_raw / correction

    def get_rl_finalized(self, device: torch.device) -> torch.Tensor:
        """Get finalized RL from the previous step. Returns zeros if not yet available.

        This is the SAME value for ALL layers within a step (positional fairness).

        Args:
            device: Device to place the tensor on.

        Returns:
            Tensor of shape (E,), detached, no gradient.
        """
        if self._rl_finalized is None:
            # Cold start: no finalized RL yet → zero vector
            return torch.zeros(self.num_experts, device=device)
        return self._rl_finalized.to(device)

    def finalize_step(self) -> None:
        """Finalize the current optimizer step.

        This MUST be called exactly once after each optimizer.step().
        It performs:
        1. Compute per-layer mean from accumulated micro-batches
        2. Update RC EMA for each layer
        3. Compute cross-layer step_mean
        4. Update RL EMA
        5. Freeze RL as finalized for the next step
        6. Reset accumulators

        Layers that did not accumulate any data in this step are skipped
        (this can happen if a layer was not called, e.g., during partial eval).
        """
        self._current_step += 1

        if not self._accum_sum:
            # No layers accumulated anything — nothing to do
            return

        # Determine device from any accumulated tensor
        any_device = next(iter(self._accum_sum.values())).device

        # Step 1 & 2: Per-layer EMA update
        layer_means = {}
        for layer_idx in sorted(self._accum_sum.keys()):
            count = self._accum_count[layer_idx]
            r_i_mean = self._accum_sum[layer_idx] / count  # mean over micro-batches
            layer_means[layer_idx] = r_i_mean

            # EMA update: RC_i = d * RC_i + (1-d) * R_i_mean
            if layer_idx not in self._rc_ema:
                self._rc_ema[layer_idx] = torch.zeros(
                    self.num_experts, device=any_device
                )
            self._rc_ema[layer_idx] = (
                self.decay * self._rc_ema[layer_idx]
                + (1.0 - self.decay) * r_i_mean
            )
            self._rc_step_count[layer_idx] = (
                self._rc_step_count.get(layer_idx, 0) + 1
            )

        # Step 3: Cross-layer step_mean = (1/L) * Σ R_i_mean
        if layer_means:
            stacked = torch.stack(list(layer_means.values()), dim=0)  # (L_active, E)
            step_mean = stacked.mean(dim=0)  # (E,)

            # Step 4: RL EMA update
            if self._rl_ema is None:
                self._rl_ema = torch.zeros(self.num_experts, device=any_device)
            self._rl_ema = (
                self.decay * self._rl_ema + (1.0 - self.decay) * step_mean
            )
            self._rl_step_count += 1

            # Step 5: Bias-correct and freeze as finalized RL
            correction = 1.0 - self.decay ** self._rl_step_count
            self._rl_finalized = (self._rl_ema / correction).clone()

        # Step 6: Reset accumulators
        self._accum_sum.clear()
        self._accum_count.clear()

    def reset(self) -> None:
        """Full reset of all state. Useful for testing or re-initialization."""
        self._rc_ema.clear()
        self._rc_step_count = {idx: 0 for idx in self._registered_layers}
        self._rl_ema = None
        self._rl_step_count = 0
        self._rl_finalized = None
        self._accum_sum.clear()
        self._accum_count.clear()
        self._current_step = 0
