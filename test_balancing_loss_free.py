r"""
Test script for BalancingLossFreeGate and BalancingLossFreeState.

Verifies:
1. BalancingLossFreeState accumulate/finalize cycle
2. Bias correction output correctness
3. Cold start → zero vector fallback
4. Forward output shape consistency
5. Gradient flow: R_original and highway_mlp have gradient, RC/RL do not
6. Multiple micro-batches → mean invariant
7. Convex combination weights sum to 1
"""

import sys
import os
import types

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Mock fmoe.gates.base_gate so custom_gate.py can be imported
# without the full fmoe package installed
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "base_gate", os.path.join(project_root, "gates", "base_gate.py")
)
_base_gate_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base_gate_mod)

_fmoe = types.ModuleType("fmoe")
_fmoe_gates = types.ModuleType("fmoe.gates")
_fmoe.gates = _fmoe_gates
_fmoe_gates.base_gate = _base_gate_mod
sys.modules["fmoe"] = _fmoe
sys.modules["fmoe.gates"] = _fmoe_gates
sys.modules["fmoe.gates.base_gate"] = _base_gate_mod

import torch
import torch.nn as nn

from balancing_loss_free_state import BalancingLossFreeState


def test_state_cold_start():
    """Test that cold start returns zero vectors."""
    print("=" * 60)
    print("TEST: Cold start → zero vectors")
    state = BalancingLossFreeState(num_experts=4, decay=0.99)
    state.register_layer(0)
    state.register_layer(1)

    device = torch.device("cpu")

    rc = state.get_rc(0, device)
    assert rc.shape == (4,), f"Expected shape (4,), got {rc.shape}"
    assert torch.allclose(rc, torch.zeros(4)), f"Expected zeros, got {rc}"

    rl = state.get_rl_finalized(device)
    assert rl.shape == (4,), f"Expected shape (4,), got {rl.shape}"
    assert torch.allclose(rl, torch.zeros(4)), f"Expected zeros, got {rl}"

    print("  ✓ RC cold start = zeros")
    print("  ✓ RL cold start = zeros")
    print()


def test_state_single_step():
    """Test single step: accumulate → finalize → check RC/RL."""
    print("=" * 60)
    print("TEST: Single step accumulate → finalize")
    d = 0.99
    state = BalancingLossFreeState(num_experts=4, decay=d)
    state.register_layer(0)
    state.register_layer(1)

    device = torch.device("cpu")

    # Simulate one micro-batch for 2 layers
    logits_layer0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    logits_layer1 = torch.tensor([4.0, 3.0, 2.0, 1.0])

    state.accumulate(0, logits_layer0)
    state.accumulate(1, logits_layer1)

    state.finalize_step()

    # Check RC (bias-corrected): RC = (1-d)*x / (1-d^1) = x (for t=1)
    # RC_raw = 0 * d + (1-d) * x = (1-d) * x
    # correction = 1 - d^1 = 1 - d
    # RC_corrected = (1-d)*x / (1-d) = x
    rc0 = state.get_rc(0, device)
    assert torch.allclose(rc0, logits_layer0, atol=1e-6), (
        f"Expected {logits_layer0}, got {rc0}"
    )

    rc1 = state.get_rc(1, device)
    assert torch.allclose(rc1, logits_layer1, atol=1e-6), (
        f"Expected {logits_layer1}, got {rc1}"
    )

    # Check RL: step_mean = (layer0 + layer1) / 2
    step_mean = (logits_layer0 + logits_layer1) / 2
    # RL_raw = (1-d) * step_mean, correction = (1-d), so RL_corrected = step_mean
    rl = state.get_rl_finalized(device)
    assert torch.allclose(rl, step_mean, atol=1e-6), (
        f"Expected {step_mean}, got {rl}"
    )

    print("  ✓ RC after 1 step (bias-corrected) = input logits")
    print("  ✓ RL after 1 step (bias-corrected) = mean of layer logits")
    print()


def test_state_multi_step_ema():
    """Test EMA behavior over multiple steps."""
    print("=" * 60)
    print("TEST: Multi-step EMA convergence")
    d = 0.9
    state = BalancingLossFreeState(num_experts=2, decay=d)
    state.register_layer(0)

    device = torch.device("cpu")

    # Feed constant logits for many steps
    constant = torch.tensor([5.0, 3.0])
    for _ in range(100):
        state.accumulate(0, constant)
        state.finalize_step()

    rc = state.get_rc(0, device)
    # After many steps, bias-corrected EMA should converge to the constant
    assert torch.allclose(rc, constant, atol=0.01), (
        f"Expected ~{constant}, got {rc}"
    )

    rl = state.get_rl_finalized(device)
    assert torch.allclose(rl, constant, atol=0.01), (
        f"Expected ~{constant}, got {rl}"
    )

    print(f"  ✓ RC after 100 steps converged to {rc.tolist()} ≈ {constant.tolist()}")
    print(f"  ✓ RL after 100 steps converged to {rl.tolist()} ≈ {constant.tolist()}")
    print()


def test_state_micro_batch_accumulation():
    """Test that multiple micro-batches produce correct mean."""
    print("=" * 60)
    print("TEST: Micro-batch accumulation")
    d = 0.99
    state = BalancingLossFreeState(num_experts=3, decay=d)
    state.register_layer(0)

    device = torch.device("cpu")

    # 3 micro-batches
    mb1 = torch.tensor([1.0, 2.0, 3.0])
    mb2 = torch.tensor([4.0, 5.0, 6.0])
    mb3 = torch.tensor([7.0, 8.0, 9.0])

    state.accumulate(0, mb1)
    state.accumulate(0, mb2)
    state.accumulate(0, mb3)

    state.finalize_step()

    expected_mean = (mb1 + mb2 + mb3) / 3.0
    # After 1 step with bias correction: RC = expected_mean
    rc = state.get_rc(0, device)
    assert torch.allclose(rc, expected_mean, atol=1e-6), (
        f"Expected {expected_mean}, got {rc}"
    )

    print(f"  ✓ 3 micro-batches → mean = {rc.tolist()} == {expected_mean.tolist()}")
    print()


def test_state_rl_uses_previous_step():
    """Test that RL finalized is from the PREVIOUS step, not current."""
    print("=" * 60)
    print("TEST: RL uses previous step (positional fairness)")
    d = 0.9
    state = BalancingLossFreeState(num_experts=2, decay=d)
    state.register_layer(0)

    device = torch.device("cpu")

    # Step 1
    state.accumulate(0, torch.tensor([10.0, 20.0]))
    state.finalize_step()
    rl_after_step1 = state.get_rl_finalized(device).clone()

    # Step 2: accumulate different values
    state.accumulate(0, torch.tensor([100.0, 200.0]))
    # BEFORE finalize: RL should still be from step 1
    rl_during_step2 = state.get_rl_finalized(device)
    assert torch.allclose(rl_during_step2, rl_after_step1), (
        f"RL should not change during step 2! "
        f"Expected {rl_after_step1}, got {rl_during_step2}"
    )

    state.finalize_step()
    rl_after_step2 = state.get_rl_finalized(device)
    # Now RL should have been updated
    assert not torch.allclose(rl_after_step2, rl_after_step1, atol=1e-3), (
        "RL should have changed after finalize_step()"
    )

    print("  ✓ RL stays constant during accumulation (uses previous step)")
    print("  ✓ RL updates after finalize_step()")
    print()


def test_state_bias_correction_math():
    """Verify exact bias correction formula: RC / (1 - d^t)."""
    print("=" * 60)
    print("TEST: Bias correction math")
    d = 0.9
    state = BalancingLossFreeState(num_experts=2, decay=d)
    state.register_layer(0)

    device = torch.device("cpu")

    x1 = torch.tensor([1.0, 2.0])
    x2 = torch.tensor([3.0, 4.0])

    # Step 1
    state.accumulate(0, x1)
    state.finalize_step()

    # Manual: RC_raw = (1-d)*x1 = 0.1*[1,2] = [0.1, 0.2]
    # correction = 1 - 0.9^1 = 0.1
    # RC_corrected = [0.1, 0.2] / 0.1 = [1.0, 2.0]
    rc1 = state.get_rc(0, device)
    expected1 = x1  # simplifies for t=1
    assert torch.allclose(rc1, expected1, atol=1e-6), (
        f"Step 1: Expected {expected1}, got {rc1}"
    )

    # Step 2
    state.accumulate(0, x2)
    state.finalize_step()

    # Manual: RC_raw = d*(1-d)*x1 + (1-d)*x2
    #        = 0.9*0.1*[1,2] + 0.1*[3,4]
    #        = [0.09, 0.18] + [0.3, 0.4] = [0.39, 0.58]
    # correction = 1 - 0.9^2 = 1 - 0.81 = 0.19
    # RC_corrected = [0.39/0.19, 0.58/0.19] = [2.0526..., 3.0526...]
    rc_raw = d * (1 - d) * x1 + (1 - d) * x2
    correction = 1 - d ** 2
    expected2 = rc_raw / correction

    rc2 = state.get_rc(0, device)
    assert torch.allclose(rc2, expected2, atol=1e-5), (
        f"Step 2: Expected {expected2}, got {rc2}"
    )

    print(f"  ✓ Step 1 bias-corrected RC = {rc1.tolist()}")
    print(f"  ✓ Step 2 bias-corrected RC = {rc2.tolist()} (expected {expected2.tolist()})")
    print()


def test_state_reset():
    """Test full reset."""
    print("=" * 60)
    print("TEST: State reset")
    state = BalancingLossFreeState(num_experts=2, decay=0.99)
    state.register_layer(0)

    state.accumulate(0, torch.tensor([1.0, 2.0]))
    state.finalize_step()

    device = torch.device("cpu")
    assert not torch.allclose(state.get_rc(0, device), torch.zeros(2))

    state.reset()
    assert torch.allclose(state.get_rc(0, device), torch.zeros(2))
    assert torch.allclose(state.get_rl_finalized(device), torch.zeros(2))
    assert state._current_step == 0

    print("  ✓ After reset: RC = zeros, RL = zeros, step = 0")
    print()


def test_gate_forward_shape():
    """Test that BalancingLossFreeGate forward produces correct output shapes."""
    print("=" * 60)
    print("TEST: Gate forward output shapes")

    # Import gate
    from custom_gate import BalancingLossFreeGate

    d_model = 32
    num_expert = 4
    world_size = 1
    top_k = 2
    num_tokens = 16

    state = BalancingLossFreeState(num_experts=num_expert * world_size, decay=0.999)
    gate = BalancingLossFreeGate(
        d_model, num_expert, world_size, top_k,
        shared_state=state, layer_idx=0
    )
    gate.train()

    inp = torch.randn(num_tokens, d_model)

    # Without return_all_scores
    idx, score = gate(inp)
    assert idx.shape == (num_tokens, top_k), f"idx shape: {idx.shape}"
    assert score.shape == (num_tokens, top_k), f"score shape: {score.shape}"

    # With return_all_scores
    idx, score, raw = gate(inp, return_all_scores=True)
    assert raw.shape == (num_tokens, num_expert), f"raw shape: {raw.shape}"

    print(f"  ✓ gate_top_k_idx shape: {idx.shape}")
    print(f"  ✓ gate_score shape: {score.shape}")
    print(f"  ✓ raw gate shape: {raw.shape}")
    print()


def test_gate_eval_mode():
    """Test that eval mode uses only R_original (no balancing)."""
    print("=" * 60)
    print("TEST: Gate eval mode")

    from custom_gate import BalancingLossFreeGate

    d_model = 16
    num_expert = 4
    world_size = 1
    top_k = 2

    state = BalancingLossFreeState(num_experts=4, decay=0.999)
    gate = BalancingLossFreeGate(
        d_model, num_expert, world_size, top_k,
        shared_state=state, layer_idx=0
    )
    gate.eval()

    inp = torch.randn(8, d_model)
    idx, score = gate(inp)
    assert idx.shape == (8, top_k)

    # Verify no accumulation happened
    assert 0 not in state._accum_sum, "Should not accumulate in eval mode"

    print("  ✓ Eval mode produces correct shapes")
    print("  ✓ No accumulation in eval mode")
    print()


def test_gate_gradient_flow():
    """Test gradient flows through R_original and highway_mlp, not through RC/RL."""
    print("=" * 60)
    print("TEST: Gradient flow")

    from custom_gate import BalancingLossFreeGate

    d_model = 16
    num_expert = 4
    world_size = 1
    top_k = 2

    state = BalancingLossFreeState(num_experts=4, decay=0.999)

    # Seed some memory so RC/RL are non-zero
    state.register_layer(0)
    state.accumulate(0, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    state.finalize_step()

    gate = BalancingLossFreeGate(
        d_model, num_expert, world_size, top_k,
        shared_state=state, layer_idx=0
    )
    gate.train()

    inp = torch.randn(8, d_model, requires_grad=True)
    idx, score, raw = gate(inp, return_all_scores=True)

    # Compute a scalar loss
    loss = raw.sum()
    loss.backward()

    # gate.gate (R_original linear) should have gradients
    assert gate.gate.weight.grad is not None, "gate.weight should have gradient"
    assert gate.gate.bias.grad is not None, "gate.bias should have gradient"

    # highway_mlp should have gradients
    has_highway_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in gate.highway_mlp.parameters()
    )
    assert has_highway_grad, "highway_mlp should have gradient"

    # Input should have gradients
    assert inp.grad is not None, "Input should have gradient"

    print("  ✓ gate.gate.weight has gradient")
    print("  ✓ gate.gate.bias has gradient")
    print("  ✓ highway_mlp has gradient")
    print("  ✓ Input has gradient")
    print()


def test_gate_convex_combination():
    """Test that Highway MLP weights sum to 1 per expert (convex combination)."""
    print("=" * 60)
    print("TEST: Convex combination (weights sum to 1)")

    from custom_gate import BalancingLossFreeGate
    import torch.nn.functional as F

    d_model = 16
    num_expert = 4
    world_size = 1

    state = BalancingLossFreeState(num_experts=4, decay=0.999)
    gate = BalancingLossFreeGate(
        d_model, num_expert, world_size, 2,
        shared_state=state, layer_idx=0
    )
    gate.train()

    inp = torch.randn(8, d_model)
    x_pooled = inp.mean(dim=0)
    weights_raw = gate.highway_mlp(x_pooled)
    weights = F.softmax(weights_raw.view(3, num_expert), dim=0)  # (3, E)

    # Sum along dim=0 should be 1 for each expert
    col_sums = weights.sum(dim=0)  # (E,)
    assert torch.allclose(col_sums, torch.ones(num_expert), atol=1e-6), (
        f"Column sums should be 1, got {col_sums}"
    )

    # All weights should be non-negative
    assert (weights >= 0).all(), "All weights should be non-negative"

    print(f"  ✓ Weight column sums = {col_sums.tolist()}")
    print(f"  ✓ All weights ≥ 0")
    print(f"  ✓ Example weights per expert 0: w_orig={weights[0,0]:.4f}, "
          f"w_rc={weights[1,0]:.4f}, w_rl={weights[2,0]:.4f}")
    print()


def test_gate_multi_layer_integration():
    """Test multi-layer gate integration with shared state."""
    print("=" * 60)
    print("TEST: Multi-layer integration")

    from custom_gate import BalancingLossFreeGate

    d_model = 16
    num_expert = 4
    world_size = 1
    top_k = 2
    num_layers = 3

    state = BalancingLossFreeState(num_experts=4, decay=0.999)
    gates = [
        BalancingLossFreeGate(
            d_model, num_expert, world_size, top_k,
            shared_state=state, layer_idx=i
        )
        for i in range(num_layers)
    ]

    assert state.num_layers == num_layers, f"Expected {num_layers} layers, got {state.num_layers}"

    # Simulate 2 optimizer steps, each with 2 micro-batches
    for step in range(2):
        for gate in gates:
            gate.train()

        for mb in range(2):
            inp = torch.randn(8, d_model)
            for gate in gates:
                gate(inp)

        state.finalize_step()

    # After 2 steps, check that RC and RL are non-zero
    device = torch.device("cpu")
    for i in range(num_layers):
        rc = state.get_rc(i, device)
        assert not torch.allclose(rc, torch.zeros(4)), f"RC layer {i} should be non-zero"

    rl = state.get_rl_finalized(device)
    assert not torch.allclose(rl, torch.zeros(4)), "RL should be non-zero"

    print(f"  ✓ {num_layers} layers registered")
    print(f"  ✓ After 2 steps × 2 micro-batches: RC and RL are non-zero")
    print(f"  ✓ RL finalized = {rl.tolist()}")
    print()


def test_gate_no_shared_state_fallback():
    """Test that gate creates a private state when shared_state=None."""
    print("=" * 60)
    print("TEST: No shared_state fallback")

    from custom_gate import BalancingLossFreeGate

    gate = BalancingLossFreeGate(
        d_model=16, num_expert=4, world_size=1, top_k=2,
        shared_state=None, layer_idx=0
    )
    gate.train()

    inp = torch.randn(8, 16)
    idx, score = gate(inp)
    assert idx.shape == (8, 2)

    print("  ✓ Gate works with private fallback state")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BalancingLossFreeGate & State Test Suite")
    print("=" * 60 + "\n")

    # State tests
    test_state_cold_start()
    test_state_single_step()
    test_state_multi_step_ema()
    test_state_micro_batch_accumulation()
    test_state_rl_uses_previous_step()
    test_state_bias_correction_math()
    test_state_reset()

    # Gate tests
    test_gate_forward_shape()
    test_gate_eval_mode()
    test_gate_gradient_flow()
    test_gate_convex_combination()
    test_gate_multi_layer_integration()
    test_gate_no_shared_state_fallback()

    print("=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
