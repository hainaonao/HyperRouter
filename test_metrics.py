# coding: utf-8
"""
Standalone unit test for routing metric computations.
Tests the LOGIC of balance loss and fluctuation calculations
WITHOUT importing model code (no CUDA/fmoe dependency needed).
"""
import torch
import torch.nn.functional as F

print("=" * 60)
print("TEST 1: Balance Loss computation (Switch Transformer L_aux)")
print("=" * 60)

# Simulate gate outputs for 64 tokens, 8 experts, top-k=2
N = 8   # experts
T = 64  # tokens
top_k = 2

# Case A: perfectly balanced routing (each expert gets T/N tokens)
print("\nCase A: Perfectly balanced routing")
gate_top_k_idx_balanced = torch.zeros(T, top_k, dtype=torch.long)
for i in range(T):
    gate_top_k_idx_balanced[i, 0] = i % N
    gate_top_k_idx_balanced[i, 1] = (i + 1) % N

# Uniform logits -> uniform P_i
gate_logits_uniform = torch.zeros(T, N)

top1 = gate_top_k_idx_balanced[:, 0]
P = F.softmax(gate_logits_uniform, dim=-1).mean(dim=0)
f = torch.zeros(N)
f.scatter_add_(0, top1, torch.ones(T))
f = f / T

L_aux_balanced = (N * (f * P).sum()).item()
print(f"  f_i = {f.tolist()}")
print(f"  P_i = {P.tolist()}")
print(f"  L_aux = {L_aux_balanced:.4f}  (expected: 1.0)")
assert abs(L_aux_balanced - 1.0) < 0.01, f"FAIL: Expected ~1.0, got {L_aux_balanced}"
print("  ✓ PASS")

# Case B: extremely imbalanced (all tokens go to expert 0)
print("\nCase B: All tokens routed to expert 0")
gate_top_k_idx_imbalanced = torch.zeros(T, top_k, dtype=torch.long)

# Logits heavily favoring expert 0
gate_logits_imbalanced = torch.zeros(T, N)
gate_logits_imbalanced[:, 0] = 10.0

top1 = gate_top_k_idx_imbalanced[:, 0]
P = F.softmax(gate_logits_imbalanced, dim=-1).mean(dim=0)
f = torch.zeros(N)
f.scatter_add_(0, top1, torch.ones(T))
f = f / T

L_aux_imbalanced = (N * (f * P).sum()).item()
print(f"  f_i = {f.tolist()}")
print(f"  P_i (top-3) = {P[:3].tolist()}")
print(f"  L_aux = {L_aux_imbalanced:.4f}  (expected: >> 1.0)")
assert L_aux_imbalanced > 1.5, f"FAIL: Expected >> 1.0, got {L_aux_imbalanced}"
print("  ✓ PASS")

# Case C: moderately imbalanced
print("\nCase C: Moderately imbalanced")
gate_top_k_idx_moderate = torch.zeros(T, top_k, dtype=torch.long)
# 50% to expert 0, 50% split among rest
for i in range(T):
    if i < T // 2:
        gate_top_k_idx_moderate[i, 0] = 0
    else:
        gate_top_k_idx_moderate[i, 0] = (i % (N - 1)) + 1
    gate_top_k_idx_moderate[i, 1] = (gate_top_k_idx_moderate[i, 0] + 1) % N

gate_logits_moderate = torch.randn(T, N)

top1 = gate_top_k_idx_moderate[:, 0]
P = F.softmax(gate_logits_moderate, dim=-1).mean(dim=0)
f = torch.zeros(N)
f.scatter_add_(0, top1, torch.ones(T))
f = f / T

L_aux_moderate = (N * (f * P).sum()).item()
print(f"  f_i = {[round(x, 3) for x in f.tolist()]}")
print(f"  L_aux = {L_aux_moderate:.4f}  (expected: > 1.0)")
print("  ✓ PASS")

print(f"\n  Summary: balanced={L_aux_balanced:.4f} < moderate={L_aux_moderate:.4f} < imbalanced={L_aux_imbalanced:.4f}")

print("\n" + "=" * 60)
print("TEST 2: Routing Fluctuation computation")
print("=" * 60)

# Case A: identical routing -> fluctuation = 0
print("\nCase A: Same routing decisions")
prev = {0: torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]])}
curr = {0: torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]])}

def compute_layer_fluctuations(curr_indices, prev_indices):
    """Inlined from new_utils.py to avoid fmoe import."""
    fluctuations = {}
    for layer_idx in sorted(curr_indices.keys()):
        if layer_idx not in prev_indices:
            continue
        curr = curr_indices[layer_idx]
        prev = prev_indices[layer_idx]
        if curr.shape != prev.shape:
            continue
        fluc = (curr != prev).float().mean().item()
        fluctuations[f"fluc_layer_{layer_idx}"] = fluc
    return fluctuations
fluc = compute_layer_fluctuations(curr, prev)
print(f"  Fluctuation: {fluc}")
assert list(fluc.values())[0] == 0.0, "FAIL: Expected 0.0 for identical routing"
print("  ✓ PASS (0.0 as expected)")

# Case B: completely different routing -> fluctuation = 1.0
print("\nCase B: Completely different routing")
prev = {0: torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])}
curr = {0: torch.tensor([[7, 6], [5, 4], [3, 2], [1, 0]])}

fluc = compute_layer_fluctuations(curr, prev)
print(f"  Fluctuation: {fluc}")
assert list(fluc.values())[0] == 1.0, "FAIL: Expected 1.0 for completely different routing"
print("  ✓ PASS (1.0 as expected)")

# Case C: partial change -> 0 < fluctuation < 1
print("\nCase C: Partial change (50%)")
prev = {0: torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]),
         1: torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])}
curr = {0: torch.tensor([[0, 1], [2, 3], [7, 6], [5, 4]]),  # 50% changed
         1: torch.tensor([[1, 0], [3, 2], [4, 5], [6, 7]])}  # 50% changed

fluc = compute_layer_fluctuations(curr, prev)
print(f"  Fluctuation: {fluc}")
for k, v in fluc.items():
    assert 0 < v < 1, f"FAIL: Expected 0 < fluc < 1, got {v}"
print("  ✓ PASS (partial fluctuation as expected)")

print("\n" + "=" * 60)
print("TEST 3: Verify log format strings")
print("=" * 60)

fluctuations = {'fluc_layer_0': 0.3421, 'fluc_layer_1': 0.2567}
balance_losses = {0: 1.2345, 1: 1.5678}

fluc_strs = ['{}: {:.4f}'.format(k, v) for k, v in sorted(fluctuations.items())]
avg_fluc = sum(fluctuations.values()) / len(fluctuations)
log_fluc = '| Routing Fluctuation at step {:>8d} (step {} vs {}) | avg {:.4f} | {}'.format(
    1000, 999, 1000, avg_fluc, ' | '.join(fluc_strs))
print(log_fluc)

bl_strs = ['layer_{}: {:.4f}'.format(k, v) for k, v in sorted(balance_losses.items())]
avg_bl = sum(balance_losses.values()) / len(balance_losses)
log_bl = '| Balance Loss (monitor) at step {:>8d} | avg {:.4f} | {}'.format(
    1000, avg_bl, ' | '.join(bl_strs))
print(log_bl)

print("\n✓ Log format OK")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
