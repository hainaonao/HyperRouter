r"""
Custom Gate
"""
from fmoe.gates.base_gate import BaseGate
from balancing_loss_free_state import BalancingLossFreeState

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import numpy as np 

__all__ = ['HyperRouterGate','CustomNaiveGate', 'CustomDropGate', 'CustomRandomGate', 'CustomRandomGate_Dense',
            'CustomDTSGate', 'CustomDTSRandomGate', 'CustomDTSGate_softmax', 'CustomDTSRandomGate_softmax',
            'CustomDenseGate', 'CustomHashGate', 'CustomNaiveGate_Balance', 'CustomNaiveGate_Attn',
            'BalancingLossFreeGate']


class HyperRouterGate(BaseGate):
    r"""
    HyperRouter Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2, hyper_size=256):
        super().__init__(num_expert, world_size)
        self.embedding = nn.Parameter(torch.randn([1, d_model], requires_grad=True).float().cuda())
        self.hypernet = nn.Sequential(
            nn.Linear(d_model, hyper_size),
            nn.ReLU(),
            nn.Linear(hyper_size, d_model * self.tot_expert+self.tot_expert)
        )
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.d_model = d_model

    def forward(self, inp, return_all_scores=False):
        self.hypernet_outputs = self.hypernet(self.embedding)[0]
        # Get the weight splice for these layers and shape to weight tensor
        weights_splice = self.hypernet_outputs.reshape([self.tot_expert, -1 ]) #(self.tot_expert, d_model+1)
        del self.gate.weight
        self.gate.weight = weights_splice[:, :-1]
        del self.gate.bias
        self.gate.bias = weights_splice[:, -1]
        
        gate = self.gate(inp)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate) # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class CustomNaiveGate(BaseGate):
    r"""
    Custom Naive Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate) # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


class CustomNaiveGate_Attn(BaseGate):
    r"""
    Naive Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate) # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


class CustomNaiveGate_Balance(BaseGate):
    r"""
    Naive Gate with Balance loss
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.loss = None

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate: (T, N) raw logits
        # gate_top_k_idx: (T, top_k) selected expert indices

        score = F.softmax(gate, dim=-1)
        T = gate_top_k_idx.shape[0]  # number of tokens

        # f_i: fraction of tokens routed to expert i (top-1 routing decision)
        top1 = gate_top_k_idx[:, 0]
        fraction_expert = torch.zeros(self.tot_expert, device=top1.device)
        fraction_expert.scatter_add_(
            0, top1, torch.ones(T, device=top1.device, dtype=torch.float))
        fraction_expert = fraction_expert / T

        # P_i: mean softmax probability per expert
        prob_expert = score.mean(dim=0)

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate) # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        # Cache for external metric computation (e.g., fluctuation)
        self.last_top_k_idx = gate_top_k_idx.detach()

        self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


class CustomHashGate(BaseGate):

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k

    def forward(self, inp, return_all_scores=False):

        if not hasattr(self, 'hash_gate'):
            # generate hash gate
            print('Generate Hash Mapping')
            token_num = inp.shape[0]
            self.register_buffer('hash_gate', torch.rand(token_num, self.tot_expert).to(inp.device))
            print(self.hash_gate.shape)
        else:
            if self.hash_gate.shape[0] != inp.shape[0]:
                if not hasattr(self, 'hash_gate_v2'):
                    print('Generate New Hash Mapping v2')
                    token_num = inp.shape[0]
                    self.register_buffer('hash_gate_v2', torch.rand(token_num, self.tot_expert).to(inp.device))
                    print(self.hash_gate_v2.shape)
                else:
                    if self.hash_gate_v2.shape[0] != inp.shape[0]:
                        if not hasattr(self, 'hash_gate_v3'):
                            print('Generate New Hash Mapping v3')
                            token_num = inp.shape[0]
                            self.register_buffer('hash_gate_v3', torch.rand(token_num, self.tot_expert).to(inp.device))
                            print(self.hash_gate_v3.shape)
                        else:
                            if self.hash_gate_v3.shape[0] != inp.shape[0]:
                                if not hasattr(self, 'hash_gate_v4'):
                                    print('Generate New Hash Mapping v4')
                                    token_num = inp.shape[0]
                                    self.register_buffer('hash_gate_v4', torch.rand(token_num, self.tot_expert).to(inp.device))
                                    print(self.hash_gate_v4.shape)
                                else:
                                    if self.hash_gate_v4.shape[0] != inp.shape[0]:
                                        print('Generate New Hash Mapping v5')
                                        token_num = inp.shape[0]
                                        self.register_buffer('hash_gate_v5', torch.rand(token_num, self.tot_expert).to(inp.device))
                                        print(self.hash_gate_v5.shape)

        if inp.shape[0] == self.hash_gate.shape[0]:
            gate = self.hash_gate
        elif inp.shape[0] == self.hash_gate_v2.shape[0]:
            gate = self.hash_gate_v2
        elif inp.shape[0] == self.hash_gate_v3.shape[0]:
            gate = self.hash_gate_v3
        elif inp.shape[0] == self.hash_gate_v4.shape[0]:
            gate = self.hash_gate_v4
        elif inp.shape[0] == self.hash_gate_v5.shape[0]:
            gate = self.hash_gate_v5
        else:
            assert False

        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_top_k_val = torch.ones_like(gate_top_k_val)
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score




class CustomDropGate(BaseGate):
    r"""
    Dropout Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.training:
            gate = self.dropout(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate) # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class CustomRandomGate(BaseGate):
    r"""
    Random Assign Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        # random gate uniform distribution
        gate = torch.rand_like(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class CustomRandomGate_Dense(BaseGate):
    r"""
    Random Assign Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        # random gate uniform distribution
        gate = torch.ones_like(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


# Dense to Sparse
class CustomDTSGate(BaseGate):
    r"""
    Dense to Sparse Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

        self.temperature = 1
        self.threshold = 0.001
        self.sum_top_k = 0
        self.forward_n = 0
        self.dynamic_top_k = top_k

    def _sample_gumbel(self, tensor, eps=1e-10):
        U = torch.rand_like(tensor).uniform_()
        return - torch.log(eps - torch.log(U + eps))

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.training:
            # dts
            gumber_noise = self._sample_gumbel(gate)
            gate_noise = (gate + gumber_noise) / self.temperature
            gate_noise = F.softmax(gate_noise, dim=-1)

            # calculate top-k number 
            enable_gate_number = gate_noise.gt(self.threshold).sum(dim=-1)
            dynamic_top_k = enable_gate_number.float().mean().int().item()
            self.dynamic_top_k = max(self.top_k, dynamic_top_k)

            self.forward_n += 1
            self.sum_top_k += self.dynamic_top_k

            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate_noise, k=self.dynamic_top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_score = gate_top_k_val.view(-1, self.dynamic_top_k)

        else:
            self.dynamic_top_k = self.top_k
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
            gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class CustomDTSRandomGate(BaseGate):
    r"""
    Dense to Sparse Gate Random Assign
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

        self.temperature = 1
        self.threshold = 0.001
        self.sum_top_k = 0
        self.forward_n = 0
        self.dynamic_top_k = top_k

    def _sample_gumbel(self, tensor, eps=1e-10):
        U = torch.rand_like(tensor).uniform_()
        return - torch.log(eps - torch.log(U + eps))

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)
        gate = torch.rand_like(gate)

        if self.training:
            # dts
            gumber_noise = self._sample_gumbel(gate)
            gate_noise = (gate + gumber_noise) / self.temperature
            gate_noise = F.softmax(gate_noise, dim=-1)

            # calculate top-k number 
            enable_gate_number = gate_noise.gt(self.threshold).sum(dim=-1)
            dynamic_top_k = enable_gate_number.float().mean().int().item()
            self.dynamic_top_k = max(self.top_k, dynamic_top_k)

            self.forward_n += 1
            self.sum_top_k += self.dynamic_top_k

            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate_noise, k=self.dynamic_top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_score = gate_top_k_val.view(-1, self.dynamic_top_k)

        else:
            self.dynamic_top_k = self.top_k
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
            gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class CustomDTSGate_softmax(BaseGate):
    r"""
    Dense to Sparse Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

        self.temperature = 1
        self.threshold = 0.001
        self.sum_top_k = 0
        self.forward_n = 0
        self.dynamic_top_k = top_k

    def _sample_gumbel(self, tensor, eps=1e-10):
        U = torch.rand_like(tensor).uniform_()
        return - torch.log(eps - torch.log(U + eps))

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.training:
            # dts
            gumber_noise = self._sample_gumbel(gate)
            gate_noise = (gate + gumber_noise) / self.temperature
            gate_noise = F.softmax(gate_noise, dim=-1)

            # calculate top-k number 
            enable_gate_number = gate_noise.gt(self.threshold).sum(dim=-1)
            dynamic_top_k = enable_gate_number.float().mean().int().item()
            self.dynamic_top_k = max(self.top_k, dynamic_top_k)

            self.forward_n += 1
            self.sum_top_k += self.dynamic_top_k

            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate_noise, k=self.dynamic_top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_score = gate_top_k_val.view(-1, self.dynamic_top_k)

        else:
            gate = F.softmax(gate, dim=-1)
            self.dynamic_top_k = self.top_k
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_score = gate_top_k_val.view(-1, self.top_k)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class CustomDTSRandomGate_softmax(BaseGate):
    r"""
    Dense to Sparse Gate Random Assign
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

        self.temperature = 1
        self.threshold = 0.001
        self.sum_top_k = 0
        self.forward_n = 0
        self.dynamic_top_k = top_k

    def _sample_gumbel(self, tensor, eps=1e-10):
        U = torch.rand_like(tensor).uniform_()
        return - torch.log(eps - torch.log(U + eps))

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)
        gate = torch.rand_like(gate)

        if self.training:
            # dts
            gumber_noise = self._sample_gumbel(gate)
            gate_noise = (gate + gumber_noise) / self.temperature
            gate_noise = F.softmax(gate_noise, dim=-1)

            # calculate top-k number 
            enable_gate_number = gate_noise.gt(self.threshold).sum(dim=-1)
            dynamic_top_k = enable_gate_number.float().mean().int().item()
            self.dynamic_top_k = max(self.top_k, dynamic_top_k)

            self.forward_n += 1
            self.sum_top_k += self.dynamic_top_k

            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate_noise, k=self.dynamic_top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_score = gate_top_k_val.view(-1, self.dynamic_top_k)

        else:
            gate = F.softmax(gate, dim=-1)
            self.dynamic_top_k = self.top_k
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_score = gate_top_k_val.view(-1, self.top_k)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class CustomDenseGate(BaseGate):
    r"""
    Dense Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)
        repeat_shape = list(gate.shape[:-1])
        repeat_shape.append(1)

        gate_top_k_idx = torch.arange(self.tot_expert).repeat(repeat_shape).to(gate.device)

        gate_top_k_val = gate.view(-1, self.tot_expert)
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


class BalancingLossFreeGate(BaseGate):
    r"""
    Balancing-Loss-Free Gate.

    Replaces auxiliary balancing loss with memory-based expert balancing using
    two EMA signals:
      - RC_i: per-layer memory (EMA of router logits at layer i across steps)
      - RL:   cross-layer memory (EMA of mean router logits across all layers,
              finalized from the previous optimizer step)

    Final router logit is a convex combination:
      R = w_orig * R_original + w_rc * RC_i + w_rl * RL

    where [w_orig, w_rc, w_rl] are computed by a Highway-style MLP + softmax
    (ensuring non-negative weights that sum to 1 per expert dimension).

    R_original is trainable (has gradient). RC_i and RL are detached memory
    (no gradient flows through them).

    Cold-start handling:
      - Adam-style bias correction: RC_corrected = RC / (1 - d^t)
      - RL fallback: zero vector until at least one step has been finalized

    Step definition:
      - 1 step = 1 optimizer.step(), NOT 1 forward()
      - Supports gradient accumulation (multiple micro-batches per step)
      - Safe with gradient checkpointing (deterministic linear cancels
        double-count via mean)

    Args:
        d_model: Input feature dimension.
        num_expert: Number of experts per worker.
        world_size: Number of workers.
        top_k: Number of experts each token is routed to.
        shared_state: BalancingLossFreeState instance shared across all layers.
        layer_idx: Index of this layer (0-based).
        highway_hidden: Hidden dimension for the Highway MLP. Default: min(d_model//4, 64).
    """

    def __init__(
        self,
        d_model: int,
        num_expert: int,
        world_size: int,
        top_k: int = 2,
        shared_state: BalancingLossFreeState = None,
        layer_idx: int = 0,
        highway_hidden: int = None,
    ):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.d_model = d_model
        self.layer_idx = layer_idx

        # --- Shared state ---
        if shared_state is None:
            # Fallback: create a private state (not recommended for multi-layer)
            shared_state = BalancingLossFreeState(
                num_experts=self.tot_expert, decay=0.999
            )
        self.shared_state = shared_state
        self.shared_state.register_layer(layer_idx)

        # --- Highway-style MLP ---
        # Outputs 3*E values, reshaped to (3, E), softmax along dim=0
        # → [w_orig, w_rc, w_rl] per expert, guaranteed to sum to 1
        if highway_hidden is None:
            highway_hidden = min(d_model // 4, 64)
        self.highway_mlp = nn.Sequential(
            nn.Linear(d_model, highway_hidden),
            nn.ReLU(),
            nn.Linear(highway_hidden, 3 * self.tot_expert),
        )

    def forward(self, inp, return_all_scores=False):
        """
        Args:
            inp: Input tensor of shape (num_tokens, d_model).
            return_all_scores: If True, also return raw gate logits.

        Returns:
            gate_top_k_idx: (num_tokens, top_k) indices of selected experts.
            gate_score: (num_tokens, top_k) softmax scores for selected experts.
            gate (optional): (num_tokens, E) raw blended gate logits.
        """
        # 1. Trainable router logit (has gradient)
        r_original = self.gate(inp)  # (num_tokens, E)

        if self.training:
            # 2. Pool tokens to get a single representation for Highway weights
            x_pooled = inp.mean(dim=0)  # (d_model,)

            # 3. Highway MLP → convex combination weights
            weights_raw = self.highway_mlp(x_pooled)  # (3*E,)
            weights = F.softmax(
                weights_raw.view(3, self.tot_expert), dim=0
            )  # (3, E), each column sums to 1
            w_orig = weights[0]  # (E,)
            w_rc = weights[1]    # (E,)
            w_rl = weights[2]    # (E,)

            # 4. Get memory signals (detached, no gradient)
            device = inp.device
            rc_i = self.shared_state.get_rc(self.layer_idx, device)  # (E,)
            rl = self.shared_state.get_rl_finalized(device)          # (E,)

            # 5. Blend: R = w_orig * R_original + w_rc * RC_i + w_rl * RL
            # w_orig broadcasts over (num_tokens, E), RC and RL broadcast from (E,)
            gate = w_orig * r_original + w_rc * rc_i + w_rl * rl  # (num_tokens, E)

            # 6. Accumulate for shared state (detached logit mean)
            self.shared_state.accumulate(
                self.layer_idx, r_original.detach().mean(dim=0)
            )
        else:
            # Inference: use only the trainable router (no balancing needed)
            gate = r_original

        # --- Top-k selection ---
        if self.dense_moe_flag:
            gate = torch.ones_like(gate)
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        # Cache for external metric computation (e.g., fluctuation)
        self.last_top_k_idx = gate_top_k_idx.detach()

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score
