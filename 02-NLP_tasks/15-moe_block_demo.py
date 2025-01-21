import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN

from moe_config import MixtralConfig
from typing import Callable, List, Optional, Tuple, Union

class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = config.router_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim) # [500,2048]=[5,10,2048]
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states) # [500,8]

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float) # [500,8]
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1) # [500,2] [500,2] topk的两个权重,以及对应的下标  每个字选择两个专家
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True) # 所选择的这两个权重进行归一化
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype) # [500,2]

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0) # [num_expert, top_k, bs*hs]=[8,2,500]

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx] # llama MLP
            idx, top_x = torch.where(expert_mask[expert_idx])  # expert_mask=[topk,bs*hs] [2,500] idx标识topk_i的专家，top_x表示具体的哪些专家

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None] # 模型并行的计算方式，对于每句话来说，都要8选2，那么对于每个专家来说都会选择一些句子

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


if __name__ == "__main__":
    
    mixtralconfig = MixtralConfig(
        vocab_size=3200, hidden_size=2048, intermediate_size=14336//2
    )
    MixtralSparsemoeblock = MixtralSparseMoeBlock(mixtralconfig)

    batch_size, seq_length, hidden_dim = 10,50,2048

    hidden_stats = torch.randn(size=(batch_size, seq_length, hidden_dim))

    final_hidden_stats, router_logits = MixtralSparsemoeblock(hidden_stats)

    print(final_hidden_stats.shape)