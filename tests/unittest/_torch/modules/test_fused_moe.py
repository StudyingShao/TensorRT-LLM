from itertools import product
from typing import Dict, List, Optional

import pytest
import triton
import prettytable as pt
import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn
# from utils.util import (skip_neither_ada_nor_hopper_unittest,
#                         skip_pre_blackwell, skip_pre_hopper)

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import (BaseMoeRoutingMethod,
                                                   DefaultMoeRoutingMethod,
                                                   FusedMoE,
                                                   RenormalizeMoeRoutingMethod)
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


# @skip_pre_hopper
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_fp8_tensor_scaling(
    dtype,
    SEQ_LEN = 4,
    HIDDEN_SIZE = 64,
    INTERMEDIATE_SIZE = 32,
    NUM_EXPERTS = 3,
    TOP_K = 2):

    # SEQ_LEN = 4
    # HIDDEN_SIZE = 64
    # INTERMEDIATE_SIZE = 32
    # NUM_EXPERTS = 3
    # TOP_K = 2
    routing_method = DefaultMoeRoutingMethod(top_k=TOP_K)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    _, x_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
    x_scale = x_scale.float().squeeze()
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()

    weights = {}
    for expert_id in range(NUM_EXPERTS):
        w1_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda()
        w2_weight = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                dtype=dtype).cuda()
        w3_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda()

        w1_weight_fp8, w1_weight_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
            w1_weight)
        w1_weight_fp8 = w1_weight_fp8.view(torch.float8_e4m3fn).cuda()

        w2_weight_fp8, w2_weight_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
            w2_weight)
        w2_weight_fp8 = w2_weight_fp8.view(torch.float8_e4m3fn).cuda()

        w3_weight_fp8, w3_weight_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
            w3_weight)
        w3_weight_fp8 = w3_weight_fp8.view(torch.float8_e4m3fn).cuda()

        w1_input_scale = x_scale.cuda()
        w2_input_scale = x_scale.cuda()
        w3_input_scale = x_scale.cuda()

        weights[f"{expert_id}.w1.weight"] = w1_weight_fp8
        weights[f"{expert_id}.w2.weight"] = w2_weight_fp8
        weights[f"{expert_id}.w3.weight"] = w3_weight_fp8
        weights[f"{expert_id}.w1.weight_scale"] = w1_weight_scale.float()
        weights[f"{expert_id}.w2.weight_scale"] = w2_weight_scale.float()
        weights[f"{expert_id}.w3.weight_scale"] = w3_weight_scale.float()
        weights[f"{expert_id}.w1.input_scale"] = w1_input_scale
        weights[f"{expert_id}.w2.input_scale"] = w2_input_scale
        weights[f"{expert_id}.w3.input_scale"] = w3_input_scale

    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    fused_moe = FusedMoE(num_experts=NUM_EXPERTS,
                         routing_method=routing_method,
                         hidden_size=HIDDEN_SIZE,
                         intermediate_size=INTERMEDIATE_SIZE,
                         dtype=dtype,
                         reduce_results=False,
                         model_config=ModelConfig(quant_config=quant_config))
    fused_moe.cuda()
    fused_moe.load_weights([weights])

    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        fused_moe.forward(x, router_logits)

    ref_fused_moe = RefGatedMLPFusedMoE(
        num_experts=NUM_EXPERTS,
        routing_method=routing_method,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=dtype,
        model_config=ModelConfig(quant_config=quant_config))
    ref_fused_moe.load_weights([weights])
    ref_fused_moe.cuda()
    with torch.inference_mode():
        nvtx.range_push("fp8 tensor")
        output = fused_moe.forward(x, router_logits)
        nvtx.range_pop()
        ref_output = ref_fused_moe.forward(x, router_logits)

    # compare
    torch.cuda.synchronize()
    # torch.testing.assert_close(output, ref_output, rtol=0.1, atol=0.5)


def test_fused_moe_fp8_block_scaling(
    dtype,
    SEQ_LEN = 4,
    HIDDEN_SIZE = 64,
    INTERMEDIATE_SIZE = 32,
    NUM_EXPERTS = 3,
    TOP_K = 2):

    # SEQ_LEN = 4
    # HIDDEN_SIZE = 64
    # INTERMEDIATE_SIZE = 32
    # NUM_EXPERTS = 3
    # TOP_K = 2
    routing_method = DefaultMoeRoutingMethod(top_k=TOP_K)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()

    weights = {}
    for expert_id in range(NUM_EXPERTS):
        w1_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda().to(torch.float8_e4m3fn)
        w2_weight = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                dtype=dtype).cuda().to(torch.float8_e4m3fn)
        w3_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda().to(torch.float8_e4m3fn)

        w1_weight_scale = torch.randn(
                    (w1_weight.shape[0] //128,
                    w1_weight.shape[1] // 128),
                    dtype=torch.float32,
                    device="cuda")
        w2_weight_scale = torch.randn(
                    (w2_weight.shape[0] // 128,
                    w2_weight.shape[1] // 128),
                    dtype=torch.float32,
                    device="cuda")
        w3_weight_scale = torch.randn(
                    (w3_weight.shape[0] // 128,
                    w3_weight.shape[1] // 128),
                    dtype=torch.float32,
                    device="cuda")

        weights[f"{expert_id}.w1.weight"] = w1_weight
        weights[f"{expert_id}.w2.weight"] = w2_weight
        weights[f"{expert_id}.w3.weight"] = w3_weight
        weights[f"{expert_id}.w1.weight_scale_inv"] = w1_weight_scale
        weights[f"{expert_id}.w2.weight_scale_inv"] = w2_weight_scale
        weights[f"{expert_id}.w3.weight_scale_inv"] = w3_weight_scale

    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    fused_moe = FusedMoE(num_experts=NUM_EXPERTS,
                         routing_method=routing_method,
                         hidden_size=HIDDEN_SIZE,
                         intermediate_size=INTERMEDIATE_SIZE,
                         dtype=dtype,
                         reduce_results=False,
                         model_config=ModelConfig(quant_config=quant_config))
    fused_moe.cuda()
    fused_moe.load_weights([weights])

    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        fused_moe.forward(x, router_logits)

    # ref_fused_moe = RefGatedMLPFusedMoE(
    #     num_experts=NUM_EXPERTS,
    #     routing_method=routing_method,
    #     hidden_size=HIDDEN_SIZE,
    #     intermediate_size=INTERMEDIATE_SIZE,
    #     dtype=dtype,
    #     model_config=ModelConfig(quant_config=quant_config))
    # ref_fused_moe.load_weights([weights])
    # ref_fused_moe.cuda()
    with torch.inference_mode():
        nvtx.range_push("fp8 block")
        output = fused_moe.forward(x, router_logits)
        nvtx.range_pop()
        # ref_output = ref_fused_moe.forward(x, router_logits)

    # compare
    torch.cuda.synchronize()
    # torch.testing.assert_close(output, ref_output, rtol=0.1, atol=0.5)

# @skip_neither_ada_nor_hopper_unittest
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_w4afp8(
    dtype,
    SEQ_LEN = 4,
    HIDDEN_SIZE = 768,
    INTERMEDIATE_SIZE = 640,
    SCALING_GROUP_SIZE = 128,
    NUM_EXPERTS = 3,
    TOP_K = 2):

    # SEQ_LEN = 4
    # HIDDEN_SIZE = 768
    # INTERMEDIATE_SIZE = 640
    # SCALING_GROUP_SIZE = 128
    # NUM_EXPERTS = 3
    # TOP_K = 2
    routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()

    affine_coeff = 0.005

    weights = {}
    for expert_id in range(NUM_EXPERTS):
        w1_weight = torch.randint(-128,
                                  127, (INTERMEDIATE_SIZE, HIDDEN_SIZE // 2),
                                  dtype=torch.int8).cuda()
        w2_weight = torch.randint(-128,
                                  127, (HIDDEN_SIZE, INTERMEDIATE_SIZE // 2),
                                  dtype=torch.int8).cuda()
        w3_weight = torch.randint(-128,
                                  127, (INTERMEDIATE_SIZE, HIDDEN_SIZE // 2),
                                  dtype=torch.int8).cuda()

        w1_scale = torch.randn(
            (INTERMEDIATE_SIZE, HIDDEN_SIZE // SCALING_GROUP_SIZE),
            dtype=dtype).cuda() * affine_coeff
        w2_scale = torch.randn(
            (HIDDEN_SIZE, INTERMEDIATE_SIZE // SCALING_GROUP_SIZE),
            dtype=dtype).cuda() * affine_coeff
        w3_scale = torch.randn(
            (INTERMEDIATE_SIZE, HIDDEN_SIZE // SCALING_GROUP_SIZE),
            dtype=dtype).cuda() * affine_coeff

        w1_input = torch.randn(1, dtype=torch.float32).cuda() * 0.02
        w2_input = w1_input
        w3_input = w1_input

        weights[f"{expert_id}.w1.weight"] = w1_weight
        weights[f"{expert_id}.w2.weight"] = w2_weight
        weights[f"{expert_id}.w3.weight"] = w3_weight
        weights[f"{expert_id}.w1.weight_scale_inv"] = w1_scale
        weights[f"{expert_id}.w2.weight_scale_inv"] = w2_scale
        weights[f"{expert_id}.w3.weight_scale_inv"] = w3_scale
        weights[f"{expert_id}.w1.input_scale"] = w1_input
        weights[f"{expert_id}.w2.input_scale"] = w2_input
        weights[f"{expert_id}.w3.input_scale"] = w3_input

    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_AWQ)
    fused_moe = FusedMoE(num_experts=NUM_EXPERTS,
                         routing_method=routing_method,
                         hidden_size=HIDDEN_SIZE,
                         intermediate_size=INTERMEDIATE_SIZE,
                         dtype=dtype,
                         reduce_results=False,
                         model_config=ModelConfig(quant_config=quant_config))
    fused_moe.load_weights([weights])
    fused_moe.cuda()

    def ref():
        results = torch.zeros_like(x)
        selected_experts, final_scales = routing_method.apply(router_logits)
        unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
        for e_idx in range(NUM_EXPERTS):
            mask = selected_experts == e_idx
            activated_tokens = mask.sum(1).bool()
            act = x[activated_tokens, :]
            if act.shape[0] == 0:
                continue
            final_scale = (final_scales *
                           mask).sum(1)[activated_tokens].unsqueeze(1)

            # weights
            w1 = weights[f"{e_idx}.w1.weight"]
            w1 = unpacker(w1.cpu()).T.contiguous().cuda()
            w2 = weights[f"{e_idx}.w2.weight"]
            w2 = unpacker(w2.cpu()).T.contiguous().cuda()
            w3 = weights[f"{e_idx}.w3.weight"]
            w3 = unpacker(w3.cpu()).T.contiguous().cuda()
            w3_w1 = torch.cat([w3, w1], dim=-1)

            # scales
            s1 = weights[f"{e_idx}.w1.weight_scale_inv"].T.contiguous().cuda()
            s2 = weights[f"{e_idx}.w2.weight_scale_inv"].T.contiguous().cuda()
            s3 = weights[f"{e_idx}.w3.weight_scale_inv"].T.contiguous().cuda()
            s3_s1 = torch.cat([s3, s1], dim=-1)

            # prequant / alpha
            p1 = weights[f"{e_idx}.w1.input_scale"].cuda()
            p2 = weights[f"{e_idx}.w2.input_scale"].cuda()
            p3 = weights[f"{e_idx}.w3.input_scale"].cuda()
            p3_p1 = max(p1, p3)

            act = torch.clamp((act / p3_p1), -448.0,
                              448.0).to(torch.float8_e4m3fn).to(dtype)
            w3_w1 = (w3_w1.float() *
                     s3_s1.repeat_interleave(128, dim=0).float()).to(dtype)
            fc1 = torch.matmul(act, w3_w1) * p3_p1
            fc1, gate = fc1.chunk(2, dim=-1)
            fc1 = fc1 * torch.nn.functional.silu(gate)

            act = torch.clamp((fc1 / p2), -448.0,
                              448.0).to(torch.float8_e4m3fn).to(dtype)
            w2 = (w2.float() *
                  s2.repeat_interleave(128, dim=0).float()).to(dtype)
            fc2 = torch.matmul(act, w2) * p2
            results[activated_tokens, :] += (fc2 * final_scale).to(
                results.dtype)
        return results

    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        fused_moe.forward(x, router_logits)

    torch.cuda.synchronize()
    with torch.inference_mode():
        nvtx.range_push("w4a8")
        output = fused_moe.forward(x, router_logits)
        nvtx.range_pop()
        ref_output = ref()

    # compare
    torch.cuda.synchronize()
    # torch.testing.assert_close(output, ref_output, rtol=0.1, atol=0.5)

class RefGatedMLPFusedMoE(nn.Module):

    def __init__(self,
                 num_experts: int,
                 routing_method: BaseMoeRoutingMethod,
                 hidden_size: int,
                 intermediate_size: int,
                 dtype: Optional[torch.dtype] = None,
                 model_config: ModelConfig = ModelConfig()):
        super().__init__()
        self.num_experts = num_experts
        self.routing_method = routing_method
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.dtype = dtype
        self.quant_config = model_config.quant_config

        self.experts = nn.ModuleList([
            GatedMLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                bias=False,
                dtype=self.dtype,
                config=model_config,
            ) for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_size
        hidden_states = hidden_states.view(-1, self.hidden_size)

        selected_experts, routing_weights = self.routing_method.apply(
            router_logits)

        final_hidden_states = torch.zeros(hidden_states.shape,
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)

        for expert_id in range(self.num_experts):
            if not torch.any(selected_experts == expert_id):
                continue
            batch_idx, nth_expert = torch.where(selected_experts == expert_id)
            expert_inputs = hidden_states[batch_idx]

            output = self.experts[expert_id](expert_inputs)
            final_hidden_states[batch_idx] += routing_weights[
                batch_idx, nth_expert, None] * output.float()

        final_hidden_states = final_hidden_states.reshape(hidden_states.shape)
        return final_hidden_states

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        weights = weights[0]

        for expert in range(self.num_experts):
            gate_up_proj_weights = [{}, {}]
            down_proj_weights = [{}]

            gate_up_proj_weights[0]['weight'] = weights[f"{expert}.w1.weight"]
            gate_up_proj_weights[1]['weight'] = weights[f"{expert}.w3.weight"]
            down_proj_weights[0]['weight'] = weights[f"{expert}.w2.weight"]

            if self.quant_config and self.quant_config.quant_algo == QuantAlgo.FP8:
                gate_up_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]['weight_scale'] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w2.weight_scale"]
                gate_up_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w1.input_scale"]
                gate_up_proj_weights[1]['input_scale'] = weights[
                    f"{expert}.w3.input_scale"]
                down_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w2.input_scale"]
            elif self.quant_config and self.quant_config.quant_algo == QuantAlgo.NVFP4:
                gate_up_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]['weight_scale'] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w2.weight_scale"]
                gate_up_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w1.input_scale"]
                gate_up_proj_weights[1]['input_scale'] = weights[
                    f"{expert}.w3.input_scale"]
                down_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w2.input_scale"]
                gate_up_proj_weights[0]['weight_scale_2'] = weights[
                    f"{expert}.w1.weight_scale_2"]
                gate_up_proj_weights[1]['weight_scale_2'] = weights[
                    f"{expert}.w3.weight_scale_2"]
                down_proj_weights[0]['weight_scale_2'] = weights[
                    f"{expert}.w2.weight_scale_2"]

            self.experts[expert].gate_up_proj.load_weights(gate_up_proj_weights)
            self.experts[expert].down_proj.load_weights(down_proj_weights)


if __name__ == "__main__":

    dtype = torch.bfloat16

    for SEQ_LEN in [1, 4, 8, 16, 32, 64, 128]:

        # SEQ_LEN = 8

        # DeepSeek-R1
        HIDDEN_SIZE = 7168
        INTERMEDIATE_SIZE = 2048
        SCALING_GROUP_SIZE = 128
        NUM_EXPERTS = 32
        TOP_K = 8

        # Qwen3-235B-A22B
        HIDDEN_SIZE = 4096
        INTERMEDIATE_SIZE = 1536
        SCALING_GROUP_SIZE = 128
        NUM_EXPERTS = 16
        TOP_K = 8

        print(f"e {NUM_EXPERTS} m {SEQ_LEN} n {INTERMEDIATE_SIZE} k {HIDDEN_SIZE} topk {TOP_K}")

        print("FP8 Tensor")
        t_fp8_tensor = test_fused_moe_fp8_tensor_scaling(
            dtype=dtype,
            SEQ_LEN=SEQ_LEN,
            HIDDEN_SIZE=HIDDEN_SIZE,
            INTERMEDIATE_SIZE=INTERMEDIATE_SIZE,
            NUM_EXPERTS=NUM_EXPERTS,
            TOP_K=TOP_K)
        print("FP8 Block")
        t_fp8_block = test_fused_moe_fp8_block_scaling(
            dtype=dtype,
            SEQ_LEN=SEQ_LEN,
            HIDDEN_SIZE=HIDDEN_SIZE,
            INTERMEDIATE_SIZE=INTERMEDIATE_SIZE,
            NUM_EXPERTS=NUM_EXPERTS,
            TOP_K=TOP_K)
        print("W4A8")
        t_w4a8 = test_fused_moe_w4afp8(
            dtype=dtype,
            SEQ_LEN=SEQ_LEN,
            HIDDEN_SIZE=HIDDEN_SIZE,
            INTERMEDIATE_SIZE=INTERMEDIATE_SIZE,
            SCALING_GROUP_SIZE=SCALING_GROUP_SIZE,
            NUM_EXPERTS=NUM_EXPERTS,
            TOP_K=TOP_K)


        # tb = pt.PrettyTable( ["Test case", "FP8 Per-tensor (us)", "FP8 Per-block (us)", "W4A8 (us)", "Acc Ratio"])
        # tb.add_row([f"m{SEQ_LEN}_n{INTERMEDIATE_SIZE}_k{HIDDEN_SIZE}_e{NUM_EXPERTS}_topk{TOP_K}", t_fp8, t_w4a8, t_fp8 / t_w4a8])
        # print(tb)
