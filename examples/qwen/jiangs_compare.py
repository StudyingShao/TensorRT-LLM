
import torch
import pickle


def compare():
    # jiangs_list = ['transformer.layers.0.mlp.mlp_fc_output']
    jiangs_list = ['transformer.layers.0.mlp.mlp_fc_output',
                'transformer.layers.0.mlp.mlp_gate_output',
                'transformer.layers.0.mlp.mlp_proj_output',
                'transformer.layers.0.attention.attn_dense_output',
                'transformer.layers.0.attention.attn_qkv_output',]

    for jiangs in jiangs_list:
        file_path1 = "/TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/examples/qwen/jiangs/debug/bf16_" + jiangs + ".pickle"
        file_path2 = "/TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/examples/qwen/jiangs/debug/fp16_" + jiangs + ".pickle"

        a = None
        b = None
        with open(file_path1, "rb") as f:
            a = pickle.load(f)
        with open(file_path2, "rb") as f:
            b = pickle.load(f)
        
        c = torch.abs(a - b)
        print(f"{jiangs} {c.shape} median: {c.median():.4g}, mean: {c.mean():.4g}, max: {c.max():.4g}")

        # 找到最大差异的位置
        max_diff_value = torch.max(c)
        max_diff_index = torch.argmax(c)

        # 获取位置的行和列索引
        row, col = divmod(max_diff_index.item(), c.size(1))

        # 输出结果
        print(f"最大差异的位置: ({row}, {col})")
        print(f"a 和 b 在该位置的值: a[{row}, {col}] = {a[row, col].item()}, b[{row}, {col}] = {b[row, col].item()}")
        print(f"最大差异值: {max_diff_value.item()}")
        for i in range(a.shape[0]):
            print(f"{i}: ", end='')
            for j in range(a.shape[1]):
                print(f"{a[i, j]:.4g} ", end='')
            print()
            print(f"{i}: ", end='')
            for j in range(a.shape[1]):
                print(f"{b[i, j]:.4g} ", end='')
            print()

def change_dtype():
    import safetensors.torch
    from safetensors import safe_open
    file_path = '/TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/examples/qwen/jiangs/ckpt/Qwen2-7B-Instruct_float16/1/w4a8_awq_kvfp8/rank0.safetensors'
    
    weights = safetensors.torch.load_file(file_path)
    # print(weights)
    for name in weights.keys():
        value = weights[name]
        print(f"{name}: {value.dtype} {value.shape}")

    # with safe_open(file_path, framework='pt', device="cuda:0") as f:
    #     for name in f.keys():
    #         print(name)
            # model_params[name] = f.get_tensor(name).to(dtype).clone()


change_dtype()