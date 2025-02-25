import numpy as np

def cos_similarity(arr1, arr2):
    vec1 = arr1.flatten().astype(float)
    vec2 = arr2.flatten().astype(float)

    dot_product = np.dot(vec1, vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
   
    if norm1 == 0 or norm2 == 0:
        return 0.0  # 零向量视为完全不相似
    
    cos_sim = dot_product / (norm1 * norm2)
    return cos_sim

outputs_hf = []
outputs_tllm = []

tensor_list = [
    'embeds', 
    'norm', 
    'attn'
    ]

for i in range(60, 66):
    print(f"iter {i} --------------------------------------------------------------------------------")
    for name in tensor_list:
        path_tllm = f'/tmp/tllm_debug/PP_1/TP_1/iteration_{i}/transformer.'
        path_hf = f'/tmp/tllm_debug/PP_1/TP_1/iteration_{i}/transformer.'

        if name in ['attn']:
            path_tllm += 'layers.0.'
            path_hf += 'layers.0.'

        path_tllm += f'trtllm_{name}_output.npy'
        path_hf += f'hf_{name}_output.npy'
        # print(path)
        output_hf = np.load(path_hf)
        outputs_hf.append(output_hf)
        output_tllm = np.load(path_tllm)
        outputs_tllm.append(output_tllm)
        print(f"{name} iter {i} {cos_similarity(output_tllm, output_hf)}")
        # print(f"iter {i} --------------------------------------------------------------------------------")
        # print(f"{name}_output_tllm {output_tllm.shape} {output_tllm}")
        # print(f"{name}_output_hf {output_hf.shape} {output_hf}")

print(f"-----------------------------------------------------------------------------------------")


path_hf = f'/tmp/tllm_debug/PP_1/TP_1/iteration_0/transformer.hf_logits.npy'
output_hf = np.load(path_hf)
path_tllm = f'/tmp/tllm_debug/PP_1/TP_1/iteration_0/transformer.trtllm_context_logits.npy'
output_tllm = np.load(path_tllm)
# print(f"HF     context logits {output_hf.shape} {output_hf}")
# print(f"TRTLLM context logits {output_tllm.shape} {output_tllm}")
print(f"logits iter {i} {cos_similarity(output_tllm, output_hf)}")

for i in range(60, 66):
    path_hf = f'/tmp/tllm_debug/PP_1/TP_1/iteration_{i}/transformer.hf_logits.npy'
    output_hf = np.load(path_hf)
    path_tllm = f'/tmp/tllm_debug/PP_1/TP_1/iteration_{i}/transformer.trtllm_gen_logits.npy'
    output_tllm = np.load(path_tllm)
    print(f"gen logits iter {i} {cos_similarity(output_tllm, output_hf)}")
    # print(f"HF     gen logits {i} {output_hf.shape} {output_hf}")
    # print(f"TRTLLM gen logits {i} {output_tllm.shape} {output_tllm}")


index = 64
path_hf = f'/tmp/tllm_debug/PP_1/TP_1/iteration_{index}/transformer.hf_logits.npy'
output_hf = np.load(path_hf)
path_tllm = f'/tmp/tllm_debug/PP_1/TP_1/iteration_{index}/transformer.trtllm_gen_logits.npy'
output_tllm = np.load(path_tllm)

print(output_hf.shape, output_tllm.shape)

max_index = np.argmax(output_hf)
max_value = output_hf[0, 0, max_index]
print(max_index, max_value)
print(102545, output_hf[0, 0, 102545])

max_index = np.argmax(output_tllm)
max_value = output_tllm[max_index]
print(max_index, max_value)
print(102545, output_tllm[102545])

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/llm-models/Qwen2-VL-7B-Instruct")
print(tokenizer.decode(44729))
print(tokenizer.decode(102545))