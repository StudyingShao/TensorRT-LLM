
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

import torch

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/llm-models/Qwen2-VL-7B-Instruct/",
    torch_dtype="auto",
    device_map="auto"
)
    # torch_dtype=torch.float32,

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

print(f"model.dtype {model.dtype}")
# print(model)

# default processer
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# print(processor)

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./pics/qwen2vl_pic2.jpg",
            },
            {"type": "text", "text": "请输出图中的文字"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(f"jiangs messages {messages}")


image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# import pdb; pdb.set_trace()

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=512, output_logits=True, output_scores=True, return_dict_in_generate=True)

print(f"inputs.input_ids {inputs.input_ids.shape} {inputs.input_ids}")
# print(generated_ids.logits)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)
]

# print("----------------------------------------")
# print(generated_ids)
# print("----------------------------------------")
print(f"generated_ids_trimmed {generated_ids_trimmed[0].shape} {generated_ids_trimmed[0]}")
for i in range(generated_ids_trimmed[0].shape[0]):
    print(f"{i} ", generated_ids_trimmed[0][i].item())
# print("----------------------------------------")

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)