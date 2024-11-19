"""
This file is part of the TensorRT-LLM repository.
"""
import sys

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.runtime import (
    PYTHON_BINDINGS,
    ModelRunner,
    ModelRunnerCpp,
    Session,
    TensorInfo,
)

from tqdm import tqdm

import csv

import time
import nvtx
from torch.nn import functional as F

import os

# from conversation import Conversation, get_conv_template

g_top_k = 1
g_top_p = 0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """
    Builds a transform that resizes an image to the given size and normalizes it.
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    Finds the closest aspect ratio to the given aspect ratio that is within
    the target ratios.
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    """
    Preprocess the given image to be used as input for the model.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    """
    Load the image from the given file and preprocess it for the model.
    """
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def run_infer(
    runner,
    batch_input_ids,
    ptuning_args,
    extra_ids,
    max_new_tokens,
    pad_id,
    end_id,
    top_k=1,
    top_p=0,
):
    """
    Run inference on the model.
    """
    length_diff = batch_input_ids[0].shape[0] - extra_ids.shape[0]

    if length_diff > 0:
        extra_ids = F.pad(extra_ids, (0, length_diff), "constant", 0)

    print(f"jiangs runner {runner}")

    outputs = runner.generate(
        batch_input_ids=batch_input_ids,
        input_token_extra_ids=extra_ids,
        max_new_tokens=max_new_tokens,
        pad_id=pad_id,
        end_id=end_id,
        return_dict=True,
        prompt_table=ptuning_args[0].unsqueeze(0),
        output_sequence_lengths=True,
        top_k=top_k,
        top_p=top_p,
    )

    return outputs

def print_output(outputs, input_lengths, tokenizer):
    """
    Print the output of the model.
    """
    outputs = outputs["output_ids"][0][:, input_lengths:]
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(response)

if __name__ == "__main__":
    """
    Usage:
        python3 main.py --image_path ./examples/image1.jpg \
    """
    # set the max number of tiles in `max_num`
    pixel_values = (
        load_image("./dog.jpg", max_num=12).to(torch.bfloat16).cuda()
    )
    generation_config = dict(max_new_tokens=100, do_sample=True)

    # single-image single-round conversation (单图单轮对话)
    question = "<image>\nPlease describe the image shortly."
    #question = "<image>\nGive me a story about this picture."

    # Init tokenizer
    # tokenizer_path = "/home/host/yangmengyu/workspace/model2/InternVL2/8B/InternVL2-8B"
    tokenizer_path = "/TRT/llm-models/internVL2/2B/"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=False
    )

    # Init model
    # model_path = "/home/host/yangmengyu/workspace/model2/InternVL2/8B/InternVL2-8B"
    # engine_path = "/home/host/yangmengyu/workspace/model2/InternVL2/8B/engine"
    model_path = "/TRT/llm-models/internVL2/2B/"
    # engine_path = "/TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/examples/internVL2/jiangs/trt_engines/2B/bf16/1-gpu"
    engine_path = "/TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/examples/internVL2/jiangs/trt_engines/2B/bf16/1-gpu_pagedcontext"

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model = (
        AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )

    model.init_language_model(engine_path, use_cpp_runner=True, kv_cache_reuse=True)
    # model.init_language_model(engine_path, use_cpp_runner=False, kv_cache_reuse=True)

    (
        input_ids,
        ptuning_args,
        extra_ids,
        input_lengths,
        generation_config,
    ) = model.preprocess(tokenizer, pixel_values, question, generation_config)

    test_cnt = 3

    jiangs_input_ids = input_ids.clone()
    jiangs_input_ids[0, 0] = 250
    print(f"jiangs input_ids {input_ids.shape} {input_ids}")
    print(f"jiangs input_ids {jiangs_input_ids.shape} {jiangs_input_ids}")

    #torch.set_printoptions(threshold=torch.inf)
    total_start_time = time.perf_counter()
    for i in range(test_cnt):
        print(
            "Loop {}/{}:".format(
                i + 1, test_cnt
            )
        )
        outputs = run_infer(
            runner=model.language_model,
            batch_input_ids=input_ids,
            extra_ids=extra_ids,
            ptuning_args=ptuning_args,
            max_new_tokens=generation_config["max_new_tokens"],
            pad_id=2,
            end_id=generation_config["eos_token_id"],
            top_k=g_top_k,
            top_p=g_top_p,
        )

        print("----------------------- model output ------------------------- {} {}".format(outputs["context_logits"][0].shape, torch.std(outputs["context_logits"][0])))
        # print("----------------------- model output ------------------------- {} {}".format(outputs["context_logits"][0].shape, outputs["context_logits"]))

        # debug_output = outputs["context_logits"][0][:,:30]
        # for i in range(debug_output.shape[0]):
        #     print(f"i {i}: ", end='')
        #     for j in range(30):
        #         print("j {}: {}, ".format(j, debug_output[i, j]), end='')
        #     print()

        print_output(outputs, input_lengths, tokenizer)

    # -----------------------------------------------------------------------------------------------------------------------------
    outputs = run_infer(
        runner=model.language_model,
        batch_input_ids=jiangs_input_ids,
        extra_ids=extra_ids,
        ptuning_args=ptuning_args,
        max_new_tokens=generation_config["max_new_tokens"],
        pad_id=2,
        end_id=generation_config["eos_token_id"],
        top_k=g_top_k,
        top_p=g_top_p,
    )

    print("----------------------- model output ------------------------- {} {}".format(outputs["context_logits"][0].shape, torch.std(outputs["context_logits"][0])))
    print_output(outputs, input_lengths, tokenizer)
    # -----------------------------------------------------------------------------------------------------------------------------

    for i in range(test_cnt):
        print(
            "Loop {}/{}:".format(
                i + 1, test_cnt
            )
        )
        outputs = run_infer(
            runner=model.language_model,
            batch_input_ids=input_ids,
            extra_ids=extra_ids,
            ptuning_args=ptuning_args,
            max_new_tokens=generation_config["max_new_tokens"],
            pad_id=2,
            end_id=generation_config["eos_token_id"],
            top_k=g_top_k,
            top_p=g_top_p,
        )

        print("----------------------- model output ------------------------- {} {}".format(outputs["context_logits"][0].shape, torch.std(outputs["context_logits"][0])))
        print_output(outputs, input_lengths, tokenizer)
    average_target_time = (time.perf_counter() - total_start_time) * 1000 / test_cnt
    print(f"average_target_time {average_target_time}")
