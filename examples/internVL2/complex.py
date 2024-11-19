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


def decode_output(tokenizer, output_ids, input_lengths, sequence_lengths):
    """
    Decode the output text.
    Args:
        tokenizer (AutoTokenizer): The tokenizer to use for decoding the output text.
    """
    output_str = ""
    batch_size, num_beams, _ = output_ids.size()
    for batch_idx in range(batch_size):
        inputs = output_ids[batch_idx][0][: input_lengths[batch_idx]].tolist()
        input_text = tokenizer.decode(inputs)
        output_str += input_text
        # print('Input [Text {}]'.format(input_text))
        for beam in range(num_beams):
            output_begin = input_lengths[batch_idx]
            output_end = sequence_lengths[batch_idx][beam]
            outputs = output_ids[batch_idx][beam][output_begin:output_end].tolist()
            output_text = tokenizer.decode(outputs)
            output_str += output_text
            # print('Output [Text {} Beam {}]: \"{}\"'.format(batch_idx, beam, output_text))

    output_ids = output_ids.reshape((-1, output_ids.size(2)))

    return output_str


def sample(probs: torch.Tensor, num_samples: int = 1):
    """
    从给定的概率分布中随机选取一个或多个样本。

    Args:
        probs (torch.Tensor, shape=(N,)): 包含N个元素的浮点型张量，表示每个类别的概率分布。
            N为输入数据的类别数。
        num_samples (int, optional, default=1): 要随机选取的样本数量。默认为1。

    Returns:
        torch.LongTensor, shape=(num_samples,), dtype=long: 随机选取的样本索引，范围为[0, N-1]。

    Raises:
        RuntimeError: 如果输入的概率分布中没有任何非零元素，则会引发RuntimeError异常。
    """
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    if idx_next.item() == 0:
        raise RuntimeError
    return idx_next


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float("-inf")
    return logits


def norm_logits(
    logits: torch.Tensor, temperature: float, top_k: float, top_p: float
) -> torch.Tensor:
    """
    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def max_fn(x):
    """
    norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum


def extract_generation_tokens(runner_output: dict, input_length: int):
    """
    Extract generation tokens from the runner output.
    """
    output_ids = runner_output["output_ids"][0]
    sequence_lengths = runner_output["sequence_lengths"][0]
    output_begin = input_length
    output_end = sequence_lengths[0]
    outputs_tokens = output_ids[0][output_begin:output_end]

    return outputs_tokens


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
    # print('extra_ids.shape: ', extra_ids.shape)
    # print('batch_input_ids.shape: ', batch_input_ids[0].shape)

    length_diff = batch_input_ids[0].shape[0] - extra_ids.shape[0]

    if length_diff > 0:
        # print(
        #     f"length_diff: {length_diff}, batch_input_ids.shape[0]: "
        #     f"{batch_input_ids[0].shape[0]}, extra_ids.shape[0]: "
        #     f"{extra_ids.shape[0]}"
        # )
        # print(f">>> extra_ids.shape: {extra_ids.shape}")
        # zero_tensor = torch.zeros(length_diff)
        extra_ids = F.pad(extra_ids, (0, length_diff), "constant", 0)
        #extra_ids = torch.cat([extra_ids, torch.zeros(length_diff, dtype=torch.extra_ids.dtype)])
        # print(f"<<< extra_ids.shape: {extra_ids.shape}")

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


def speculative_sampling(
    batch_input_ids: list,
    target_ptuning_args,
    target_extra_ids,
    draft_ptuning_args,
    draft_extra_ids,
    tokenizer,
    target_runner,
    draft_runner,
    end_id,
    pad_id,
    input_length: int,
    max_len=100,
    gamma=4,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 0,
    random_seed: int = None,
):
    """
    Speculative Sampling
    """
    total_start_time = time.perf_counter()
    draft_runner_time = 0
    target_runner_time = 0

    T = batch_input_ids[0].shape[0] + max_len
    print(f"Speculative sampling: T = {T}")

    assert len(batch_input_ids) == 1, "input batch size must be 1"

    prefix = batch_input_ids[0].to("cuda")

    n_accept = 0
    n_reject = 0

    with tqdm(total=T, desc="speculative sampling") as pbar:
        # while n < T do
        while prefix.shape[0] < T:
            x = prefix[:]
            prefix_len = prefix.shape[0]

            # for t = 1 : K do
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            rng = nvtx.start_range(message="draft", color="green")
            cur_draft_runner_start_time = time.perf_counter()
            cur_draft_output = run_infer(
                runner=draft_runner,
                batch_input_ids=[x],
                extra_ids=draft_extra_ids,
                ptuning_args=draft_ptuning_args,
                max_new_tokens=gamma,
                pad_id=pad_id,
                end_id=end_id,
                top_k=top_k,
                top_p=top_p,
            )

            draft_runner_time += (
                time.perf_counter() - cur_draft_runner_start_time
            ) * 1000
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            nvtx.end_range(rng)

            p = cur_draft_output["generation_logits"][0]

            # q = q[:, :, :128264]
            for i in range(p.shape[1]):
                p[:, i, :] = norm_logits(p[:, i, :], temperature, top_k, top_p)

            cur_draft_generation_tokens = extract_generation_tokens(
                cur_draft_output, prefix_len
            )

            x = torch.cat([x, cur_draft_generation_tokens])

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            rng = nvtx.start_range(message="target", color="red")
            cur_target_runner_start_time = time.perf_counter()

            cur_target_output_ids = run_infer(
                runner=target_runner,
                batch_input_ids=[x],
                extra_ids=target_extra_ids,
                ptuning_args=target_ptuning_args,
                max_new_tokens=1,
                pad_id=pad_id,
                end_id=end_id,
                top_k=top_k,
                top_p=top_p,
            )
            target_runner_time += (
                time.perf_counter() - cur_target_runner_start_time
            ) * 1000
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            nvtx.end_range(rng)
            q = cur_target_output_ids["context_logits"][0].unsqueeze(0)[
                :, (0 - gamma - 1) :, :
            ]
            q = torch.cat([q, cur_target_output_ids["generation_logits"][0]], dim=1)

            for i in range(q.shape[1]):
                q[:, i, :] = norm_logits(q[:, i, :], temperature, top_k, top_p)

            is_all_accept = True
            n = prefix_len - 1
            x = x.unsqueeze(0)

            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device=p.device)

                j = x[:, prefix_len + i]

                if r < torch.min(
                    torch.tensor([1], device=q.device), q[:, i, j] / p[:, i, j]
                ):
                    # accept, and update n
                    n += 1
                    print("prefix_str [0] ", tokenizer.decode(x[:, prefix_len + i]))
                else:
                    # reject
                    t = sample(
                        max_fn(
                            q[:, n - prefix_len + 1, :] - p[:, n - prefix_len + 1, :]
                        )
                    )
                    is_all_accept = False
                    n_reject += 1
                    print(
                        "prefix_str [1] {} -> {} {}".format(
                            tokenizer.decode(t[0]),
                            r,
                            torch.min(
                                torch.tensor([1], device=q.device),
                                q[:, i, j] / p[:, i, j],
                            ),
                        )
                    )
                    break

                n_accept += 1

            prefix = x[:, : n + 1]

            if is_all_accept:
                t = sample(q[:, -1, :])

            prefix_str = tokenizer.decode(prefix.squeeze(0)[input_lengths:])
            print("prefix_str {}".format(prefix_str))
            # xxx_t = tokenizer.decode(t[0])
            # print("xxx_t {}".format(xxx_t))

            print("---------------------------------------------------")
            prefix = torch.cat((prefix, t), dim=1)
            prefix = prefix.squeeze(0)
            pbar.update(n - pbar.n)

    text = tokenizer.decode(prefix[input_lengths:])
    print("speculative_sampling text: {}".format(text))

    print("n_accept {}, n_reject {}".format(n_accept, max_len - n_accept))
    # print(text)

    totoal_run_time = (time.perf_counter() - total_start_time) * 1000
    print(
        f"draft_runner_time {draft_runner_time}, target_runner_time {target_runner_time}"
    )
    print(f"total_run_time {totoal_run_time}")

    return text


def print_output(outputs, input_lengths, tokenizer):
    """
    Print the output of the model.
    """
    outputs = outputs["output_ids"][0][:, input_lengths:]
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(response)


def SPD_token_judge(
    batch_input_ids: list,
    target_ptuning_args,
    target_extra_ids,
    draft_ptuning_args,
    draft_extra_ids,
    tokenizer,
    target_runner,
    draft_runner,
    end_id,
    pad_id,
    input_length: int,
    max_len=100,
    gamma=4,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 0,
    random_seed: int = None,
):
    """
    SPD token judge.
    """
    draft_generation_max_new_tokens = gamma

    input_ids = batch_input_ids[0][:]

    global_generation_sequence = torch.tensor([], dtype=int, device="cuda")

    draft_model_call_time = 0
    large_model_call_time = 0
    acc_token_num = 0
    reject_times = 0

    input_lengths = input_ids.shape[0]

    import time
    draft_model_total_time = 0
    target_model_total_time = 0

    generation_mark = True
    while generation_mark:
        # draft_model generation
        draft_model_call_time += 1

        start_time = time.time()

        draft_generation_dict = run_infer(
            runner=draft_runner,
            batch_input_ids=
                torch.cat((input_ids, global_generation_sequence), dim=0).to("cuda").unsqueeze(dim=0),
            extra_ids=draft_extra_ids,
            ptuning_args=draft_ptuning_args,
            max_new_tokens=draft_generation_max_new_tokens,
            pad_id=pad_id,
            end_id=end_id,
            top_k=top_k,
            top_p=top_p,
        )

        draft_model_total_time += time.time() - start_time

        current_draft_length = draft_generation_dict["generation_logits"][0].shape[1]
        draft_generation_sequence = draft_generation_dict["output_ids"][0][
            :,
            input_lengths
            + global_generation_sequence.shape[0] : input_lengths
            + global_generation_sequence.shape[0]
            + current_draft_length,
        ].squeeze(0)

        # target_model evaluation
        large_model_call_time += 1

        target_generation_sequence = None
        # large model generation
        with torch.no_grad():
            
            start_time = time.time()

            target_model_generation = run_infer(
                runner=target_runner,
                batch_input_ids=
                    torch.cat(
                        (
                            input_ids,
                            global_generation_sequence.to(input_ids.device),
                            draft_generation_sequence.to(input_ids.device),
                        ),
                        dim=0,
                    ).to(input_ids.device).unsqueeze(dim=0),
                extra_ids=target_extra_ids,
                ptuning_args=target_ptuning_args,
                max_new_tokens=1,
                pad_id=pad_id,
                end_id=end_id,
                top_k=top_k,
                top_p=top_p,
            )

            target_model_total_time += time.time() - start_time

            # target_generation_logits =target_model_generation['context_logits'][0][
            #     input_lengths
            #     + global_generation_sequence.shape[0] - 1: 
            #     input_lengths
            #     + global_generation_sequence.shape[0] - 1
            #     + current_draft_length
            #     + 1,
            # ]
            target_generation_logits = target_model_generation['context_logits'][0][-current_draft_length-1:]
            target_generation_sequence = torch.argmax(target_generation_logits, dim=-1)

            # target_generation_sequence = target_model_generation["output_ids"][0][
            #     :,
            #     input_lengths
            #     + global_generation_sequence.shape[0] : input_lengths
            #     + global_generation_sequence.shape[0]
            #     + current_draft_length
            #     + 1 :,
            # ].squeeze(0)

        is_all_accept = True
        accept_num = 0

        for token_idx in range(current_draft_length):
            current_token = draft_generation_sequence[token_idx]
            target_token = target_generation_sequence[token_idx]
            target_next_token = target_generation_sequence[token_idx + 1]

            if target_next_token == end_id:
                # accept_num -= 1
                acc_token_num += 1
                generation_mark = False
                break
            if current_token == target_token:
                accept_num += 1
                acc_token_num += 1
                continue
            else:
                is_all_accept = False
                reject_times += 1
                break

        global_generation_sequence = torch.cat(
            (
                global_generation_sequence,
                target_generation_sequence[: accept_num + 1],
            ),
            dim=0,
        )

        if is_all_accept:
            draft_generation_max_new_tokens += 2
        else:
            draft_generation_max_new_tokens = max(
                1, draft_generation_max_new_tokens - 1
            )

        if global_generation_sequence.shape[0] >= max_len:
            break

    res_str = tokenizer.decode(global_generation_sequence)
    print("[SPD output]: ", res_str)

    state_dict = {
        "draft_model_call_time": draft_model_call_time,
        "large_model_call_time": large_model_call_time,
        "reject_times": reject_times,
        "acc_token_num": acc_token_num,
        "draft_model_total_time": draft_model_total_time,
        "target_model_total_time": target_model_total_time,
    }
    print(state_dict)
    return res_str


def main(unimodel_forward=True, spd_forward=True, use_cpp=True, kvcache_reuse=True):
    """
    Usage:
        python3 main.py --image_path ./examples/image1.jpg \
    """
    # set the max number of tiles in `max_num`
    pixel_values = (
        load_image("./examples/image1.jpg", max_num=12).to(torch.bfloat16).cuda()
    )
    generation_config = dict(max_new_tokens=100, do_sample=True)

    # single-image single-round conversation (单图单轮对话)
    question = "<image>\nPlease describe the image shortly."
    #question = "<image>\nGive me a story about this picture."

    model_root_path = "/home/host/yangmengyu/workspace/model2/InternVL2/"

    # Init tokenizer
    tokenizer_path = model_root_path + "/8B/InternVL2-8B"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=False
    )

    # Init model
    model_path = model_root_path + "/8B/InternVL2-8B"
    engine_path = model_root_path + "/8B/engine"

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

    model.init_language_model(engine_path, use_cpp_runner=use_cpp, kv_cache_reuse=kvcache_reuse)

    (
        input_ids,
        ptuning_args,
        extra_ids,
        input_lengths,
        generation_config,
    ) = model.preprocess(tokenizer, pixel_values, question, generation_config)

    test_cnt = 2

    if unimodel_forward:
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
            # print("----------------------- model output ------------------------- {} {}".format(outputs["context_logits"][0].shape, torch.std(outputs["context_logits"][0])))
            #print("------------------- model output ------------------- {}, shape is {} ".format(outputs["context_logits"][0], outputs["context_logits"][0].shape))
            print_output(outputs, input_lengths, tokenizer)
        average_target_time = (time.perf_counter() - total_start_time) * 1000 / test_cnt
        print(f"average_target_time {average_target_time}")

    # Init draft model
    draft_model_path = model_root_path + "/2B/InternVL2-2B"
    draft_engine_path = model_root_path + "/2B/engine"

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    draft_model = (
        AutoModel.from_pretrained(
            draft_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )

    draft_model.init_language_model(
        draft_engine_path, use_cpp_runner=use_cpp, kv_cache_reuse=kvcache_reuse)

    (
        draft_input_ids,
        draft_ptuning_args,
        draft_extra_ids,
        draft_input_lengths,
        draft_generation_config,
    ) = draft_model.preprocess(tokenizer, pixel_values, question, generation_config)

    if unimodel_forward:
        total_start_time = time.perf_counter()
        for i in range(test_cnt):
            print(
                "Loop {}/{} -------------------------------------------------".format(
                    i + 1, test_cnt
                )
            )
            outputs = run_infer(
                runner=draft_model.language_model,
                batch_input_ids=input_ids,
                extra_ids=draft_extra_ids,
                ptuning_args=draft_ptuning_args,
                max_new_tokens=generation_config["max_new_tokens"],
                pad_id=2,
                end_id=generation_config["eos_token_id"],
                top_k=g_top_k,
                top_p=g_top_p,
            )
            print("----------------------- draft model -------------------------")
            print_output(outputs, input_lengths, tokenizer)
        average_draft_time = (time.perf_counter() - total_start_time) * 1000 / test_cnt
        print(f"average_draft_time {average_draft_time}")

    if spd_forward:
        print("SPD forwarding...")
        total_start_time = time.perf_counter()
        for _ in range(5):
            SPD_token_judge(
                batch_input_ids=input_ids,
                draft_ptuning_args=draft_ptuning_args,
                draft_extra_ids=draft_extra_ids,
                target_ptuning_args=ptuning_args,
                target_extra_ids=extra_ids,
                tokenizer=tokenizer,
                target_runner=model.language_model,
                draft_runner=draft_model.language_model,
                end_id=generation_config["eos_token_id"],
                pad_id=2,
                top_k=g_top_k,
                top_p=g_top_p,
                max_len=generation_config["max_new_tokens"],
                input_length=input_lengths,
            )
        speculative_time = (time.perf_counter() - total_start_time) * 1000 / test_cnt
        print(f"speculative_time {speculative_time}")


if __name__ == "__main__":
    print('**************采用CPP后端并开启单模型推理******************')
    main(unimodel_forward=True, spd_forward=True, use_cpp=True, kvcache_reuse=True)
    print('**************采用CPP后端并关闭单模型推理，直接进行SPD推理******************')
    main(unimodel_forward=False, spd_forward=True, use_cpp=True, kvcache_reuse=True)
    print('**************采用Python后端并开启单模型推理******************')
    main(unimodel_forward=True, spd_forward=True, use_cpp=False, kvcache_reuse=True)
    print('**************采用CPP后端并关闭kv_cache_reuse******************')
    main(unimodel_forward=True, spd_forward=True, use_cpp=True, kvcache_reuse=False)


'''
代码包含三部分，分别是大模型推理（723-747行）、小模型推理（777-799行）和大小模型SPD协同推理（801-822行），每个部分均推理多轮。

需先改686行模型的根目录（模型已随附件压缩包发送）

目前遇到的问题是：

1.采用python后端推理时，大模型小模型以及SPD协同推理一切正常。
当开启cpp后端后，大模型第一次推理出现乱码，第二次推理不是乱码但是也与python后端协同推理的结果不一致。
（开关：709和767行）

在SPD协同推理第一轮出问题的情况下尝试debug，发现target_model输出的logits被大量置为0，且没有任何报错信息。怀疑是命中了某些不正确的kv_cache_reuse。


2.开启cpp后端，但是关闭大模型和小模型的推理时，SPD协同推理正常，与python后端协同推理的结果一致。

3.开启cpp后端后，关闭kv_cache_reuse，会出现out of memory。



协同推理部分是采用小模型进行step-by-step的预先生成，大模型在此基础上进行并行的验证，修正错误，以此来规避大模型step-by-step生成较慢的问题。
伪代码：
1.使用贪心解码与辅助模型生成一定数量的候选token。当第一次调用辅助生成时，生成的候选token的数量被初始化为5。
2.使用我们的模型，对候选token进行前向计算，获得每个token对应的概率。
3.使用token选择方法（使用.argmax()进行贪心搜索）来从概率中选取 next_tokens。
4.比较步骤3中选择的next_tokens和候选token中相同的token数量。请注意，我们需要从左到右进行比较，在第一次不匹配后，后续
所有候选token都无效。
5.使用步骤4得到的匹配数量将候选token分割。也就是，将输入tokens加上刚刚验证得到的正确的tokens。
6.调整下一次选代中生成的候选token的数量—使用启发式方法，如果步骤3中所有token都匹配，则候选token的长度增加2，否
则减少1。

'''
