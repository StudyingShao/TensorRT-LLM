

78f5c2936b7bdaa56859075a3f2fcc5a63952134
78f5c2936b7bdaa56859075a3f2fcc5a63952134
bfloat16



git diff --no-index -U9999 ./tensorrt_llm/runtime/model_runner_cpp.py ./examples/internVL2/model_runner_cpp.py > model_runner_cpp.diff

pip install timm
pip install decord


python convert_checkpoint.py --model_dir /TRT/llm-models/internVL2/2B/ \
                --dtype bfloat16 \
                --output_dir ./jiangs/ckpt/2B/bf16/1-gpu/ \
                > ./jiangs/convert_checkpoint_2B_bf16_1gpu.log 2>&1 

trtllm-build --checkpoint_dir ./jiangs/ckpt/2B/bf16/1-gpu/ \
             --output_dir ./jiangs/trt_engines/2B/bf16/1-gpu_pagedcontext/ \
             --max_batch_size 1 \
             --max_input_len 4096 \
             --gather_all_token_logits \
             --context_fmha enable \
             --paged_kv_cache enable \
             --use_paged_context_fmha enable \
             --max_multimodal_len 4096 \
             --log_level=verbose \
             > ./jiangs/trtllm-build_2B_bf16_1gpu_pagedcontext.log 2>&1 

python3 simple.py > output.log 2>&1


# 关掉 use_paged_context_fmha
trtllm-build --checkpoint_dir ./jiangs/ckpt/2B/bf16/1-gpu/ \
             --output_dir ./jiangs/trt_engines/2B/bf16/1-gpu/ \
             --max_batch_size 1 \
             --max_input_len 4096 \
             --gather_all_token_logits \
             --context_fmha enable \
             --paged_kv_cache enable \
             --max_multimodal_len 4096 \
             --log_level=verbose \
             > ./jiangs/trtllm-build_2B_bf16_1gpu.log 2>&1
结果没问题了


# 模型 layer num  24 -> 1
# 关闭 gather_all_token_logits
trtllm-build --checkpoint_dir ./jiangs/ckpt/2B/bf16/1-gpu/ \
             --output_dir ./jiangs/trt_engines/2B/bf16/1-gpu_pagedcontext_nogather/ \
             --max_batch_size 1 \
             --max_input_len 4096 \
             --context_fmha enable \
             --paged_kv_cache enable \
             --use_paged_context_fmha enable \
             --max_multimodal_len 4096 \
             --log_level=verbose \
             > ./jiangs/trtllm-build_2B_bf16_1gpu_pagedcontextnogather.log 2>&1 