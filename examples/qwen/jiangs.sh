
cp /TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/tensorrt_llm/models/modeling_utils.py /usr/local/lib/python3.10/dist-packages/tensorrt_llm/models/modeling_utils.py

tp_size=1
pp_size=1
# qformat=int8_wo
# kv_cache_dtype=int8
qformat=w4a8_awq
kv_cache_dtype=fp8


echo -e "0 fp16 \n1 bf16"
read -p "dtype: " dtype

if [ $dtype -eq 0 ]
then
    dtype=float16
elif [ $dtype -eq 1 ]
then
    dtype=bfloat16
fi

# model_name=Qwen2-72B-Instruct
# ROOT_DIR=/TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_internal/examples/qwen
# MODEL_DIR=/llm-models/Qwen2-72B-Instruct

model_name=Qwen2-7B-Instruct
ROOT_DIR=/TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/examples/qwen
MODEL_DIR=/llm-models/Qwen2-7B-Instruct

CKPT_DIR=${ROOT_DIR}/jiangs/ckpt/${model_name}_${dtype}/${tp_size}/${qformat}_kv${kv_cache_dtype}
ENGINE_DIR=${ROOT_DIR}/jiangs/engines/${model_name}_${dtype}/${tp_size}/${qformat}_kv${kv_cache_dtype}

max_input_len=1024
max_seq_len=2048
max_output_len=200
max_batch_size=1
num_gpus=${tp_size}

mkdir -p ${CKPT_DIR}
mkdir -p ${ENGINE_DIR}


echo -e "0 convert \n1 quantize \n2 build_engine\n3 run"
read -p "task: " task
git submodule update --init --recursive

if [ $task -eq 0 ]
then
    python convert_checkpoint.py \
        --model_dir ${MODEL_DIR} \
        --output_dir ${CKPT_DIR} \
        --dtype ${dtype} \
        --tp_size ${tp_size} \
        --pp_size ${pp_size} \
        > ${CKPT_DIR}/convert.log 2>&1
elif [ $task -eq 1 ]
then
    python ../quantization/quantize.py \
        --model_dir ${MODEL_DIR} \
        --output_dir ${CKPT_DIR} \
        --tp_size ${tp_size} \
        --pp_size ${pp_size} \
        --dtype ${dtype} \
        --qformat ${qformat:int8_wo} \
        --kv_cache_dtype ${kv_cache_dtype:int8} \
        --awq_block_size 128 \
        --calib_size ${calib_size:-32} \
        --calib_dataset ${calib_dataset:-cnn_dailymail} \
        > ${CKPT_DIR}/quantize.log 2>&1
elif [ $task -eq 2 ]
then
    trtllm-build \
        --checkpoint_dir ${CKPT_DIR} \
        --output_dir ${ENGINE_DIR} \
        --workers ${num_gpus} \
        --max_input_len ${max_input_len} \
        --max_seq_len ${max_seq_len} \
        --max_batch_size ${max_batch_size} \
        --paged_kv_cache ${paged_kv_cache:-enable} \
        --context_fmha ${context_fmha:-enable} \
        --use_paged_context_fmha ${use_paged_context_fmha:-disable} \
        --use_fp8_context_fmha ${use_fp8_context_fmha:-disable} \
        --gpt_attention_plugin ${dtype} \
        --gemm_plugin ${dtype} \
        --enable_xqa ${enable_xqa:-enable} \
        --enable_debug_output \
        > ${ENGINE_DIR}/trtllm-build.log 2>&1
elif [ $task -eq 3 ]
then
    echo -e "run ${model_name} ${dtype} TP${tp_size} ${qformat} kv${kv_cache_dtype}"

    python3 ../run.py \
        --tokenizer_dir ${MODEL_DIR} \
        --engine_dir ${ENGINE_DIR} \
        --input_text "你好，请问你叫什么？" \
        --max_output_len ${max_output_len} \
        --use_py_session \
        --debug_mode \
        > ${ENGINE_DIR}/run.log 2>&1
fi






   
        
# trtllm-build --checkpoint_dir '/mnt/modelops/314373/trt_llm/20241018/ckpt/Bailing_V4_80B_4K_Chat_w8a8kv8_tp4' \
#             --output_dir '/mnt/modelops/314373/trt_llm/20241018/l20/Bailing_V4_80B_4K_Chat_w8a8kv8_tp4'\
#             --gemm_plugin bfloat16\
#             --use_paged_context_fmha 'enable' \
#             --use_fp8_context_fmha 'enable'\
#             --multiple_profiles 'enable'\
#             --max_batch_size 128\
#             --max_seq_len 8192\
#             --max_num_tokens 2048  2>&1  | tee build.log







