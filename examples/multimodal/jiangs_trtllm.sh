
DTYPE=bfloat16

echo "0 pip install"
echo "1 trtllm convert"
echo "2 trtllm build"
echo "3 HF run"
echo "4 trtllm run"
echo "5 trtllm debug"
read -p "task? " task
if [ $task -eq 0 ]
then

    pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate
    pip install qwen-vl-utils


elif [ $task -eq 1 ]
then

    # Convert
    python3 ../qwen/convert_checkpoint.py \
            --model_dir=/llm-models/Qwen2-VL-7B-Instruct/ \
            --output_dir=./jiangs/ckpt/Qwen2-VL-7B-Instruct/${DTYPE} \
            --dtype ${DTYPE} \
            > jiangs_convert.log 2>&1


elif [ $task -eq 2 ]
then

    cp /TRT/GPTQ_CheckPoints_GPT_NeoX/trtllm_github_0170release/tensorrt_llm/models/qwen/model.py \
    /usr/local/lib/python3.12/dist-packages/tensorrt_llm/models/qwen/model.py

    cp /TRT/GPTQ_CheckPoints_GPT_NeoX/trtllm_github_0170release/tensorrt_llm/models/modeling_utils.py \
    /usr/local/lib/python3.12/dist-packages/tensorrt_llm/models/modeling_utils.py

    # Build TensorRT-LLM engine             
    trtllm-build --checkpoint_dir ./jiangs/ckpt/Qwen2-VL-7B-Instruct/${DTYPE} \
                --output_dir ./jiangs/engine/Qwen2-VL-7B-Instruct/${DTYPE} \
                --gather_all_token_logits \
                --gemm_plugin=${DTYPE} \
                --gpt_attention_plugin=${DTYPE} \
                --max_batch_size=4 \
                --max_input_len=2048 --max_seq_len=3072 \
                --max_multimodal_len=4096 \
                --enable_debug_output \
                > jiangs_build.log 2>&1

    # Generate TensorRT engines for visual components and combine everything into final pipeline.

    # python build_visual_engine.py --model_type qwen2_vl --model_path /llm-models/Qwen2-VL-7B-Instruct/

elif [ $task -eq 3 ]
then

    cp ./HF_files/utils.py /usr/local/lib/python3.12/dist-packages/transformers/generation/utils.py
    cp ./HF_files/modeling_qwen2_vl.py /usr/local/lib/python3.12/dist-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py

    python jiangs_HF.py > jiangs_run_HF.log 2>&1

elif [ $task -eq 4 ]
then

    cp /TRT/GPTQ_CheckPoints_GPT_NeoX/trtllm_github_0170release/tensorrt_llm/runtime/multimodal_model_runner.py \
    /usr/local/lib/python3.12/dist-packages/tensorrt_llm/runtime/multimodal_model_runner.py

    cp /TRT/GPTQ_CheckPoints_GPT_NeoX/trtllm_github_0170release/tensorrt_llm/runtime/model_runner_cpp.py \
    /usr/local/lib/python3.12/dist-packages/tensorrt_llm/runtime/model_runner_cpp.py

    python3 run.py \
        --hf_model_dir /llm-models/Qwen2-VL-7B-Instruct/ \
        --llm_engine_dir ./jiangs/engine/Qwen2-VL-7B-Instruct/${DTYPE} \
        --visual_engine_dir ./tmp/trt_engines/vision_encoder \
        --image_path ./pics/qwen2vl_pic1.jpg \
        --max_new_tokens 512 \
        --enable_context_fmha_fp32_acc \
        --input_text "请输出图中的文字" \
        > jiangs_run_${DTYPE}.log 2>&1
        # --use_py_session \

elif [ $task -eq 5 ]
then

    python jiangs_trtllm_debug.py > jiangs_trtllm_debug.log

fi





