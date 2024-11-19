

# cp /TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/tensorrt_llm/builder.py /usr/local/lib/python3.10/dist-packages/tensorrt_llm/builder.py
cp /TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/tensorrt_llm/runtime/model_runner_cpp.py /usr/local/lib/python3.10/dist-packages/tensorrt_llm/runtime/model_runner_cpp.py
cp /TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/examples/internVL2/modeling_internvl_chat.py /TRT/llm-models/internVL2/2B/modeling_internvl_chat.py

# python3 simple.py > output.log 2>&1
# python3 simple.py > output_pagedcontext.log 2>&1
python3 simple.py > output_new.log 2>&1