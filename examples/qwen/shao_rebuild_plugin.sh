


pushd /TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/cpp/build/
make -j nvinfer_plugin_tensorrt_llm > /TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/examples/qwen/build_plugin.log 2>&1
popd

cp /TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/cpp/build/tensorrt_llm/libtensorrt_llm.so \
   /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/libtensorrt_llm.so
cp /TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so \
   /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so

cp /TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/tensorrt_llm/layers/mlp.py \
   /usr/local/lib/python3.10/dist-packages/tensorrt_llm/layers/mlp.py
cp /TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/tensorrt_llm/layers/attention.py \
   /usr/local/lib/python3.10/dist-packages/tensorrt_llm/layers/attention.py

cp /TRT/GPTQ_CheckPoints_GPT_NeoX/latest_trtllm_github/tensorrt_llm/runtime/generation.py \
   /usr/local/lib/python3.10/dist-packages/tensorrt_llm/runtime/generation.py


# python test_weight_only_groupwise_quant_matmul.py > run.log

# nsys profile --gpu-metrics-device=0 python test_weight_only_groupwise_quant_matmul.py > run.log

# compute-sanitizer --tool memcheck python test_weight_only_groupwise_quant_matmul.py > memcheck.log 2>&1

# cp  ./libnvinfer_plugin_tensorrt_llm_backup.so /TRT/GPTQ_CheckPoints_GPT_NeoX/trtllm_github/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so