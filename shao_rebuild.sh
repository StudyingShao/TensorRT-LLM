
read -p "task id: " task


if [ $task -eq 0 ]
then
    read -p "clean? 0 no 1 yes: " clean
    git submodule update --init --recursive

    if [ $clean -eq 0 ]
    then
      # ./scripts/build_wheel.py --cuda_architectures "80-real;89-real;90-real" --trt_root /usr/local/tensorrt > build.log 2>&1
      ./scripts/build_wheel.py --cuda_architectures "89-real" --trt_root /usr/local/tensorrt > build.log 2>&1
    elif [ $clean -eq 1 ]
    then
      # ./scripts/build_wheel.py --clean --cuda_architectures "80-real;89-real;90-real" --trt_root /usr/local/tensorrt > build.log 2>&1
      ./scripts/build_wheel.py --clean --cuda_architectures "89-real" --trt_root /usr/local/tensorrt > build.log 2>&1
    fi

    echo y | pip uninstall tensorrt-llm
elif [ $task -eq 1 ]
then
    # nvidia-smi -q -d SUPPORTED_CLOCKS
    nvidia-smi -ac "9001,2520"
    # pip install ./build/tensorrt_llm-0.13.0.dev2024090300-cp310-cp310-linux_x86_64.whl # baiduVIS 2024.10.29
    pip install ./build/tensorrt_llm-0.15.0.dev2024101500-cp310-cp310-linux_x86_64.whl # Ant 2024.10.31
    
    # pip install ./build/tensorrt_llm-0.14.0.dev2024100800-cp310-cp310-linux_x86_64.whl
    pip install parameterized
else
  echo "------------------------------------wrong task id------------------------------------"
fi
