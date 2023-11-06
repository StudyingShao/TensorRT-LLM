# nvidia-smi dmon -s pc
# nvidia-smi -q -d SUPPORTED_CLOCKS
# nvidia-smi -ac "9001,2520" -i "1,2,3,4,5,6,7"

# GPU       : L40S
# Memory    : 9001 MHz
# Graphics  : 2520 MHz


# python build.py --model_dir ./codellama_34b_model \
#                 --quant_ckpt_path ./codellama-34b-4bit-gs128.safetensors \
#                 --dtype float16 \
#                 --remove_input_padding \
#                 --use_gpt_attention_plugin float16 \
#                 --enable_context_fmha \
#                 --use_gemm_plugin float16 \
#                 --use_rmsnorm_plugin float16 \
#                 --rotary_base 1000000 \
#                 --vocab_size 32000 \
#                 --use_weight_only \
#                 --weight_only_precision int4_gptq \
#                 --per_group \
#                 --world_size 1 \
#                 --tp_size 1 \
#                 --max_batch_size 16 \
#                 --max_input_len 1024 \
#                 --max_output_len 512 \
#                 --output_dir ./codellama_34b/tp_1_b16_i1024_o512/

# python build.py --model_dir ./codellama_34b_model \
#                 --quant_ckpt_path ./codellama-34b-4bit-gs128.safetensors \
#                 --dtype float16 \
#                 --remove_input_padding \
#                 --use_gpt_attention_plugin float16 \
#                 --enable_context_fmha \
#                 --use_gemm_plugin float16 \
#                 --use_rmsnorm_plugin float16 \
#                 --rotary_base 1000000 \
#                 --vocab_size 32000 \
#                 --use_weight_only \
#                 --weight_only_precision int4_gptq \
#                 --per_group \
#                 --world_size 2 \
#                 --tp_size 2 \
#                 --max_batch_size 16 \
#                 --max_input_len 1024 \
#                 --max_output_len 512 \
#                 --output_dir ./codellama_34b/tp_2_b16_i1024_o512/

# python build.py --model_dir ./codellama_34b_model \
#                 --quant_ckpt_path ./codellama-34b-4bit-gs128.safetensors \
#                 --dtype float16 \
#                 --remove_input_padding \
#                 --use_gpt_attention_plugin float16 \
#                 --enable_context_fmha \
#                 --use_gemm_plugin float16 \
#                 --use_rmsnorm_plugin float16 \
#                 --rotary_base 1000000 \
#                 --vocab_size 32000 \
#                 --use_weight_only \
#                 --weight_only_precision int4_gptq \
#                 --per_group \
#                 --world_size 4 \
#                 --tp_size 4 \
#                 --max_batch_size 16 \
#                 --max_input_len 1024 \
#                 --max_output_len 512 \
#                 --output_dir ./codellama_34b/tp_4_b16_i1024_o512/

echo "--------------------------------------- tp 1 ---------------------------------------"
for b in 1 2 4 8 16
do
  for i in 16 64 256 1024
  do
    o=`expr $i / 2`
    echo "case" $b $i $o
    python3 run.py --max_output_len=$o \
               --input_len=$i \
               --batch_size=$b \
               --tokenizer_dir ./codellama_34b_model \
               --engine_dir=./codellama_34b/tp_1_b16_i1024_o512/
  done
done

echo "--------------------------------------- tp 2 ---------------------------------------"
for b in 1 2 4 8 16
do
  for i in 16 64 256 1024
  do
    o=`expr $i / 2`
    echo "case" $b $i $o
    mpirun -n 2 --allow-run-as-root \
    python3 run.py --max_output_len=$o \
               --input_len=$i \
               --batch_size=$b \
               --tokenizer_dir ./codellama_34b_model \
               --engine_dir=./codellama_34b/tp_2_b16_i1024_o512/
  done
done

echo "--------------------------------------- tp 4 ---------------------------------------"
for b in 1 2 4 8 16
do
  for i in 16 64 256 1024
  do
    o=`expr $i / 2`
    echo "case" $b $i $o
    mpirun -n 4 --allow-run-as-root \
    python3 run.py --max_output_len=$o \
               --input_len=$i \
               --batch_size=$b \
               --tokenizer_dir ./codellama_34b_model \
               --engine_dir=./codellama_34b/tp_4_b16_i1024_o512/
  done
done