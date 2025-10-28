MODEL=longchat-v1.5-7b-32k
MODEL=Meta-Llama-3.1-8B-Instruct

# Whether enable LongBench-E benchmark
enable_e=0

time=0

OUTPUT_DIR=results_longbench/$MODEL/
OUTPUT_PATH=results_longbench/$MODEL/final-twi

python -u ../LongBench/eval.py --model $MODEL --e $enable_e --t $time --output-dir $OUTPUT_DIR --output-path $OUTPUT_PATH
