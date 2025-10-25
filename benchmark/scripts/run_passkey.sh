MODEL=longchat-7b-v1.5-32k
MODELPATH=/data/chaofan/llm/longchat-7b-v1.5-32k

OUTPUT_DIR=results_passkey/$MODEL

mkdir -p $OUTPUT_DIR

length=13000
algo_cfg_path=.configs/config_quest_1024.json

python3 ../passkey/passkey.py -m $MODELPATH \
    --iterations 100 --fixed-length $length \
    --algo-config-path $algo_cfg_path \
    --output-dir $OUTPUT_DIR