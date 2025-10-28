MODEL=Meta-Llama-3.1-8B-Instruct
MODELPATH=/workspace/models/Meta-Llama-3.1-8B-Instruct

OUTPUT_DIR=results_longbench/$MODEL

# Use HF load_dataset. Modify this if you want to use local data set.
DATASET_PATH="HF"

CONFIG_PATH=LongBench/config/

mkdir -p $OUTPUT_DIR

# algo_cfg_path=./config_snapkv.json
# algo_cfg_path=./config_quest_8192.json
# algo_cfg_path=./config_ds_4096.json
# algo_cfg_path=./config_quest_twi.json
algo_cfg_path=$1

# Whether enable LongBench-E benchmark
enable_e=0

# ignore: vcsum multifieldqa_zh lsht passage_retrieval_zh dureader samsum passage_count trec
datasets=('narrativeqa' 'qasper' 'multifieldqa_en' 'hotpotqa'
          '2wikimqa' 'musique' 'gov_report' 'qmsum' 'multi_news'
          'triviaqa' 'passage_retrieval_en' 'lcc' 'repobench-p')

dataset_e=('qasper' 'multifieldqa_en' 'hotpotqa' '2wikimqa' 'gov_report'
           'multi_news' 'trec' 'triviaqa' 'passage_count'
           'passage_retrieval_en' 'lcc' 'repobench-p')

if [ "$enable_e" -eq 0 ]; then
  benchmark_dataset=("${datasets[@]}")
else
  benchmark_dataset=("${dataset_e[@]}")
fi

month=$(date +"%m")
day=$(date +"%d")
hour=$(date +"%H")
minute=$(date +"%M")
time="${month}${day}${hour}${minute}"

for task in "${benchmark_dataset[@]}"
# for task in "narrativeqa" "hotpotqa"
# for task in "qasper" "hotpotqa" "gov_report" "passage_count"
# for task in "triviaqa"
# for task in "qasper"
# for task in "triviaqa" "multifieldqa_en"
# for task in "hotpotqa" "gov_report" "triviaqa" "narrativeqa" "multifieldqa_en"
# for task in "gov_report"
# for task in "hotpotqa" "gov_report" "triviaqa"
# for task in "qasper" "narrativeqa" "multifieldqa_en" "hotpotqa" "triviaqa" "gov_report"
# for task in "narrativeqa"
# for task in "qmsum" "multi_news" "triviaqa" "passage_retrieval_en" "lcc" "repobench-p"
# for task in "qmsum" "multi_news" "repobench-p"
do
    echo "Running $task:"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u LongBench/pred.py \
        --model $MODEL --model_path $MODELPATH --task $task \
        --algo-config-path $algo_cfg_path \
        --e $enable_e --t $time \
        --dataset-path $DATASET_PATH \
        --output-dir $OUTPUT_DIR \
        --config-path $CONFIG_PATH
    # for budget in 512 1024 2048 4096
    # do
    #     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u ../LongBench/pred.py \
    #         --model $MODEL --model_path $MODELPATH --task $task \
    #         --algo $algo$budget \
    #         --e $enable_e --t $time \
    #         --dataset-path $DATASET_PATH \
    #         --output-dir $OUTPUT_DIR \
    #         --config-path $CONFIG_PATH
    # done
done

python -u LongBench/eval.py --model $MODEL --e $enable_e --t $time --output-dir $OUTPUT_DIR
