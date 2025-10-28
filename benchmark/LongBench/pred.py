# Modified from: https://github.com/mit-han-lab/Quest/blob/main/evaluation/LongBench/pred.py

import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse

from twilight.pyimpl import enable_sparse_attention


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "llama2-7b-chat-4k",
            "longchat-v1.5-7b-32k",
            "xgen-7b-8k",
            "internlm-7b-8k",
            "chatglm2-6b",
            "chatglm2-6b-32k",
            "chatglm3-6b-32k",
            "vicuna-v1.5-7b-16k",
            "Mistral-7B-Instruct-v0.3",
            "Meta-Llama-3.1-8B-Instruct",
        ],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="If model has been downloaded locally",
    )
    parser.add_argument("--e", type=int, default=0, help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", required=True)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--t", type=str, default="benchmark launch time")
    parser.add_argument(
        "--algo-config-path",
        type=str,
        help="Algorithm config path",
    )

    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "Llama-3.1" in model_name:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|> {prompt} <|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
    budget_info,
    score_info,
):
    preds = []

    # HACK: sample 10 examples for debugging
    # sampled = np.random.choice(data, 10)

    for json_obj in tqdm(data):
        torch.cuda.empty_cache()
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        # split the prompt and question (simulate decoding in the question stage)
        if dataset in ["qasper", "hotpotqa"]:
            q_pos = prompt.rfind("Question:")
        elif dataset in ["multifieldqa_en", "gov_report"]:
            q_pos = prompt.rfind("Now,")
        elif dataset in ["triviaqa"]:
            q_pos = prompt.rfind("Answer the question")
        elif dataset in ["narrativeqa"]:
            q_pos = prompt.rfind("Do not provide")
        else:
            q_pos = -1

        # max simulation length is 100
        q_pos = max(len(prompt) - 100, q_pos)

        if q_pos != None:
            question = prompt[q_pos:]
            prompt = prompt[:q_pos]

        if "chatglm3" in model_name:
            # input = prompt.to(device)
            input = prompt.to("cuda")
        else:
            # input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
            q_input = tokenizer(question, truncation=False, return_tensors="pt").to(
                "cuda"
            )
            q_input.input_ids = q_input.input_ids[:, 1:]

        context_length = input.input_ids.shape[-1] + q_input.input_ids.shape[-1]

        if (
            dataset == "samsum"
        ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            # assert False
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ],
            )[0]
        else:
            with torch.no_grad():
                output = model(
                    input_ids=input.input_ids,
                    past_key_values=None,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
                for input_id in q_input.input_ids[0]:
                    output = model(
                        input_ids=input_id.unsqueeze(0).unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values

                pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content = [pred_token_idx.item()]
                for _ in range(max_gen - 1):
                    outputs = model(
                        input_ids=pred_token_idx,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                    past_key_values = outputs.past_key_values
                    pred_token_idx = (
                        outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    )
                    generated_content += [pred_token_idx.item()]
                    if pred_token_idx.item() == tokenizer.eos_token_id:
                        break

            # output = model.generate(
            #     **input,
            #     max_new_tokens=max_gen,
            #     num_beams=1,
            #     do_sample=False,
            #     temperature=1.0,
            # )[0]

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        # pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)

        # print("Score: ", flush=True)
        # score_info.print_min_sum_single_query()

        # Record budget
        avg_budget = budget_info.get_total_avg_budget()
        avg_score = score_info.get_total_avg_score()
        budget_info.reset()
        score_info.reset()
        # avg_B0 = budget_info.get_total_avg_budget_B0()

        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
                "budget": avg_budget,
                "score_sum": avg_score,
                # "B0": avg_B0,
                # "B1": avg_budget,
            }
        )
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device, args):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = model.eval()

    with open(args.algo_config_path, "r") as f:
        algo_config = json.load(f)

    budget_info, score_info = enable_sparse_attention(
        model,
        sparse_config=algo_config,
        enable_budget_info=True,
        enable_score_info=True,
    )

    return model, tokenizer, algo_config, budget_info, score_info


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    config_path = args.config_path
    model2maxlen = json.load(open(f"{config_path}/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define your model
    model_name = args.model
    if args.model_path is None:
        model2path_list = json.load(open(f"{config_path}/model2path.json", "r"))
        model2path = model2path_list[model_name]
    else:
        model2path = args.model_path
    model, tokenizer, algo_config, budget_info, score_info = load_model_and_tokenizer(
        model2path, model_name, device, args
    )
    max_length = model2maxlen[model_name]

    datasets = [args.task]
    dataset2prompt = json.load(open(f"{config_path}/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(f"{config_path}/dataset2maxlen.json", "r"))

    if args.e:
        dir = f"{args.output_dir}/pred_e"
        if not os.path.exists(dir):
            os.makedirs(dir)
    else:
        dir = f"{args.output_dir}/pred"
        if not os.path.exists(dir):
            os.makedirs(dir)

    local_data_set = args.dataset_path
    has_local_data_set = os.path.exists(local_data_set)
    for dataset in datasets:
        if args.e:
            if not has_local_data_set:
                data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
            else:
                data = load_dataset(
                    "json", data_files=f"{local_data_set}/{dataset}_e.jsonl", split="train"
                )
        else:
            if not has_local_data_set:
                data = load_dataset("THUDM/LongBench", dataset, split="test")
            else:
                data = load_dataset(
                    "json", data_files=f"{local_data_set}/{dataset}.jsonl", split="train"
                )

        # Create output.jsonl file name
        dataset_prefix = dataset.split("-")[0]

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(
            model,
            tokenizer,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            device,
            model_name,
            budget_info,
            score_info,
        )

        if "config_path" in algo_config["selector"]:
            algo_config["selector"].pop("config_path")
        out_fn = f"{dataset_prefix}-{str(algo_config)}-{args.t}"
        # avoid too long file name
        out_fn = out_fn.replace(" ", "")
        out_fn = out_fn.replace("'", "")
        if len(out_fn) > 245:
            out_fn = out_fn[:245] + "..."
        out_path = f"{dir}/{out_fn}.jsonl"

        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
