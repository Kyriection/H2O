from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel, AutoConfig
from datasets import load_dataset
import torch
from tqdm import tqdm
import argparse
import os
# from longllm.extend import extend_llama
from utils_lm_eval.modify_llama import (
    convert_kvcache_llama_heavy_recent,
)
import copy


device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="models/llama/llama-7b")
parser.add_argument("--dataset_name", type=str, default="wikitext")

parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])

parser.add_argument(
    "--context_size",
    type=int,
    default=512,
)

parser.add_argument(
    "--window_size",
    type=int,
    default=256,
)

parser.add_argument(
    "--num_samples",
    type=int,
    default=1,
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs/debug",
)

parser.add_argument(
    "--extend_ratio",
    type=int,
    default=None,
)

parser.add_argument(
    "--cache_dir",
    type=int,
    default=None,
)

parser.add_argument("--enable_h2o", action="store_true")
parser.add_argument("--heavy_ratio", type=float, default=0.1)
parser.add_argument("--recent_ratio", type=float, default=0.1)

args = parser.parse_args()

data = load_dataset(args.dataset_name, args.task, split=args.split)

model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

# if args.extend_ratio is not None:
#     extend_llama(model, args.extend_ratio)

if args.enable_h2o:
    checkpoint = copy.deepcopy(model.state_dict())
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = convert_kvcache_llama_heavy_recent(
        model,
        config,
        heavy_ratio=args.heavy_ratio,
        recent_ratio=args.recent_ratio,
    )
    model.load_state_dict(checkpoint)

model.half().eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

text = "\n\n".join(data["text"][: args.num_samples])

encodings = tokenizer(text, return_tensors="pt")

max_length = args.context_size + args.window_size
stride = args.window_size
seq_len = encodings.input_ids.size(1)
print(f"seq_len: {seq_len}")

nlls = []
prev_end_loc = 0
pbar = tqdm(range(0, seq_len, stride))
for begin_loc in pbar:
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    pbar.set_description(
        f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
    )

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
# os.makedirs(args.output_dir, exist_ok=True)
# with open(f"{args.output_dir}/ppl.txt", "w") as f:
#     f.write(f"{ppl.item()}\n")
