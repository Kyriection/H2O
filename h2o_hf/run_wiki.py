


import argparse
import json
from tqdm import tqdm
import torch
import copy
import os
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils_lm_eval.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter



parser = argparse.ArgumentParser()
parser.add_argument("--heavy_ratio", type=float, default=0.1)
parser.add_argument("--recent_ratio", type=float, default=0.1)
parser.add_argument("--context_size", type=int, default=1024)
parser.add_argument("--window_size", type=int, default=1024)
parser.add_argument("--num_samples", type=int, default=1)
args = parser.parse_args()


device = "cuda"
# dataset_name = 'wikitext'
# task = 'wikitext-2-raw-v1'
dataset_name = 'pg19'
task = 'default'
split = 'test'
model_name_or_path = 'huggyllama/llama-7b'
cache_dir = '../hf_cache'

data = load_dataset(dataset_name, task, split=split)

config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)

if args.heavy_ratio + args.recent_ratio < 1:
    print('Enable H2O')
    config.heavy_ratio = args.heavy_ratio
    config.recent_ratio = args.recent_ratio
    checkpoint = copy.deepcopy(model.state_dict())
    model = convert_kvcache_llama_heavy_recent(model, config)
    model.load_state_dict(checkpoint)


model.half().eval().cuda()

text = "\n\n".join(data["text"][: args.num_samples])
encodings = tokenizer(text, return_tensors="pt")

max_length = args.context_size
stride = args.window_size
seq_len = encodings.input_ids.size(1)
print(f"seq_len: {seq_len}")

nlls = []
prev_end_loc = 0
visulize = True
pbar = tqdm(range(0, seq_len, stride))
for begin_loc in pbar:
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop

    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

    if visulize:
        for name, m in model.named_modules():
            if isinstance(m, LlamaAttention_heavy_hitter):
                plt.subplot(1,4,1)
                head_idx = 0
                plt.imshow(m.mask_hh[0, head_idx,:,:].int().cpu().numpy())
                plt.xticks([], [])
                plt.yticks([], [])
                plt.title('Head {}'.format(head_idx))
                plt.subplot(1,4,2)
                head_idx = 5
                plt.imshow(m.mask_hh[0, head_idx,:,:].int().cpu().numpy())
                plt.xticks([], [])
                plt.yticks([], [])
                plt.title('Head {}'.format(head_idx))
                plt.subplot(1,4,3)
                head_idx = 10
                plt.imshow(m.mask_hh[0, head_idx,:,:].int().cpu().numpy())
                plt.xticks([], [])
                plt.yticks([], [])
                plt.title('Head {}'.format(head_idx))
                plt.subplot(1,4,4)
                head_idx = 25
                plt.imshow(m.mask_hh[0, head_idx,:,:].int().cpu().numpy())
                plt.xticks([], [])
                plt.yticks([], [])
                plt.title('Head {}'.format(head_idx))
                plt.savefig('{}.png'.format(name))
                plt.close()
        visulize = False


    pbar.set_description(
        f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
    )

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
