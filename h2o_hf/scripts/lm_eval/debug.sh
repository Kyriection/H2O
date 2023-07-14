python run_lm.py --dataset_name pg19 \
    --task default \
    --model_name_or_path huggyllama/llama-7b \
    --cache_dir $1 \
    --context_size 256 \
    --enable_h2o \
    --heavy_ratio 0.1 \
    --recent_ratio 0.1 