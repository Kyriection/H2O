python -u run_wiki.py \
    --heavy_ratio 1 \
    --recent_ratio 1 \
    --context_size 256 \
    --window_size 256 

python -u run_wiki.py \
    --heavy_ratio 1 \
    --recent_ratio 1 \
    --context_size 1024 \
    --window_size 1024

python -u run_wiki.py \
    --heavy_ratio 0 \
    --recent_ratio 0.25 \
    --context_size 1024 \
    --window_size 1024

python -u run_wiki.py \
    --heavy_ratio 0.125 \
    --recent_ratio 0.125 \
    --context_size 1024 \
    --window_size 1024
