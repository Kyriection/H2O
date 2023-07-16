CUDA_VISIBLE_DEVICES=0 nohup bash scripts/ablation/ex0.sh > log_ex0.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ablation/ex1.sh > log_ex1.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/ablation/ex2.sh > log_ex2.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/ablation/ex3.sh > log_ex3.out 2>&1 &