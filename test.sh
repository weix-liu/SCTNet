export PYTHONPATH=/data/weix/cloudseg-main

#python src/eval/eval_single_model_dataset.py --dataset_name=hrc_whu --model_name=sctnet --model_path=/data/weix/cloudseg-main/logs/hrc_whu/sctnet/2025-03-05_10-06-36/checkpoints/8922last.ckpt --gpu="cuda:1"
python src/eval/eval_single_model_dataset.py --dataset_name=gf12ms_whu_gf1 --model_name=sctnet --model_path=/data/weix/cloudseg-main/logs/gf12ms_whu_gf1/sctnet/2025-03-05_12-16-47/checkpoints/last.ckpt --gpu="cuda:1"