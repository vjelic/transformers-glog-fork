export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,12,13
python3 gpt2_train_distributed.py tf --gpus 8 --output_dir .
