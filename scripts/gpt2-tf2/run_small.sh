#export HIP_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=4,5,6,7
python3 gpt2_train_distributed.py tf --tf_fp16 1 --learning_rate 5e-5 --output_dir . |& tee -a trace.txt
#python3 gpt2_train_distributed.py tf --learning_rate 5e-5 --output_dir .
#python3 gpt2_train_distributed.py tf --output_dir .
