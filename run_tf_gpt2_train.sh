set -eux

export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

CURRENTDATE=`date +"%Y-%m-%d-%T"`

MODEL_SIZE="Small"    # Small 117M model
# MODEL_SIZE="Medium"   # Medium 345M model
# MODEL_SIZE="Large"    # Large 762M model
# MODEL_SIZE="XL"       # XL 1542M model

DATA_PATH="/data/tf-gpt-2/data/"

NUM_EPOCHS=1

TRUNCATE=1

USE_FP16=1

python3 scripts/gpt2-tf2/gpt2_train.py \
	$MODEL_SIZE \
	$DATA_PATH \
	$NUM_EPOCHS\
	$TRUNCATE \
	$USE_FP16 \
	
	# 2>&1 | tee run.log.${CURRENTDATE}
