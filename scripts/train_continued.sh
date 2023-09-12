
echo $EXP-cntd
echo $MODEL_TYPE
echo $VID
echo $CKPT_PATH
# EXP=$VID-$MODEL_TYPE
echo $EXP

export EPOCHS=10; CUDA_VISIBLE_DEVICES=0 python train.py \
    exp_name=$EXP-cntd num_epochs=$EPOCHS vid=$VID scale=1 \
    with_radialdist=1 ckpt_path=$CKPT_PATH model_type=$MODEL_TYPE

