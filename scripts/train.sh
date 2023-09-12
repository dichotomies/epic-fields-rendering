
echo $MODEL_TYPE
echo $VID

EXP_NAME=$VID-$MODEL_TYPE

export EPOCHS=10; CUDA_VISIBLE_DEVICES=0 python train.py \
    exp_name=$VID-$MODEL_TYPE num_epochs=$EPOCHS vid=$VID \
    scale=1 with_radialdist=1 model_type=$MODEL_TYPE
