
echo VID: $VID
echo MODEL_TYPE: $MODEL_TYPE
echo DIR_CKPTS: $DIR_CKPTS

python evaluate.py eval2.is_debug=0 vid=$VID eval2.dir_dst=outputs/results/$VID-$MODEL_TYPE eval2.ckpt=$DIR_CKPTS/$VID-$MODEL_TYPE
