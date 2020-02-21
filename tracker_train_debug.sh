source activate mmlab
CUDA_VISIBLE_DEVICES=7 python ./mmdetection/tools/train_reppoint.py /home/ld/RepPoints/configs/reppoint_baseline_do3.py --gpus 1
# CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/test_custom.py
# 0,1,2,3,4,5,6,7
# 4,5,6,7
