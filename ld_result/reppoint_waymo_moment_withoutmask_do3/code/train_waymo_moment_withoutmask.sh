source activate mmlab
CUDA_VISIBLE_DEVICES=4,5,6,7 python ./mmdetection/tools/train_reppoint_zxn.py /home/ld/RepPoints/configs/reppoint_waymo_moment_withoutmask_baseline_do3.py --gpus 4
# CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/test_custom.py
# 0,1,2,3,4,5,6,7
# 4,5,6,7