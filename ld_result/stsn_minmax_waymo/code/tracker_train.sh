source activate mmlab
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./mmdetection/tools/train_reppoint.py /home/ld/RepPoints/configs/reppoint_waymo_minmax_stsn.py --gpus 8
# CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/test_custom.py
# 0,1,2,3,4,5,6,7
# 4,5,6,7
