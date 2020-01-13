source activate mmlab
CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/eval_agg.py
# bash ./mmdetection/tools/dist_train_custom.sh /home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_kitti_agg.py 8