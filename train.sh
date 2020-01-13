source activate mmlab
# bash ./mmdetection/tools/dist_train.sh /home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_waymo_mt.py 5
# bash ./mmdetection/tools/dist_train.sh /home/ld/RepPoints/configs/retinanet_r101_fpn_kitti.py 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python ./mmdetection/tools/train_gpus.py /home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_waymo_mt.py --gpus 5