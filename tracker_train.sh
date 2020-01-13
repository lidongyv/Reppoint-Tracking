source activate mmlab
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python ./mmdetection/tools/train_custom.py /home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_kitti_agg_fuse_st.py --gpus 5
# bash ./mmdetection/tools/dist_train_custom.sh /home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_kitti_agg.py 8