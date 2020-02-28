source activate mmlab
bash ./mmdetection/tools/dist_train.sh /home/ld/RepPoints/configs/reppoint_coco_moment_withmask.py 8
# bash ./mmdetection/tools/dist_train.sh /home/ld/RepPoints/configs/retinanet_r101_fpn_kitti.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./mmdetection/tools/train_gpus.py /home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_kitti_agg_fuse_st.py --gpus 8