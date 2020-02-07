source activate mmlab
# ,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=3,4,6,7 python ./mmdetection/tools/train_custom.py /home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_bdd_agg_fuse_st.py --gpus 4
# CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/test_custom.py