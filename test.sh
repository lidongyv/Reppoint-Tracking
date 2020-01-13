source activate mmlab
python ./mmdetection/tools/test.py /home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_kitti_mt.py /home/ld/RepPoints/work_dirs/reppoints_moment_r101_dcn_fpn_kitti_mt_class3/epoch_30.pth --out /home/ld/RepPoints/out/reppoint_class3/test.pkl --eval bbox #--show
# python ./mmdetection/tools/test_custom.py /home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_kitti_st.py /home/ld/RepPoints/work_dirs/agg_st_back/epoch_29.pth --out /home/ld/RepPoints/out/stsn_st/test.pkl --eval bbox #--show
# python ./mmdetection/tools/kitti_eval.py /home/ld/RepPoints/out/reppoint/test.pkl  /home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_kitti.py 


