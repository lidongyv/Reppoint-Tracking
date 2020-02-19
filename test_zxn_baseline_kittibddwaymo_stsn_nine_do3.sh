source activate mmlab
# python ./mmdetection/tools/test.py /home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_kitti_mt.py /home/ld/RepPoints/work_dirs/reppoints_moment_r101_dcn_fpn_kitti_mt_class3/epoch_30.pth --out /home/ld/RepPoints/out/reppoint_class3/test.pkl --eval bbox #--show
CUDA_VISIBLE_DEVICES=7 python /home/zxn/RepPoints/mmdetection/tools/test_custom_zxn_kittibddwaymo_stsn_nine_do3.py
# python ./mmdetection/tools/kitti_eval.py /home/ld/RepPoints/out/reppoint/test.pkl  /home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_kitti.py 


