import mmcv
offset=mmcv.load('/home/ld/RepPoints/ld_result/reppoint_do3/epoch_23_thres0.3_nms0.3/refer/reppoints.pkl')
print(offset[0][0].shape)
