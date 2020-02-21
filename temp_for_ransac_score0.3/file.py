import mmcv
import json
import numpy as np
reppoints=mmcv.load('/home/ld/RepPoints/temp_for_ransac/reppoints_5_all.pkl')
offset1=reppoints[0][1]
offset2=reppoints[0][2]
# print(reppoints1.shape)
feature=mmcv.load('/home/ld/RepPoints/temp_for_ransac/cls_feat_5_all.pkl')
# for m in range(len(feature)):
#     for n in range(len(feature[0])):
#             print(m,n,feature[m][n].shape)
# 0 0 0 (1, 256, 96, 160)
# 0 0 1 (1, 256, 48, 80)
# 0 0 2 (1, 256, 24, 40)
# 0 0 3 (1, 256, 12, 20)
# 0 0 4 (1, 256, 6, 10)
# print(len(feature),len(feature[0]),len(feature[0][0]),feature[0][0][0].shape)
# 5 5 5 (1, 256, 96, 160)
#82 frame
feature1=feature[0][2]
#83 frame
feature1=feature[0][3]
#82 to 83 frame
flow=np.load('/home/ld/RepPoints/temp_for_ransac/1233_later_pair_0.npy')
# print(flow.shape)
# (1, 2, 768, 1280)
info=mmcv.load('/home/ld/RepPoints/temp_for_ransac/temp_5.json')
img1=info[1]
img2=info[2]


