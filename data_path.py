import mmcv
import os
import time
import ffmpeg
import json
import numpy as np
import copy


data_path='/backdata01/KITTI/kitti/tracking'
test_data_all=mmcv.load(os.path.join(data_path,'occlusion_test.json'))
train_data_all=mmcv.load(os.path.join(data_path,'occlusion_train.json'))
test_data_v=mmcv.load(os.path.join(data_path,'occlusion_test_v.json'))
test_data_p=mmcv.load(os.path.join(data_path,'occlusion_test_p.json'))
test_data_vp=mmcv.load(os.path.join(data_path,'occlusion_test_vp.json'))
print(test_data_all[0])
	
