import matplotlib.pyplot as plt
import numpy as np
import cv2
import mmcv
import matplotlib
import os
import shutil
import glob
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
offsets=mmcv.load('/home/ld/RepPoints/ld_result/stsn_class_all/epoch_9_thres0.1_nms0.5_with2/agg/offset.pkl')
for i in range(len(offsets)):
	for j in range(len(offsets[i])):
		for m in range(len(offsets[i][j])):
			print(offsets[i][j][m].shape,i,j,m)
			if torch.is_tensor(offsets[i][j][m]):
				offsets[i][j][m]=offsets[i][j][m].cpu().numpy()
mmcv.dump(offsets, '/home/ld/RepPoints/ld_result/stsn_class_all/epoch_9_thres0.1_nms0.5_with2/agg/offset_numpy.pkl')