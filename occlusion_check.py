# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-11-29 01:47:58  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-11-29 01:47:58
import numpy as np
import os
import json
def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious
data_path='/backdata01/KITTI/kitti/tracking'
with open(os.path.join(data_path,'kitti_train.json'),'r',encoding='utf-8') as f:
	data=json.load(f)
occlusion=[]
partial_occlusion=[]
for i in data:
	boxt=np.array(i['ann']['bboxes'])
	if len(boxt)==0:
		continue
	iou=bbox_overlaps(boxt,boxt)
	check=np.sum(iou,axis=0)
	# print(check)
	# print(np.max(check))
	if np.max(check)>1:
		occlusion.append(i)
	if np.max(check)>1.5:
		partial_occlusion.append(i)
print(len(occlusion)/len(data),len(partial_occlusion)/len(data), \
	len(partial_occlusion)/len(occlusion))
#0.563941686658055 0.18269053330872856 0.3239528795811518
with open(os.path.join(data_path,'occlusion.json'), 'w', encoding='utf-8') as f:
	json.dump({'info':{'occlusion':len(occlusion)/len(data), \
		'partial_occlusion':len(partial_occlusion)/len(data), \
		'hard_occlusion':len(partial_occlusion)/len(occlusion),}, \
		'occlusion':occlusion,'partial_occlusion':partial_occlusion}, f, ensure_ascii=False)