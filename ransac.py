import torch
import numpy as np
import os
import mmcv
import cv2
# reppoints=mmcv.load('/home/ld/RepPoints/ld_result/reppoint_do3/epoch_23_thres0.3_nms0.3/refer/reppoints.pkl')
# sample=[torch.from_numpy(reppoints[0][i]) for i in range(3)]
# sample=torch.cat(sample,dim=0)
# print(sample.shape)
# mmcv.dump(sample,'repoints_sample.pkl')
import random
def augment(xyzs):
	axyz = np.ones((len(xyzs), 4))
	axyz[:, :3] = xyzs
	return axyz

def estimate(point1,point2):
	flow=point2-point1
	return flow

def is_inlier(point1, point2, flow,threshold=1):
	return torch.norm((point2-point1)-flow,dim=0)<threshold

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
	best_ic = torch.zeros(data.shape[-2],data.shape[-1])
	best_model = torch.zeros(2,data.shape[-2],data.shape[-1])
	np.random.seed(random_seed)
	x=torch.arange(data.shape[-2])
	y=torch.arange(data.shape[-1])
	x,y=torch.meshgrid(x,y)
	for i in range(max_iterations):
		point1 = torch.randint(8,size=(data.shape[-2],data.shape[-1]))
		point2 = torch.randint(8, size=(data.shape[-2],data.shape[-1]))
		point1=torch.stack([data[0,2*point1,x,y],data[0,2*point1+2,x,y]])
		point2=torch.stack([data[1,2*point2,x,y],data[1,2*point2+2,x,y]])
		flow = estimate(point2,point1)
		data1=data[0]
		data2=data[2]
		inner_count = torch.zeros(9,data.shape[-2],data.shape[-1])
		for m in range(9):
			inner_check=torch.zeros(data.shape[-2],data.shape[-1])
			for n in range(9):
				inner_check=torch.where(is_inlier(data1[2*n:2*n+2,...],data2[2*m:2*m+2,...],flow),torch.ones(1),inner_check)
			inner_count[m]=inner_check
		inner_count=torch.sum(inner_count,dim=0)

		best_ic=torch.where(inner_count > best_ic,inner_count,best_ic)
		best_model=torch.where(inner_count > best_ic,flow,best_model)
		print(i,torch.mean(best_ic),torch.min(best_ic),torch.max(best_ic))
		if torch.all(best_ic > goal_inliers) and stop_at_goal:
			break
	print('iterations:', i+1, 'best model:', best_model, 'final reppoints:', best_ic)
	return best_model, best_ic
if __name__=='__main__':
	reppoints=mmcv.load('repoints_sample.pkl')
	# print(reppoints.shape)
	n = 9
	max_iterations = 1000
	goal_inliers = 3

	# RANSAC
	m, b = run_ransac(reppoints, estimate, is_inlier, 1, goal_inliers, max_iterations)
