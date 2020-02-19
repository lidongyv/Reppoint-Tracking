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


def estimate(point1,point2):

	flow=point2-point1
	return flow

def is_inlier(point1, point2, flow,threshold=1):

	difference=(point2-point1)-flow
	return torch.norm(difference,dim=1)<threshold
def generate_grid(x):
	n=x.shape[0]
	c=x.shape[1]
	h=x.shape[2]
	w=x.shape[3]
	n_grid=torch.arange(n).view(n,1,1,1).expand_as(x)
	c_grid=torch.arange(c).view(1,c,1,1).expand_as(x)
	h_grid=torch.arange(h).view(1,1,h,1).expand_as(x)
	w_grid=torch.arange(w).view(1,1,1,w).expand_as(x)
	return n_grid,c_grid,h_grid,w_grid
def run_ransac(data1,data2, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
	best_ic = torch.zeros(data1.shape[0],1,data1.shape[-2],data1.shape[-1])
	best_model = torch.zeros(data1.shape[0],2,data1.shape[-2],data1.shape[-1])
	best_mask=torch.zeros(data1.shape[0],9,data1.shape[-2],data1.shape[-1])
	np.random.seed(random_seed)
	n_grid,c_grid,h_grid,w_grid=generate_grid(data1[:,:1,:,:])
	for i in range(max_iterations):
		point1 = torch.randint(8,size=(data1.shape[0],1,data1.shape[-2],data1.shape[-1]))
		point2 = torch.randint(8, size=(data1.shape[0],1,data1.shape[-2],data1.shape[-1]))
		point1=torch.cat([data1[n_grid,2*point1,h_grid,w_grid],data1[n_grid,2*point1+1,h_grid,w_grid]],dim=1)
		point2=torch.cat([data2[n_grid,2*point2,h_grid,w_grid],data2[n_grid,2*point2+1,h_grid,w_grid]],dim=1)
		flow = estimate(point2,point1)

		inner_count = torch.zeros(data1.shape[0],9,data1.shape[-2],data1.shape[-1])
		inner_mask = torch.zeros(data1.shape[0],9,data1.shape[-2],data1.shape[-1])
		for m in range(9):
			inner_check=torch.zeros(data1.shape[0],data1.shape[-2],data1.shape[-1])
			for n in range(9):
				inner_check=torch.where(is_inlier(data1[:,2*n:2*n+2,...],data2[:,2*m:2*m+2,...],flow),torch.ones(1),inner_check)
			inner_count[:,m,:,:]=inner_check
		inner_mask=inner_count+0
		inner_count=torch.sum(inner_count,dim=1,keepdim=True)

		best_ic=torch.where(inner_count > best_ic,inner_count,best_ic)
		best_model=torch.where(inner_count > best_ic,flow,best_model)
		best_mask=torch.where(inner_count > best_ic,inner_mask,best_mask)
		print(i,torch.mean(best_ic),torch.min(best_ic),torch.max(best_ic))
		if torch.all(best_ic > goal_inliers) and stop_at_goal:
			break
	print('iterations:', i+1)
	return best_model, best_ic,best_mask
if __name__=='__main__':
	reppoints=mmcv.load('repoints_sample.pkl')
	#N-1
	data1=reppoints[:2]
	#N
	data2=reppoints[1:]
	#N to N-1
	inv_flow=reppoints[1:,:2,:,:]
	n,c,h,w=generate_grid(inv_flow[:,:1,:,:])
	inv_flow=inv_flow.permute(0,2,3,1)
	w=((w/w.shape[-1])-0.5)*2
	h=((h/h.shape[-2])-0.5)*2
	grid=torch.cat([w,h],dim=1).permute(0,2,3,1)
	#warp N-1 to N
	data1=torch.nn.functional.grid_sample(data1,grid+inv_flow,mode='nearest')
	n = 9
	max_iterations = 10
	goal_inliers = 3
	# RANSAC to check the cosistentcy
	valid_num, mean_flow,valid_points = run_ransac(data1,data2, estimate, is_inlier, 1, goal_inliers, max_iterations)
	valid_points=torch.repeat_interleave(valid_points,2,dim=1)
	valid_reppoints=data2*valid_points