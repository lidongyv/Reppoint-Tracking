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


def is_inlier(mean, offset,threshold=1):

    difference=offset-mean
    return torch.norm(difference,dim=-1)<threshold
def generate_grid(x):
    # print(x.shape)
    if len(x.shape)==4:
        n=x.shape[0]
        c=x.shape[1]
        h=x.shape[2]
        w=x.shape[3]
        n_grid=torch.arange(n).view(n,1,1,1).expand_as(x).to(x.device)
        c_grid=torch.arange(c).view(1,c,1,1).expand_as(x).to(x.device)
        h_grid=torch.arange(h).view(1,1,h,1).expand_as(x).to(x.device)
        w_grid=torch.arange(w).view(1,1,1,w).expand_as(x).to(x.device)
        return n_grid,c_grid,h_grid,w_grid
    if len(x.shape)==3:
        n=x.shape[0]
        h=x.shape[1]
        w=x.shape[2]
        n_grid=torch.arange(n).view(n,1,1).expand_as(x).to(x.device)
        h_grid=torch.arange(h).view(1,h,1).expand_as(x).to(x.device)
        w_grid=torch.arange(w).view(1,1,w).expand_as(x).to(x.device)
        return n_grid,h_grid,w_grid
    if len(x.shape)==5:
        n=x.shape[0]
        h=x.shape[1]
        w=x.shape[2]
        c=x.shape[3]
        s=x.shape[4]
        n_grid=torch.arange(n).view(n,1,1,1,1).expand_as(x).to(x.device)
        h_grid=torch.arange(h).view(1,h,1,1,1).expand_as(x).to(x.device)
        w_grid=torch.arange(w).view(1,1,w,1,1).expand_as(x).to(x.device)
        c_grid=torch.arange(c).view(1,1,1,c,1).expand_as(x).to(x.device)
        s_grid=torch.arange(s).view(1,1,1,1,s).expand_as(x).to(x.device)
        return n_grid,h_grid,w_grid,c_grid,s_grid
def run_ransac(offset, sample_mask,is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = torch.zeros(offset.shape[0],offset.shape[1],offset.shape[2]).to(offset.device)
    best_model = torch.zeros(offset.shape[0],offset.shape[1],offset.shape[2],2).to(offset.device)
    best_mask=torch.zeros(offset.shape[0],offset.shape[1],offset.shape[2],offset.shape[4]).to(offset.device)
    np.random.seed(random_seed)
    # print(offset.shape)
    n_grid,h_grid,w_grid,c_grid,s_grid=generate_grid(offset[:,:,:,:,:sample_size])
    mn_grid,mh_grid,mw_grid,ms_grid=generate_grid(sample_mask[:,:,:,:sample_size])
    for i in range(max_iterations):
        point1 = torch.randint(offset.shape[-1],size=(offset.shape[0],offset.shape[1],offset.shape[2],1,sample_size)).to(offset.device)
        offset_select=offset[n_grid,h_grid,w_grid,c_grid,point1.repeat(1,1,1,2,1)]
        mask_select=sample_mask[mn_grid,mh_grid,mw_grid,point1.squeeze(-2)]
        mean_offset=torch.sum(offset_select,dim=-1)/torch.sum(mask_select,dim=-1).unsqueeze(-1)
        inner_mask = torch.zeros(offset.shape[0],offset.shape[1],offset.shape[2],offset.shape[4]).to(offset.device)
        # print(offset.shape[-1])
        for m in range(offset.shape[-1]):
            # print(inner_mask[:,:,:,m].shape,is_inlier(mean_offset,offset[:,:,:,:,m]).shape)
            inner_mask[:,:,:,m]=torch.where(is_inlier(mean_offset,offset[:,:,:,:,m]),torch.ones(1).to(offset.device),inner_mask[:,:,:,m])

        inner_mask=sample_mask*inner_mask
        inner_count=torch.sum(inner_mask,dim=-1)
        # print(inner_count.shape,best_ic.shape,)
        # torch.Size([2, 92, 160, 1]) torch.Size([2, 92, 160])
        best_ic=torch.where(inner_count > best_ic,inner_count,best_ic)
        
        best_model=torch.where(inner_count.unsqueeze(-1).repeat(1,1,1,2) > best_ic.unsqueeze(-1).repeat(1,1,1,2),mean_offset,best_model)
        best_mask=torch.where(inner_count.unsqueeze(-1).repeat(1,1,1,offset.shape[4]) > best_ic.unsqueeze(-1).repeat(1,1,1,offset.shape[4]),inner_mask,best_mask)
        print(i,torch.mean(best_ic),torch.min(best_ic),torch.max(best_ic))
        # if torch.all(best_ic > goal_inliers) and stop_at_goal:
        #     break
    print('iterations:', i+1)
    return best_model, best_ic,best_mask
if __name__=='__main__':
    sample_num=1
    x=torch.ones(1).cuda(0)
    one=torch.ones(1).to(x.device)
    zero=torch.zeros(1).to(x.device)
    reppoints=mmcv.load('/home/ld/RepPoints/temp_for_ransac_score0.3/pts_out_init_grad_mul_5_all.pkl')
    offset1=reppoints[0][1].to(x.device)
    offset2=reppoints[0][2].to(x.device)
    offset1_ori_out=offset1+0
    offset2_ori_out=offset2+0
    feature=mmcv.load('/home/ld/RepPoints/temp_for_ransac_score0.3/cls_feat_5_all.pkl')
    #82 frame
    feature1=torch.from_numpy(feature[0][2]).to(x.device)
    #83 frame
    feature2=torch.from_numpy(feature[0][3]).to(x.device)
    n,c,h,w=generate_grid(offset1[:,:1,:,:])
    w=((w/w.shape[-1])-0.5)*2
    h=((h/h.shape[-2])-0.5)*2
    grid=torch.cat([w,h],dim=1).permute(0,2,3,1).to(x.device)
    # print(grid.shape,offset1.shape)
    # inv_flow=torch.rand_like(reppoints[1:,:2,:,:])
    inv_flow=torch.from_numpy(np.load('/home/ld/RepPoints/temp_for_ransac_score0.3/1233_later_pair_0.npy'))
    scale=inv_flow.shape[-1]/offset1.shape[-1]
    inv_flow=torch.nn.functional.interpolate(inv_flow,size=(offset1.shape[-2],offset1.shape[-1]))/scale
    inv_flow=inv_flow.permute(0,2,3,1).to(x.device)
    # print(inv_flow.shape)
    #N-1,xy
    # offset1=reppoints[:2]
    #xy to wh
    for i in range(9):
        offset1[:,2*i,:,:]=offset1[:,2*i,:,:]+offset1[:,2*i+1,:,:]
        offset1[:,2*i+1,:,:]=offset1[:,2*i,:,:]-offset1[:,2*i+1,:,:]
        offset1[:,2*i+1,:,:]=offset1[:,2*i,:,:]-offset1[:,2*i+1,:,:]
    sample1=torch.randn(offset1.shape[0],sample_num*offset1.shape[1],offset1.shape[2],offset1.shape[3]).to(x.device)
    # feature1=torch.randn(2,256,offset1.shape[-2],offset1.shape[-1]).to(x.device)
    #N
    # offset2=reppoints[1:]
    for i in range(9):
        offset2[:,2*i,:,:]=offset2[:,2*i,:,:]+offset2[:,2*i+1,:,:]
        offset2[:,2*i+1,:,:]=offset2[:,2*i,:,:]-offset2[:,2*i+1,:,:]
        offset2[:,2*i+1,:,:]=offset2[:,2*i,:,:]-offset2[:,2*i+1,:,:]
    sample2=torch.randn(offset2.shape[0],sample_num*offset2.shape[1],offset2.shape[2],offset2.shape[3]).to(x.device)
    # feature2=torch.randn(2,256,offset2.shape[-2],offset2.shape[-1]).to(x.device)
    for i in range(sample_num):
        sample1[:,18*i:18*(i+1),:,:]=sample1[:,18*i:18*(i+1),:,:]+offset1
        sample2[:,18*i:18*(i+1),:,:]=sample2[:,18*i:18*(i+1),:,:]+offset2
    offset1=torch.cat([offset1,sample1],dim=1)
    offset2=torch.cat([offset2,sample2],dim=1)
    offset1_sample_out=offset1+0
    offset2_sample_out=offset2+0
    # print(grid.shape)
    # print(offset1.permute(0,2,3,1).shape,grid.repeat(1,1,1,9*(sample_num+1)).shape)
    sample1_feature_grid=offset1.permute(0,2,3,1)+grid.repeat(1,1,1,9*(sample_num+1))
    sample2_feature_grid=offset2.permute(0,2,3,1)+grid.repeat(1,1,1,9*(sample_num+1))

    sample1_feature=[]
    sample2_feature=[]
    for i in range(9*(sample_num+1)):
        sample1_feature.append(torch.nn.functional.grid_sample(feature1,sample1_feature_grid[:,:,:,2*i:2*i+2]+inv_flow,mode='nearest'))
        sample2_feature.append(torch.nn.functional.grid_sample(feature2,sample2_feature_grid[:,:,:,2*i:2*i+2],mode='nearest'))
        # print(sample1_feature[-1].shape)
    sample1_feature=torch.stack(sample1_feature,dim=4).unsqueeze(-1).repeat_interleave(9*(sample_num+1),dim=-1)
    sample2_feature=torch.stack(sample2_feature,dim=4).unsqueeze(-1).repeat(1,1,1,1,1,9*(sample_num+1))
    # print(sample1_feature.shape,sample2_feature.shape)
    #similarity computation
    # sample_similarity=torch.nn.functional.cosine_similarity(sample2_feature,sample1_feature,dim=1)
    # sample_mask=torch.where(torch.max(sample_similarity,dim=-1)[0]>=0.01,one,zero)
    # sample_correspondense=torch.argmax(sample_similarity,dim=-1)
    sample_similarity=torch.norm(sample2_feature-sample1_feature,dim=1)
    # print(sample_similarity.shape)
    sample_mask=torch.where(torch.min(sample_similarity,dim=-1)[0]<=0.1,one,zero)
    # print(torch.sum(sample_mask,dim=-1))
    # exit()
    sample_correspondense=torch.argmin(sample_similarity,dim=-1)
    # print(sample_correspondense.shape,sample1_feature_grid.shape)
    n,h,w=generate_grid(sample1_feature_grid[:,:,:,0])
    # torch.Size([2, 92, 160, 18]) torch.Size([2, 92, 160, 36])
    offsets=[]
    for i in range(sample_correspondense.shape[-1]):
        # print(sample_correspondense.shape)
        offset_w=sample2_feature_grid[:,:,:,2*i]+inv_flow[:,:,:,0]-sample1_feature_grid[n,h,w,sample_correspondense[:,:,:,i]*2]#check the wh or xy of flow
        offset_h=sample2_feature_grid[:,:,:,2*i+1]+inv_flow[:,:,:,1]-sample1_feature_grid[n,h,w,sample_correspondense[:,:,:,i]*2+1]
        offsets.append(torch.stack([offset_w,offset_h],dim=-1))
    #N to N-1
    print('mean offset',torch.sum(sample_mask,dim=-1).mean())
    offsets=torch.stack(offsets,dim=-1)*sample_mask.unsqueeze(-2)
    print(offsets.shape)

    n = 9
    max_iterations = 100
    goal_inliers = 3
    # RANSAC to check the cosistentcy
    valid_num, mean_flow,valid_points = run_ransac(offsets,sample_mask, is_inlier, sample_num+1, goal_inliers, max_iterations)
    valid_points=torch.repeat_interleave(valid_points.unsqueeze(-2),2,dim=-2)
    valid_reppoints=offsets*valid_points
    mmcv.dump(offset1_ori_out.data.cpu().numpy(),'./offset1_ori.pkl')
    mmcv.dump(offset2_ori_out.data.cpu().numpy(),'./offset2_ori.pkl')
    mmcv.dump(offset1_sample_out.data.cpu().numpy(),'./offset1_sample.pkl')
    mmcv.dump(offset2_sample_out.data.cpu().numpy(),'./offset2_sample.pkl')
    mmcv.dump(valid_reppoints.data.cpu().numpy(),'./ransac2.pkl')
