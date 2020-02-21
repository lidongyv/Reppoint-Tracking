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


def is_inlier(mean, offset,threshold=0.3):

    difference=offset-mean
    return torch.norm(difference,dim=-1)<threshold*torch.norm(offset,dim=-1)
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
        # print(mean_offset.shape)
        # while((mean_offset>0).float().mean(-1).sum(-1).sum(-1).mean()<(mean_offset.shape[-3]*mean_offset.shape[-2]*0.3)):
        #     print(offset_select.shape,mask_select.shape)
        #     print('resample',(offset_select>0).float().mean(-1).mean(-1).sum(-1).sum(-1).mean(),(mask_select>0).float().mean(-1).sum(-1).sum(-1).mean())
        #     point1 = torch.randint(offset.shape[-1],size=(offset.shape[0],offset.shape[1],offset.shape[2],1,sample_size)).to(offset.device)
        #     offset_select=offset[n_grid,h_grid,w_grid,c_grid,point1.repeat(1,1,1,2,1)]
        #     mask_select=sample_mask[mn_grid,mh_grid,mw_grid,point1.squeeze(-2)]
        #     mean_offset=torch.sum(offset_select,dim=-1)/torch.sum(mask_select,dim=-1).unsqueeze(-1)
        inner_mask = torch.zeros(offset.shape[0],offset.shape[1],offset.shape[2],offset.shape[4]).to(offset.device)
        # print(offset.shape[-1])
        for m in range(offset.shape[-1]):
            # print(inner_mask[:,:,:,m].shape,is_inlier(mean_offset,offset[:,:,:,:,m]).shape)
            inner_mask[:,:,:,m]=torch.where(is_inlier(mean_offset,offset[:,:,:,:,m],threshold=3),torch.ones(1).to(offset.device),inner_mask[:,:,:,m])

        inner_mask=sample_mask*inner_mask
        inner_count=torch.sum(inner_mask,dim=-1)
        # print(inner_count.shape,best_ic.shape,)
        # torch.Size([2, 92, 160, 1]) torch.Size([2, 92, 160])
        print(torch.sum(best_ic))
        print('update',torch.sum((inner_count > best_ic).float()))
        
        best_model=torch.where(inner_count.unsqueeze(-1).repeat(1,1,1,2) > best_ic.unsqueeze(-1).repeat(1,1,1,2),mean_offset,best_model)
        best_mask=torch.where(inner_count.unsqueeze(-1).repeat(1,1,1,offset.shape[4]) > best_ic.unsqueeze(-1).repeat(1,1,1,offset.shape[4]),inner_mask,best_mask)
        best_ic=torch.where(inner_count > best_ic,inner_count,best_ic)
        print('best count',i,torch.mean(best_ic),torch.min(best_ic),torch.max(best_ic))
        print('mean model',i,torch.mean(best_model[:,:,:,0].abs()),torch.mean(best_model[:,:,:,1].abs()))
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
    # print(torch.max(offset1),torch.max(offset2),torch.mean(offset1))
    feature=mmcv.load('/home/ld/RepPoints/temp_for_ransac_score0.3/cls_feat_5_all.pkl')
    #82 frame
    feature1=torch.from_numpy(feature[0][1]).to(x.device)
    #83 frame
    feature2=torch.from_numpy(feature[0][2]).to(x.device)
    n,c,h,w=generate_grid(offset1[:,:1,:,:])
    # w=((w/w.shape[-1])-0.5)*2
    # h=((h/h.shape[-2])-0.5)*2
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
    sample1=torch.randint(low=-2,high=3,size=(offset1.shape[0],sample_num*offset1.shape[1],offset1.shape[2],offset1.shape[3])).to(x.device).float()
    # print(torch.mean(sample1),torch.max(sample1))
    # exit()
    # feature1=torch.randn(2,256,offset1.shape[-2],offset1.shape[-1]).to(x.device)
    #N
    # offset2=reppoints[1:]
    for i in range(9):
        offset2[:,2*i,:,:]=offset2[:,2*i,:,:]+offset2[:,2*i+1,:,:]
        offset2[:,2*i+1,:,:]=offset2[:,2*i,:,:]-offset2[:,2*i+1,:,:]
        offset2[:,2*i+1,:,:]=offset2[:,2*i,:,:]-offset2[:,2*i+1,:,:]
    offset1_ori_out=offset1+0
    offset2_ori_out=offset2+0
    print(torch.max(offset1),torch.max(grid))
    sample2=torch.randint(low=-2,high=3,size=(offset1.shape[0],sample_num*offset1.shape[1],offset1.shape[2],offset1.shape[3])).to(x.device).float()
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
        grid_t1=sample1_feature_grid[:,:,:,2*i:2*i+2]
        grid_t1[...,0]=((grid_t1[...,0]/w.shape[-1])-0.5)*2
        grid_t1[...,1]=((grid_t1[...,1]/w.shape[-2])-0.5)*2
        sample1_feature.append(torch.nn.functional.grid_sample(feature1,grid_t1,mode='nearest'))
        grid_t2=sample2_feature_grid[:,:,:,2*i:2*i+2]
        grid_t2[...,0]=(((grid_t2[...,0]+inv_flow[...,0])/w.shape[-1])-0.5)*2
        grid_t2[...,1]=(((grid_t2[...,1]+inv_flow[...,1])/w.shape[-2])-0.5)*2
        sample2_feature.append(torch.nn.functional.grid_sample(feature2,grid_t2,mode='nearest'))
        # print(sample1_feature[-1].shape)
    sample1_feature=torch.stack(sample1_feature,dim=4).unsqueeze(-1).repeat(1,1,1,1,1,9*(sample_num+1))
    sample2_feature=torch.stack(sample2_feature,dim=4).repeat_interleave(9*(sample_num+1),dim=-1).view_as(sample1_feature)
    # print((sample1_feature.sum(1)==sample2_feature.sum(1)).float().sum(-1).sum(-1).mean())

    # print(sample1_feature.shape,sample2_feature.shape)
    #similarity computation
    #distance mask
    # sample_distance=torch.norm(,dim=1)
    sample_similarity=torch.nn.functional.cosine_similarity(sample2_feature,sample1_feature,dim=1)
    sample_mask=torch.where(torch.max(sample_similarity,dim=-1)[0]>=0.9,one,zero)
    sample_correspondense=torch.argmax(sample_similarity,dim=-1)
    # sample_similarity=torch.norm(sample1_feature-sample2_feature,dim=1)
    # sample_mask=torch.where(torch.min(sample_similarity,dim=-1)[0]<0.1,one,zero)
    # sample_correspondense=torch.argmin(sample_similarity,dim=-1)
    # print(torch.min(torch.sum(sample_mask,dim=-1)))
    # exit()
    
    # print(sample_correspondense.shape,sample1_feature_grid.shape)
    n,h,w=generate_grid(sample1_feature_grid[:,:,:,0])
    # torch.Size([2, 92, 160, 18]) torch.Size([2, 92, 160, 36])
    offsets=[]
    for i in range(sample_correspondense.shape[-1]):
        # print(sample_correspondense.shape)
        offset_w=sample2_feature_grid[n,h,w,sample_correspondense[:,:,:,i]*2]+inv_flow[:,:,:,0]-sample1_feature_grid[:,:,:,2*i]#check the wh or xy of flow
        offset_h=sample2_feature_grid[n,h,w,sample_correspondense[:,:,:,i]*2+1]+inv_flow[:,:,:,1]-sample1_feature_grid[:,:,:,2*i+1]
        offsets.append(torch.stack([offset_w,offset_h],dim=-1))
    #N to N-1
    print('mean correspondense',torch.sum(sample_mask,dim=-1).mean())
    offsets=torch.stack(offsets,dim=-1)
    print('mean offset',torch.mean(offsets[:,:,:,0,:].abs()),torch.mean(offsets[:,:,:,1,:].abs()))
    offsets=offsets*sample_mask.unsqueeze(-2)
    print('mean offset with mask',(offsets[:,:,:,0,:].abs()).float().sum(-2).sum(-1).mean(),(offsets[:,:,:,1,:].abs()).float().sum(-2).sum(-1).mean())
    print('mean flow',torch.max(inv_flow),torch.min(inv_flow))
    print(offsets.shape)

    n = 9
    max_iterations = 100
    goal_inliers = 3
    # RANSAC to check the cosistentcy
    mean_flow,valid_num,valid_points = run_ransac(offsets,sample_mask, is_inlier, sample_num+1, goal_inliers, max_iterations)
    print(torch.mean(valid_num))
    valid_points=torch.repeat_interleave(valid_points,2,dim=-1).permute(0,3,1,2)
    # print(valid_points.shape,offset1.shape)
    valid_reppoints=offset1*valid_points
    # print(sample_mask.shape)
    sample_mask=torch.repeat_interleave(sample_mask,2,dim=-1).permute(0,3,1,2)
    valid_correspondes=offset1*sample_mask
    mmcv.dump(offset1_ori_out.data.cpu().numpy(),'./offset1_ori.pkl')
    mmcv.dump(offset2_ori_out.data.cpu().numpy(),'./offset2_ori.pkl')
    mmcv.dump(offset1_sample_out.data.cpu().numpy(),'./offset1_sample.pkl')
    mmcv.dump(offset2_sample_out.data.cpu().numpy(),'./offset2_sample.pkl')
    mmcv.dump(valid_reppoints.data.cpu().numpy(),'./ransac2.pkl')
    mmcv.dump(valid_correspondes.data.cpu().numpy(),'./correspondense2.pkl')
