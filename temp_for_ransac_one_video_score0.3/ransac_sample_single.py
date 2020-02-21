import torch
import numpy as np
import os
import mmcv
import cv2
import json
# reppoints=mmcv.load('/home/ld/RepPoints/ld_result/reppoint_do3/epoch_23_thres0.3_nms0.3/refer/reppoints.pkl')
# sample=[torch.from_numpy(reppoints[0][i]) for i in range(3)]
# sample=torch.cat(sample,dim=0)
# print(sample.shape)
# mmcv.dump(sample,'repoints_sample.pkl')
import random


def is_inlier(mean, offset,threshold=1):

    difference=offset-mean
    return torch.norm(difference,dim=-1)<threshold*torch.norm(mean,dim=-1)
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
def run_ransac(offset, sample_mask,is_inlier, sample_size, goal_inliers, max_iterations, x_loc1,y_loc1,stop_at_goal=True, random_seed=None):
    best_ic = torch.zeros(offset.shape[0],offset.shape[1],offset.shape[2]).to(offset.device)
    best_model = torch.zeros(offset.shape[0],offset.shape[1],offset.shape[2],2).to(offset.device)
    best_mask=torch.zeros(offset.shape[0],offset.shape[1],offset.shape[2],offset.shape[4]).to(offset.device)
    np.random.seed(random_seed)
    # print(offset.shape)
    n_grid,h_grid,w_grid,c_grid,s_grid=generate_grid(offset[:,:,:,:,:sample_size])
    mn_grid,mh_grid,mw_grid,ms_grid=generate_grid(sample_mask[:,:,:,:sample_size])
    for i in range(max_iterations):
        point1 = torch.randint(offset.shape[-1],size=(offset.shape[0],offset.shape[1],offset.shape[2],1,sample_size)).to(offset.device)
        print('point select',point1[0,x_loc1,y_loc1])
        offset_select=offset[n_grid,h_grid,w_grid,c_grid,point1.repeat(1,1,1,2,1)]
        print('offset select',offset_select[0,x_loc1,y_loc1])
        mask_select=sample_mask[mn_grid,mh_grid,mw_grid,point1.squeeze(-2)]
        print('mask select',mask_select[0,x_loc1,y_loc1])
        mean_offset=torch.sum(offset_select,dim=-1)/torch.sum(mask_select,dim=-1).unsqueeze(-1)
        print('mean offset',mean_offset[0,x_loc1,y_loc1])
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
            inner_mask[:,:,:,m]=torch.where(is_inlier(mean_offset,offset[:,:,:,:,m],threshold=0.2),torch.ones(1).to(offset.device),inner_mask[:,:,:,m])
        print('inner mask',inner_mask[0,x_loc1,y_loc1])
        inner_mask=sample_mask*inner_mask
        print('variance check',torch.sum((sample_mask!=inner_mask).float().max(-1)[0]))
        print('sample mask',sample_mask[0,x_loc1,y_loc1])
        inner_count=torch.sum(inner_mask,dim=-1)
        # print(inner_count.shape,best_ic.shape,)
        # torch.Size([2, 92, 160, 1]) torch.Size([2, 92, 160])
        # print(torch.sum(best_ic))
        # print('update',torch.sum((inner_count > best_ic).float()))
        
        best_model=torch.where(inner_count.unsqueeze(-1).repeat(1,1,1,2) > best_ic.unsqueeze(-1).repeat(1,1,1,2),mean_offset,best_model)
        best_mask=torch.where(inner_count.unsqueeze(-1).repeat(1,1,1,offset.shape[4]) > best_ic.unsqueeze(-1).repeat(1,1,1,offset.shape[4]),inner_mask,best_mask)
        best_ic=torch.where(inner_count > best_ic,inner_count,best_ic)
        # print('best count',i,torch.mean(best_ic),torch.min(best_ic),torch.max(best_ic))
        # print('mean model',i,torch.mean(best_model[:,:,:,0].abs()),torch.mean(best_model[:,:,:,1].abs()))
        # if torch.all(best_ic > goal_inliers) and stop_at_goal:
        #     break
    print('iterations:', i+1)
    return best_model, best_ic,best_mask
if __name__=='__main__':
    sample_num=1
    x=torch.ones(1).cuda(0)
    one=torch.ones(1).to(x.device)
    zero=torch.zeros(1).to(x.device)
    root_path = '/home/ld/RepPoints/temp_for_ransac_one_video_score0.3'
    reppoints=mmcv.load(os.path.join(root_path, 'pkl', 'pts_out_init_grad_mul_video_all.pkl'))
    feature=mmcv.load(os.path.join(root_path, 'pkl', 'cls_feat_video_all.pkl'))
    loc_record=mmcv.load(os.path.join(root_path, 'pkl', 'loc_result_video_all.pkl'))
    
    
    with open(os.path.join(root_path ,'json' , 'waymo_one_video.json'), 'r') as f:
        all_data = json.load(f)
    all_imgname_list = []
    for each in all_data:
        if each['filename'].split('/')[-1].split('.')[0] not in all_imgname_list:
            all_imgname_list.append(each['filename'].split('/')[-1].split('.')[0])
    # list all flow files
    flow_dir_names = os.listdir(os.path.join(root_path ,'flow'))
    flow_dir_names.sort()

    for each_flow_name in flow_dir_names:
        print(each_flow_name)
        current_index = all_imgname_list.index(each_flow_name)
        offset1 = reppoints[0][current_index].to(x.device)  # 前一帧的 reppoints  第二个维度的index 要改
        offset2 = reppoints[0][current_index + 1].to(x.device)  # 后一帧的 reppoints  第二个维度的index 要改
        loc1 = torch.from_numpy(np.vstack(loc_record[current_index]))  # 前一帧的 reppoints

        #这里需要指定一个bbox
        bbox_index = 0
        x_loc1 = (loc1[bbox_index][1] // 8).long()
        y_loc1 = (loc1[bbox_index][0] // 8).long()
        feature1 = torch.from_numpy(feature[0][current_index]).to(x.device)
        feature2 = torch.from_numpy(feature[0][current_index + 1]).to(x.device)
        n, c, h, w = generate_grid(offset1[:, :1, :, :])
        grid = torch.cat([w, h], dim=1).permute(0, 2, 3, 1).to(x.device)
        inv_flow = torch.from_numpy(np.load(os.path.join(root_path,'flow','%s/'%each_flow_name,'flow_%06dto_%06d.npy'%(current_index, current_index +1))))
        # offset1=reppoints[0][1].to(x.device)  #前一帧的 reppoints  第二个维度的index 要改
        # offset2=reppoints[0][2].to(x.device)  #后一帧的 reppoints  第二个维度的index 要改
        # print(torch.max(offset1),torch.max(offset2),torch.mean(offset1))

        # loc1=torch.from_numpy(np.vstack(loc_record[1]))  #前一帧的 reppoints
        # x_loc1=(loc1[10][1]//8).long()
        # y_loc1=(loc1[10][0]//8).long()
        #82 frame
        # feature1=torch.from_numpy(feature[0][1]).to(x.device)
        #83 frame
        # feature2=torch.from_numpy(feature[0][2]).to(x.device)
        # n,c,h,w=generate_grid(offset1[:,:1,:,:])
        # w=((w/w.shape[-1])-0.5)*2
        # h=((h/h.shape[-2])-0.5)*2
        # grid=torch.cat([w,h],dim=1).permute(0,2,3,1).to(x.device)
        # print(grid.shape,offset1.shape)
        # inv_flow=torch.rand_like(reppoints[1:,:2,:,:])


        scale=inv_flow.shape[-1]/offset1.shape[-1]
        inv_flow=torch.nn.functional.interpolate(inv_flow,size=(offset1.shape[-2],offset1.shape[-1]))/scale
        x_loc2=(x_loc1+inv_flow[0,1,x_loc1,y_loc1]).long()
        y_loc2=(y_loc1+inv_flow[0,0,x_loc1,y_loc1]).long()
        inv_flow=inv_flow.permute(0,2,3,1).to(x.device)
        # print(inv_flow.shape)
        #N-1,xy
        # offset1=reppoints[:2]
        #xy to wh
        for i in range(9):
            offset1[:,2*i,:,:]=offset1[:,2*i,:,:]+offset1[:,2*i+1,:,:]
            offset1[:,2*i+1,:,:]=offset1[:,2*i,:,:]-offset1[:,2*i+1,:,:]
            offset1[:,2*i+1,:,:]=offset1[:,2*i,:,:]-offset1[:,2*i+1,:,:]
        sample1=torch.rand(size=(offset1.shape[0],sample_num*offset1.shape[1],offset1.shape[2],offset1.shape[3])).to(x.device).float()*3
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
        # print(torch.max(offset1),torch.max(grid))
        sample2=torch.rand(size=(offset1.shape[0],sample_num*offset1.shape[1],offset1.shape[2],offset1.shape[3])).to(x.device).float()*3
        # feature2=torch.randn(2,256,offset2.shape[-2],offset2.shape[-1]).to(x.device)
        for i in range(sample_num):
            sample1[:,18*i:18*(i+1),:,:]=sample1[:,18*i:18*(i+1),:,:]+offset1
            sample2[:,18*i:18*(i+1),:,:]=sample2[:,18*i:18*(i+1),:,:]+offset2
        offset1=torch.cat([offset1,sample1],dim=1)
        offset2=torch.cat([offset2,sample2],dim=1)
        offset1_sample_out=offset1+0
        offset2_sample_out=offset2+0
        # for i in range(18):
        #     print(i,offset1_sample_out[0,2*i:2*i+2,x_loc1,y_loc1],offset2_sample_out[0,2*i:2*i+2,x_loc2,y_loc2])
        #check sample loc is right
        # print(grid.shape)
        # print(offset1.permute(0,2,3,1).shape,grid.repeat(1,1,1,9*(sample_num+1)).shape)
        sample1_feature_grid=offset1.permute(0,2,3,1)+grid.repeat(1,1,1,9*(sample_num+1))
        sample2_feature_grid=offset2.permute(0,2,3,1)+grid.repeat(1,1,1,9*(sample_num+1))
        sample1_feature=[]
        sample2_feature=[]
        for i in range(9*(sample_num+1)):
            grid_t1=sample1_feature_grid[:,:,:,2*i:2*i+2]
            # print(i)
            # print(grid_t1[0,x_loc1,y_loc1])
            grid_t1[...,0]=((grid_t1[...,0]/w.shape[-1])-0.5)*2
            grid_t1[...,1]=((grid_t1[...,1]/w.shape[-2])-0.5)*2
            sample1_feature.append(torch.nn.functional.grid_sample(feature1,grid_t1,mode='nearest'))
            grid_t2=sample2_feature_grid[:,:,:,2*i:2*i+2]
            # print(grid_t2[0,x_loc2,y_loc2])
            grid_t2[...,0]=(((grid_t2[...,0])/w.shape[-1])-0.5)*2
            grid_t2[...,1]=(((grid_t2[...,1])/w.shape[-2])-0.5)*2
            sample2_feature_ori=torch.nn.functional.grid_sample(feature2,grid_t2,mode='nearest')
            grid_w2=grid+inv_flow
            # print(grid_w2[0,x_loc1,y_loc1])
            grid_w2[...,0]=(((grid_w2[...,0])/w.shape[-1])-0.5)*2
            grid_w2[...,1]=(((grid_w2[...,1])/w.shape[-2])-0.5)*2
            sample2_feature.append(torch.nn.functional.grid_sample(sample2_feature_ori,grid_w2,mode='nearest'))

            # print(sample1_feature[-1].shape)
        # exit()
        sample1_feature=torch.stack(sample1_feature,dim=-1).repeat(1,1,1,1,9*(sample_num+1))
        sample1_feature=sample1_feature.view(sample1_feature.shape[0],sample1_feature.shape[1],sample1_feature.shape[2],sample1_feature.shape[3], \
            9*(sample_num+1),9*(sample_num+1))
        #[0,:,x_loc1,y_loc1,:].repeat(1,18)
        #.unsqueeze(-1).repeat(1,1,1,1,1,9*(sample_num+1))
        sample2_feature=torch.stack(sample2_feature,dim=-1).repeat_interleave(9*(sample_num+1),dim=-1).view_as(sample1_feature)
        #[0,:,x_loc1,y_loc1,:].repeat_interleave(18,dim=1)
        #.repeat_interleave(9*(sample_num+1),dim=-1).view_as(sample1_feature)
        # print((sample1_feature.sum(1)==sample2_feature.sum(1)).float().sum(-1).sum(-1).mean())

        # print(sample1_feature.shape,sample2_feature.shape)
        #similarity computation
        #distance mask
        # sample_distance=torch.norm(,dim=1)
        sample_similarity=torch.nn.functional.cosine_similarity(sample1_feature,sample2_feature,dim=1)
        #[0,x_loc1,y_loc1,:,:]

        sample_mask=torch.where(torch.max(sample_similarity,dim=-1)[0]>=0.9,one,zero)
        sample_correspondense=torch.argmax(sample_similarity,dim=-1)
        # print(sample_correspondense,sample_mask)
        # exit()
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
        # print('mean correspondense',torch.sum(sample_mask,dim=-1).mean())
        offsets=torch.stack(offsets,dim=-1)
        print(offsets.shape)
        # print(offsets[0,x_loc1,y_loc1,:,:])

        # print('mean offset',torch.mean(offsets[:,:,:,0,:].abs()),torch.mean(offsets[:,:,:,1,:].abs()))
        offsets=offsets*sample_mask.unsqueeze(-2)
        print(offsets[0,x_loc1,y_loc1,:,:])
        # exit()
        # print('mean offset with mask',(offsets[:,:,:,0,:].abs()).float().sum(-2).sum(-1).mean(),(offsets[:,:,:,1,:].abs()).float().sum(-2).sum(-1).mean())
        # print('mean flow',torch.max(inv_flow),torch.min(inv_flow))


        n = 9
        max_iterations = 10
        goal_inliers = 3
        # RANSAC to check the cosistentcy
        mean_flow,valid_num,valid_points = run_ransac(offsets,sample_mask, is_inlier, sample_num+1, goal_inliers, max_iterations,x_loc1,y_loc1)
        print(torch.mean(valid_num))
        valid_points=torch.repeat_interleave(valid_points,2,dim=-1).permute(0,3,1,2)
        # print(valid_points.shape,offset1.shape)
        valid_reppoints=offset1*valid_points
        # print(sample_mask.shape)
        sample_mask=torch.repeat_interleave(sample_mask,2,dim=-1).permute(0,3,1,2)
        valid_correspondes=offset1*sample_mask

        mmcv.dump(offset1_ori_out.data.cpu().numpy(), os.path.join(root_path, 'flow', '%s/'%each_flow_name, 'offset1_ori.pkl'))
        mmcv.dump(offset2_ori_out.data.cpu().numpy(), os.path.join(root_path, 'flow', '%s/'%each_flow_name, 'offset2_ori.pkl'))
        mmcv.dump(offset1_sample_out.data.cpu().numpy(), os.path.join(root_path, 'flow', '%s/'%each_flow_name, 'offset1_sample.pkl'))
        mmcv.dump(offset2_sample_out.data.cpu().numpy(), os.path.join(root_path, 'flow', '%s/'%each_flow_name, 'offset2_sample.pkl'))
        mmcv.dump(valid_reppoints.data.cpu().numpy(), os.path.join(root_path, 'flow', '%s/'%each_flow_name, 'ransac2.pkl'))
        mmcv.dump(valid_correspondes.data.cpu().numpy(), os.path.join(root_path, 'flow', '%s/'%each_flow_name, 'correspondense2.pkl'))
