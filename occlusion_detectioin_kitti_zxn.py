import mmcv
import os
import time
import ffmpeg
import json
import numpy as np
import copy
def bbox_overlaps(bboxes1, bboxes2):
	bboxes1=np.array(bboxes1) # 转成array
	bboxes2=np.array(bboxes2)
	bboxes1 = bboxes1.astype(np.float32) #转换类型
	bboxes2 = bboxes2.astype(np.float32)
	if bboxes1.shape[1]>4:
		bboxes1=bboxes1[:,:4]  #如果有score的话就去除score 只保留Bbox的部分
		bboxes2=bboxes2[:,:4]  #如果有score的话就去除score 只保留Bbox的部分
	rows = bboxes1.shape[0]  #bbox1的shape
	cols = bboxes2.shape[0]  #bbox2的shape
	ious = np.zeros((rows, cols), dtype=np.float32) # 这个影响很大吗
	if rows * cols == 0:
		# 如果两个BBox都没有的话就返回一个为0的array
		return ious
	# exchange 这个变量的作用是什么？
	exchange = False
	# 如果bboxes1的shape大于bboxes2的shape，交换两个Bbox的，将第二个Bbox作为第一个Bbox，将第一个Bbox作为第二个Bbox
	if bboxes1.shape[0] > bboxes2.shape[0]:
		bboxes1, bboxes2 = bboxes2, bboxes1
		ious = np.zeros((cols, rows), dtype=np.float32)
		exchange = True
	# 这里交换不交换有什么区别吗？？
	# 计算BBox1的面积
	area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
	# 计算BBox2的面积
	area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
	# 为什么要从 bboxes1.shape[0] 进行循环？？
	for i in range(bboxes1.shape[0]):
		x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0]) #X最小值的最大值
		y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1]) #Y最小值的最大值
		x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2]) #X最大值的最小值
		y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3]) #Y最大值的最小值
		# 计算两个相交的BBox的面积 U
		overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(y_end - y_start + 1, 0)
		# 计算 I
		union = area1[i] + area2 - overlap
		# 计算IOU
		ious[i, :] = overlap / union  # 一行一行地记录IOU的数值
	# 如果前面的 exchange == 1 的话，这里要记得交换回来
	if exchange:
		ious = ious.T
	return ious

def bbox_occluded(bboxes1, bboxes2,same=False):
	bboxes1=np.array(bboxes1)
	bboxes2=np.array(bboxes2)
	bboxes1 = bboxes1.astype(np.float32)
	bboxes2 = bboxes2.astype(np.float32)
	if bboxes1.shape[1]>4:
		bboxes1=bboxes1[:,:4]
		bboxes2=bboxes2[:,:4]
	rows = bboxes1.shape[0]
	cols = bboxes2.shape[0]
	overlap = np.zeros((rows, cols), dtype=np.float32)
	if rows * cols == 0:
		return overlap

	area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
	area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
	for i in range(bboxes1.shape[0]):
		x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
		y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
		x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
		y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
		overlap_t = np.maximum(x_end - x_start + 1, 0) * np.maximum(y_end - y_start + 1, 0)
		overlap[i, :] = overlap_t
	if not same:
		bboxes1_occluded=np.max(overlap,axis=1)/area1
		bboxes2_occluded=np.max(overlap,axis=0)/area2
		return bboxes1_occluded,bboxes2_occluded
	else:
		diagonal=np.arange(overlap.shape[0])
		overlap[diagonal,diagonal]=0
		bboxes1_occluded=np.max(overlap,axis=1)/area1
		return bboxes1_occluded


def ratio_check(occluded):
	ratio=np.linspace(0,1,11).tolist()
	count=[0 for i in range(11)]
	index=[[] for i in range(11)]
	all_index=np.arange(len(occluded))
	ratio_select=np.where((occluded==0))[0]
	count[0]=len(ratio_select)
	index[0]=ratio_select
	for i in range(1,len(ratio)-1):
		#ratio_select=(occluded>ratio[i]).astype(np.int)*(occluded<ratio[i+1]).astype(np.int)
		ratio_select=np.where((occluded>ratio[i]) * (occluded<=ratio[i+1]))[0]
		count[i]=len(ratio_select)
		index[i]=ratio_select
		# if count[i]>0:
		# 	print(ratio_select)

	return count,index
jsonfile_name='kitti_2class.json'
data_path='/backdata01/KITTI/kitti/tracking'
data=mmcv.load(os.path.join(data_path,jsonfile_name))



# new_classes=['vehicle','person']
new_classes=[1,2]
#11 ratio from 0.1 to 1
occlusion_ratio=np.linspace(0,1,11).tolist()
occlusion_count_init=[0 for i in range(11)]
occlusion_index_init=[[] for i in range(11)]
occlusion_case_count_init=[[] for i in range(11)]
occlusion_case_index_init=[[] for i in range(11)]
occlusion_record_init=[[] for i in range(11)]
occlusion_record_count_init=[0 for i in range(11)]
v_video=None
for i in range(len(data)):
	box=np.array(data[i]['ann']['bboxes'])
	label=np.array(data[i]['ann']['labels'])
	if v_video==None:
		vvcase_index_count=occlusion_case_count_init.copy()
		vvcase_index=occlusion_case_index_init.copy()
		vvocclusion_record=occlusion_record_init.copy()
		vvocclusion_record_count=occlusion_record_count_init.copy()
		vvcase_count=occlusion_record_count_init.copy()
		v_video=[{'video_id':data[i]['video_id'],'length':1,'occluded':[],'vvcase_index_count':vvcase_index_count,'vvcase_index':vvcase_index,'vvocclusion_record':vvocclusion_record,'vvocclusion_record_count':vvocclusion_record_count, \
			'vvcase_count':vvcase_count}]
	else:
		if data[i]['video_id']!=v_video[-1]['video_id']:

			v_video[-1]['vvcase_index_count']=vvcase_index_count.copy()
			v_video[-1]['vvcase_index']=vvcase_index.copy()
			v_video[-1]['vvocclusion_record']=vvocclusion_record.copy()
			v_video[-1]['vvocclusion_record_count']=vvocclusion_record_count.copy()
			v_video[-1]['vvcase_count']=vvcase_count.copy()
			occlusion_count_init=[0 for i in range(11)]
			occlusion_index_init=[[] for i in range(11)]
			occlusion_case_count_init=[[] for i in range(11)]
			occlusion_case_index_init=[[] for i in range(11)]
			occlusion_record_init=[[] for i in range(11)]
			occlusion_record_count_init=[0 for i in range(11)]
			vvcase_index_count=occlusion_case_count_init.copy()
			vvcase_index=occlusion_case_index_init.copy()
			vvocclusion_record=occlusion_record_init.copy()
			vvocclusion_record_count=occlusion_record_count_init.copy()
			vvcase_count=occlusion_record_count_init.copy()
			v_video.append({'video_id':data[i]['video_id'],'length':1,'occluded':[],'vvcase_index_count':vvcase_index_count,'vvcase_index':vvcase_index,'vvocclusion_record':vvocclusion_record,'vvocclusion_record_count':vvocclusion_record_count, \
				'vvcase_count':vvcase_count})
		else:
			v_video[-1]['length']+=1
	if label.shape[0]==0:
		continue
	vbox=box[label==1]
	b_index=np.where(label==1)[0]
	if len(vbox)>0:
		voccluded=bbox_occluded(vbox,vbox,same=True)
		vcount,vindex=ratio_check(voccluded)

		for j in range(11):
			if vcount[j]>0:
				vvocclusion_record[j].append(i)
				vvcase_index[j].append(b_index[vindex[j]])
				vvcase_index_count[j].append(vcount[j])
		if (vcount[0]/(1e-4+np.sum(vcount)))<2/3:
			v_video[-1]['occluded'].append(i)
testv=[]
for i in range(len(v_video)):
	vvcase_index_count=v_video[i]['vvcase_index_count']
	vv_count=np.array(vvcase_index_count)
	vvocclusion_record=v_video[i]['vvocclusion_record']
	vvcase_count=v_video[i]['vvcase_count']
	for j in range(11):
		vvocclusion_record_count[j]=len(vvocclusion_record[j])
		# 这个记录的所有帧中标记框的数量，那么每个变量里面记录的是每帧中满足某个条件的数量
		vvcase_count[j]=np.array(vvcase_index_count[j]).sum()
	# print('vv count of bbox')
	# print(vvcase_count)
	# print('vv count of frame')
 	# 没有遮挡的框数量的4倍跟有遮挡的框的总数量
	#
	if len(v_video[i]['occluded'])/v_video[i]['length']>0.6 or vvcase_count[0]*4<np.sum(vvcase_count[1:]):
		testv.append([v_video[i]['video_id'],len(v_video[i]['occluded']),v_video[i]['length']])
	print('174 debug')
	from IPython import embed
	embed()
print(testv)





occlusion_count_init=[0 for i in range(11)]
occlusion_index_init=[[] for i in range(11)]
occlusion_case_count_init=[[] for i in range(11)]
occlusion_case_index_init=[[] for i in range(11)]
occlusion_record_init=[[] for i in range(11)]
occlusion_record_count_init=[0 for i in range(11)]
p_video=None
for i in range(len(data)):
	box=np.array(data[i]['ann']['bboxes'])
	label=np.array(data[i]['ann']['labels'])
	if p_video==None:
		ppcase_index_count=occlusion_case_count_init.copy()
		ppcase_index=occlusion_case_index_init.copy()
		ppocclusion_record=occlusion_record_init.copy()
		ppocclusion_record_count=occlusion_record_count_init.copy()
		ppcase_count=occlusion_record_count_init.copy()
		p_video=[{'video_id':data[i]['video_id'],'length':1,'occluded':[],'ppcase_index_count':ppcase_index_count,'ppcase_index':ppcase_index,'ppocclusion_record':ppocclusion_record,'ppocclusion_record_count':ppocclusion_record_count, \
			'ppcase_count':ppcase_count}]
	else:
		if data[i]['video_id']!=p_video[-1]['video_id']:

			p_video[-1]['ppcase_index_count']=ppcase_index_count.copy()
			p_video[-1]['ppcase_index']=ppcase_index.copy()
			p_video[-1]['ppocclusion_record']=ppocclusion_record.copy()
			p_video[-1]['ppocclusion_record_count']=ppocclusion_record_count.copy()
			p_video[-1]['ppcase_count']=ppcase_count.copy()
			occlusion_count_init=[0 for i in range(11)]  #没用
			occlusion_index_init=[[] for i in range(11)]  #没用
			occlusion_case_count_init=[[] for i in range(11)]  #针对BBox
			occlusion_case_index_init=[[] for i in range(11)]  #
			occlusion_record_init=[[] for i in range(11)]  #在整个数据库中的index
			occlusion_record_count_init=[0 for i in range(11)]
			ppcase_index_count=occlusion_case_count_init.copy()
			ppcase_index=occlusion_case_index_init.copy()
			ppocclusion_record=occlusion_record_init.copy()
			ppocclusion_record_count=occlusion_record_count_init.copy()
			ppcase_count=occlusion_record_count_init.copy()
			p_video.append({'video_id':data[i]['video_id'],'length':1,'occluded':[],'ppcase_index_count':ppcase_index_count,'ppcase_index':ppcase_index,'ppocclusion_record':ppocclusion_record,'ppocclusion_record_count':ppocclusion_record_count, \
				'ppcase_count':ppcase_count})
		else:
			p_video[-1]['length']+=1
	if label.shape[0]==0:
		continue
	pbox=box[label==2]
	b_index=np.where(label==2)[0]
	if len(pbox)>0:
		poccluded=bbox_occluded(pbox,pbox,same=True)
		pcount,pindex=ratio_check(poccluded)
		if (pcount[0]/(1e-4+np.sum(pcount)))<2/3:
			p_video[-1]['occluded'].append(i)
		for j in range(11):
			if pcount[j]>0:
				ppocclusion_record[j].append(i)
				ppcase_index[j].append(b_index[pindex[j]])
				ppcase_index_count[j].append(pcount[j])

testp=[]
for i in range(len(p_video)):
	ppcase_index_count=p_video[i]['ppcase_index_count']
	pp_count=np.array(ppcase_index_count)
	ppocclusion_record=p_video[i]['ppocclusion_record']
	ppcase_count=p_video[i]['ppcase_count']
	for j in range(11):
		ppocclusion_record_count[j]=len(ppocclusion_record[j])
		ppcase_count[j]=np.array(ppcase_index_count[j]).sum()
	# print('pp count of bbox')
	# print(ppcase_count)
	# print('pp count of frame')
	if len(p_video[i]['occluded'])/p_video[i]['length']>0.6 or ppcase_count[0]*4<np.sum(ppcase_count[1:]):
		testp.append([p_video[i]['video_id'],len(p_video[i]['occluded']),p_video[i]['length']])
print(testp)



occlusion_count_init=[0 for i in range(11)]
occlusion_index_init=[[] for i in range(11)]
occlusion_case_count_init=[[] for i in range(11)]
occlusion_case_index_init=[[] for i in range(11)]
occlusion_record_init=[[] for i in range(11)]
occlusion_record_count_init=[0 for i in range(11)]
vp_video=None
for i in range(len(data)):
	box=np.array(data[i]['ann']['bboxes'])
	label=np.array(data[i]['ann']['labels'])
	if vp_video==None:
		vpcase_index_count=occlusion_case_count_init.copy()
		vpcase_index=occlusion_case_index_init.copy()
		vpocclusion_record=occlusion_record_init.copy()
		vpocclusion_record_count=occlusion_record_count_init.copy()
		vpcase_count=occlusion_record_count_init.copy()
		vp_video=[{'video_id':data[i]['video_id'],'length':1,'occluded':[],'vpcase_index_count':vpcase_index_count,'vpcase_index':vpcase_index,'vpocclusion_record':vpocclusion_record,'vpocclusion_record_count':vpocclusion_record_count, \
			'vpcase_count':vpcase_count}]
	else:
		if data[i]['video_id']!=vp_video[-1]['video_id']:

			vp_video[-1]['vpcase_index_count']=vpcase_index_count.copy()
			vp_video[-1]['vpcase_index']=vpcase_index.copy()
			vp_video[-1]['vpocclusion_record']=vpocclusion_record.copy()
			vp_video[-1]['vpocclusion_record_count']=vpocclusion_record_count.copy()
			vp_video[-1]['vpcase_count']=vpcase_count.copy()
			occlusion_count_init=[0 for i in range(11)]
			occlusion_index_init=[[] for i in range(11)]
			occlusion_case_count_init=[[] for i in range(11)]
			occlusion_case_index_init=[[] for i in range(11)]
			occlusion_record_init=[[] for i in range(11)]
			occlusion_record_count_init=[0 for i in range(11)]
			vpcase_index_count=occlusion_case_count_init.copy()
			vpcase_index=occlusion_case_index_init.copy()
			vpocclusion_record=occlusion_record_init.copy()
			vpocclusion_record_count=occlusion_record_count_init.copy()
			vpcase_count=occlusion_record_count_init.copy()
			vp_video.append({'video_id':data[i]['video_id'],'length':1,'occluded':[],'vpcase_index_count':vpcase_index_count,'vpcase_index':vpcase_index,'vpocclusion_record':vpocclusion_record,'vpocclusion_record_count':vpocclusion_record_count, \
				'vpcase_count':vpcase_count})
		else:
			vp_video[-1]['length']+=1
	if label.shape[0]==0:
		continue
	vbox=box[label==1]
	vbindex=np.where(label==1)[0]
	pbox=box[label==2]
	pbindex=np.where(label==2)[0]
	if len(vbox)>0 and len(pbox)>0:
		vpoccluded,pvoccluded=bbox_occluded(vbox,pbox,same=False)
		vcount,vindex=ratio_check(vpoccluded)
		pcount,pindex=ratio_check(pvoccluded)
		for j in range(11):
			if vcount[j]+pcount[j]>0:
				vpocclusion_record[j].append(i)
				vpcase_index[j].append(np.array(vbindex[vindex[j]].tolist() + pbindex[pindex[j]].tolist()))
				vpcase_index_count[j].append(vcount[j]+pcount[j])
		if (vcount[0]/(1e-4+np.sum(vcount)))<2/3 or (pcount[0]/(1e-4+np.sum(pcount)))<2/3 :
			vp_video[-1]['occluded'].append(i)





testvp=[]
for i in range(len(vp_video)):

	vpcase_index_count=vp_video[i]['vpcase_index_count']
	vp_count=np.array(vpcase_index_count)
	vpocclusion_record=vp_video[i]['vpocclusion_record']
	vpcase_count=vp_video[i]['vpcase_count']
	for j in range(11):
		vpocclusion_record_count[j]=len(vpocclusion_record[j])
		vpcase_count[j]=np.array(vpcase_index_count[j]).sum()
	# print('vp count of bbox')
	# print(vpcase_count)
	# print('vp count of frame')
	# print(vpocclusion_record_count)
	if len(vp_video[i]['occluded'])/vp_video[i]['length']>0.5 or vpcase_count[0]*3<np.sum(vpcase_count[1:]):
		testvp.append([vp_video[i]['video_id'],len(vp_video[i]['occluded']),vp_video[i]['length']])
print(testvp)
test_all=testv+testp+testvp
test_fuse=[]
for i in range(len(test_all)):
	if test_all[i][0] not in test_fuse:
		test_fuse.append(test_all[i][0])
test_fuse.sort()
print(test_fuse)
test_data_all=[]
train_data_all=[]
for i in range(len(data)):
	if data[i]['video_id'] in test_fuse:
		test_data_all.append(data[i])
	else:
		train_data_all.append(data[i])






test_data_v=[[] for i in range(10)]	
for i in range(len(v_video)):
	if v_video[i]['video_id'] in test_fuse:
		vvcase_index=v_video[i]['vvcase_index']
		vvocclusion_record=v_video[i]['vvocclusion_record']
		for j in range(1,11):
			for m in range(len(vvcase_index[j])):
				data_index=vvocclusion_record[j][m]
				vvcase_index_t=vvcase_index[j][m]
				data_t=data[data_index]
				data_t['occ_index']=vvcase_index_t.tolist()
				data_t['data_index']=data_index
				test_data_v[j-1].append(data_t)
test_data_p=[[] for i in range(10)]	
for i in range(len(p_video)):
	if p_video[i]['video_id'] in test_fuse:
		ppcase_index=p_video[i]['ppcase_index']
		ppocclusion_record=p_video[i]['ppocclusion_record']
		for j in range(1,11):
			for m in range(len(ppcase_index[j])):
				data_index=ppocclusion_record[j][m]
				ppcase_index_t=ppcase_index[j][m]
				data_t=data[data_index]
				data_t['occ_index']=ppcase_index_t.tolist()
				data_t['data_index']=data_index
				test_data_p[j-1].append(data_t)

test_data_vp=[[] for i in range(10)]	
for i in range(len(vp_video)):
	if vp_video[i]['video_id'] in test_fuse:
		vpcase_index=vp_video[i]['vpcase_index']
		vpocclusion_record=vp_video[i]['vpocclusion_record']
		for j in range(1,11):
			for m in range(len(vpcase_index[j])):
				data_index=vpocclusion_record[j][m]
				vpcase_index_t=vpcase_index[j][m]
				data_t=data[data_index]
				data_t['occ_index']=vpcase_index_t.tolist()
				data_t['data_index']=data_index
				test_data_vp[j-1].append(data_t)
mmcv.dump(test_data_all,os.path.join(data_path,'occlusion_test_zxn.json'))
mmcv.dump(train_data_all,os.path.join(data_path,'occlusion_train_zxn.json'))
mmcv.dump(test_data_v,os.path.join(data_path,'occlusion_test_v_zxn.json'))
mmcv.dump(test_data_p,os.path.join(data_path,'occlusion_test_p_zxn.json'))
mmcv.dump(test_data_vp,os.path.join(data_path,'occlusion_test_vp_zxn.json'))
print('frame count')
print('train_data_all:',len(train_data_all),'test_data_all:',len(test_data_all))
for i in range(10):
	print('ratio:',occlusion_ratio[i],'-',occlusion_ratio[i+1],'vehecle and vehicle:',len(test_data_v[i]),'person and person:',len(test_data_p[i]),'vehicle and person:',len(test_data_vp[i]))
