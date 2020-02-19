import mmcv
import os
import time
import ffmpeg
import json
import numpy as np
import copy
def bbox_overlaps(bboxes1, bboxes2):
	bboxes1=np.array(bboxes1)
	bboxes2=np.array(bboxes2)
	bboxes1 = bboxes1.astype(np.float32)
	bboxes2 = bboxes2.astype(np.float32)
	if bboxes1.shape[1]>4:
		bboxes1=bboxes1[:,:4]
		bboxes2=bboxes2[:,:4]
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
		union = area1[i] + area2 - overlap
		ious[i, :] = overlap / union
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

	area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
		bboxes1[:, 3] - bboxes1[:, 1] + 1)
	area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
		bboxes2[:, 3] - bboxes2[:, 1] + 1)
	for i in range(bboxes1.shape[0]):
		x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
		y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
		x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
		y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
		overlap_t = np.maximum(x_end - x_start + 1, 0) * np.maximum(
			y_end - y_start + 1, 0)
		overlap[i, :] = overlap_t
	if not same:
		bboxes1_occluded=np.max(overlap,axis=1)/area1
		bboxes2_occluded=np.max(overlap,axis=0)/area2
		bboxes1_pair=np.argmax(overlap,axis=1)
		bboxes2_pair=np.argmax(overlap,axis=0)
		return bboxes1_occluded,bboxes1_pair,bboxes2_occluded,bboxes2_pair
	else:
		diagonal=np.arange(overlap.shape[0])
		overlap[diagonal,diagonal]=0
		bboxes1_occluded=np.max(overlap,axis=1)/area1
		occluded_pair=np.argmax(overlap,axis=1)
		return bboxes1_occluded,occluded_pair
def ratio_check(occluded):
	ratio=np.arange(0,11).astype(np.float)/10
	count=[0 for i in range(11)]
	index=[[] for i in range(11)]
	all_index=np.arange(len(occluded))
	ratio_select=np.where((occluded==0))[0]
	count[0]=len(ratio_select)
	index[0]=ratio_select
	for i in range(1,len(ratio)-1):
		#ratio_select=(occluded>ratio[i]).astype(np.int)*(occluded<ratio[i+1]).astype(np.int)
		ratio_select=np.where((occluded>ratio[i]) * (occluded<=ratio[i+1]))[0]
		# print(ratio[i],occluded,ratio_select)
		count[i]=len(ratio_select)
		index[i]=ratio_select
		# if count[i]>0:
		# 	print(ratio_select)

	return count,index
jsonfile_name='bdd_all_2class.json'
data_path='/backdata01/bdd/'
data=mmcv.load(os.path.join(data_path,jsonfile_name))


# new_classes=['vehicle','person']
new_classes=[1,2]
#11 ratio from 0.1 to 1
occlusion_ratio=np.arange(0,11).astype(np.float)/10
occlusion_count_init=[0 for i in range(11)]
occlusion_index_init=[[] for i in range(11)]
occlusion_case_count_init=[[] for i in range(11)]
occlusion_case_index_init=[[] for i in range(11)]
occlusion_pair_index_init=[[] for i in range(11)]
occlusion_record_init=[[] for i in range(11)]
occlusion_record_count_init=[0 for i in range(11)]
v_video=None
for i in range(len(data)):
	box=np.array(data[i]['ann']['bboxes'])
	label=np.array(data[i]['ann']['labels'])
	if v_video==None:
		vvcase_index_count=occlusion_case_count_init.copy()
		vvcase_index=occlusion_case_index_init.copy()
		vvcase_index_pair=occlusion_pair_index_init.copy()

		vvocclusion_record=occlusion_record_init.copy()
		vvocclusion_record_count=occlusion_record_count_init.copy()
		vvcase_count=occlusion_record_count_init.copy()
		v_video=[{'video_id':data[i]['video_id'],'length':1,'occluded':[],'vvcase_index_count':vvcase_index_count,'vvcase_index':vvcase_index, 'vvcase_index_pair':vvcase_index_pair,\
			'vvocclusion_record':vvocclusion_record,'vvocclusion_record_count':vvocclusion_record_count, \
			'vvcase_count':vvcase_count}]
	else:
		if data[i]['video_id']!=v_video[-1]['video_id']:

			v_video[-1]['vvcase_index_count']=vvcase_index_count.copy()
			v_video[-1]['vvcase_index']=vvcase_index.copy()
			v_video[-1]['vvcase_index_pair']=vvcase_index_pair.copy()
			v_video[-1]['vvocclusion_record']=vvocclusion_record.copy()
			v_video[-1]['vvocclusion_record_count']=vvocclusion_record_count.copy()
			v_video[-1]['vvcase_count']=vvcase_count.copy()
			occlusion_count_init=[0 for i in range(11)]
			occlusion_index_init=[[] for i in range(11)]
			occlusion_case_count_init=[[] for i in range(11)]
			occlusion_case_index_init=[[] for i in range(11)]
			occlusion_pair_index_init=[[] for i in range(11)]
			occlusion_record_init=[[] for i in range(11)]
			occlusion_record_count_init=[0 for i in range(11)]
			vvcase_index_count=occlusion_case_count_init.copy()
			vvcase_index=occlusion_case_index_init.copy()
			vvcase_index_pair=occlusion_pair_index_init.copy()
			vvocclusion_record=occlusion_record_init.copy()
			vvocclusion_record_count=occlusion_record_count_init.copy()
			vvcase_count=occlusion_record_count_init.copy()
			v_video.append({'video_id':data[i]['video_id'],'length':1,'occluded':[],'vvcase_index_count':vvcase_index_count,'vvcase_index':vvcase_index, 'vvcase_index_pair':vvcase_index_pair,\
				'vvocclusion_record':vvocclusion_record,'vvocclusion_record_count':vvocclusion_record_count, \
				'vvcase_count':vvcase_count})
		else:
			v_video[-1]['length']+=1
	if label.shape[0]==0:
		continue
	vbox=box[label==1]
	b_index=np.where(label==1)[0]

	if len(vbox)>0:
		voccluded,voccluded_pair=bbox_occluded(vbox,vbox,same=True)
		vcount,vindex=ratio_check(voccluded)
		for j in range(11):
			if vcount[j]>0:
				vvocclusion_record[j].append(i)
				vvcase_index[j].append(b_index[vindex[j]])
				vvcase_index_count[j].append(vcount[j])
				vvcase_index_pair[j].append(b_index[voccluded_pair[vindex[j]]])
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
		vvcase_count[j]=np.array(vvcase_index_count[j]).sum()
	# print('vv count of bbox')
	# print(vvcase_count)
	# print('vv count of frame')
	# print(v_video[i]['length'])
	if len(v_video[i]['occluded'])/v_video[i]['length']>0.8 and vvcase_count[0]*5<np.sum(vvcase_count[1:]) and np.sum(vvcase_count[1:])>5*v_video[i]['length']:
		testv.append([v_video[i]['video_id'],len(v_video[i]['occluded']),v_video[i]['length']])
print(testv,len(testv))
# exit()

occlusion_ratio=np.arange(0,11).astype(np.float)/10
occlusion_count_init=[0 for i in range(11)]
occlusion_index_init=[[] for i in range(11)]
occlusion_case_count_init=[[] for i in range(11)]
occlusion_case_index_init=[[] for i in range(11)]
occlusion_pair_index_init=[[] for i in range(11)]
occlusion_record_init=[[] for i in range(11)]
occlusion_record_count_init=[0 for i in range(11)]
p_video=None
for i in range(len(data)):
	box=np.array(data[i]['ann']['bboxes'])
	label=np.array(data[i]['ann']['labels'])
	if p_video==None:
		ppcase_index_count=occlusion_case_count_init.copy()
		ppcase_index=occlusion_case_index_init.copy()
		ppcase_index_pair=occlusion_pair_index_init.copy()
		ppocclusion_record=occlusion_record_init.copy()
		ppocclusion_record_count=occlusion_record_count_init.copy()
		ppcase_count=occlusion_record_count_init.copy()
		p_video=[{'video_id':data[i]['video_id'],'length':1,'occluded':[],'ppcase_index_count':ppcase_index_count,'ppcase_index':ppcase_index, 'ppcase_index_pair':ppcase_index_pair,\
			'ppocclusion_record':ppocclusion_record,'ppocclusion_record_count':ppocclusion_record_count, \
			'ppcase_count':ppcase_count}]
	else:
		if data[i]['video_id']!=p_video[-1]['video_id']:

			p_video[-1]['ppcase_index_count']=ppcase_index_count.copy()
			p_video[-1]['ppcase_index']=ppcase_index.copy()
			p_video[-1]['ppcase_index_pair']=ppcase_index_pair.copy()
			p_video[-1]['ppocclusion_record']=ppocclusion_record.copy()
			p_video[-1]['ppocclusion_record_count']=ppocclusion_record_count.copy()
			p_video[-1]['ppcase_count']=ppcase_count.copy()
			occlusion_count_init=[0 for i in range(11)]
			occlusion_index_init=[[] for i in range(11)]
			occlusion_case_count_init=[[] for i in range(11)]
			occlusion_case_index_init=[[] for i in range(11)]
			occlusion_pair_index_init=[[] for i in range(11)]
			occlusion_record_init=[[] for i in range(11)]
			occlusion_record_count_init=[0 for i in range(11)]
			ppcase_index_count=occlusion_case_count_init.copy()
			ppcase_index=occlusion_case_index_init.copy()
			ppcase_index_pair=occlusion_pair_index_init.copy()
			ppocclusion_record=occlusion_record_init.copy()
			ppocclusion_record_count=occlusion_record_count_init.copy()
			ppcase_count=occlusion_record_count_init.copy()
			p_video.append({'video_id':data[i]['video_id'],'length':1,'occluded':[],'ppcase_index_count':ppcase_index_count,'ppcase_index':ppcase_index, 'ppcase_index_pair':ppcase_index_pair,\
				'ppocclusion_record':ppocclusion_record,'ppocclusion_record_count':ppocclusion_record_count, \
				'ppcase_count':ppcase_count})
		else:
			p_video[-1]['length']+=1
	if label.shape[0]==0:
		continue
	pbox=box[label==2]
	b_index=np.where(label==2)[0]
	if len(pbox)>0:
		poccluded,poccluded_pair=bbox_occluded(pbox,pbox,same=True)
		pcount,pindex=ratio_check(poccluded)
		for j in range(11):
			if pcount[j]>0:
				ppocclusion_record[j].append(i)
				ppcase_index[j].append(b_index[pindex[j]])
				ppcase_index_count[j].append(pcount[j])
				ppcase_index_pair[j].append(b_index[poccluded_pair[pindex[j]]])
		if (pcount[0]/(1e-4+np.sum(pcount)))<2/3:
			p_video[-1]['occluded'].append(i)
testp=[]
for i in range(len(p_video)):
	ppcase_index_count=p_video[i]['ppcase_index_count']
	pp_count=np.array(ppcase_index_count)
	ppocclusion_record=p_video[i]['ppocclusion_record']
	ppcase_count=p_video[i]['ppcase_count']
	for j in range(11):
		ppocclusion_record_count[j]=len(ppocclusion_record[j])
		ppcase_count[j]=np.array(ppcase_index_count[j]).sum()
	# print(p_video[i]['video_id'])
	# print('pp count of bbox')
	# print(ppcase_count)
	# print('pp count of frame')
	# print(p_video[i]['length'])
	if ppcase_count[0]>3*p_video[i]['length']:
		testp.append([p_video[i]['video_id'],len(p_video[i]['occluded']),p_video[i]['length']])
print(testp)

occlusion_ratio=np.arange(0,11).astype(np.float)/10
occlusion_count_init=[0 for i in range(11)]
occlusion_index_init=[[] for i in range(11)]
occlusion_case_count_init=[[] for i in range(11)]
occlusion_case_index_init=[[] for i in range(11)]
occlusion_pair_index_init=[[] for i in range(11)]
occlusion_record_init=[[] for i in range(11)]
occlusion_record_count_init=[0 for i in range(11)]
vp_video=None
for i in range(len(data)):
	box=np.array(data[i]['ann']['bboxes'])
	label=np.array(data[i]['ann']['labels'])
	if vp_video==None:
		vpcase_index_count=occlusion_case_count_init.copy()
		vpcase_index=occlusion_case_index_init.copy()
		vpcase_index_pair=occlusion_pair_index_init.copy()
		vpocclusion_record=occlusion_record_init.copy()
		vpocclusion_record_count=occlusion_record_count_init.copy()
		vpcase_count=occlusion_record_count_init.copy()
		vp_video=[{'video_id':data[i]['video_id'],'length':1,'occluded':[],'vpcase_index_count':vpcase_index_count,'vpcase_index':vpcase_index, 'vpcase_index_pair':vpcase_index_pair,\
			'vpocclusion_record':vpocclusion_record,'vpocclusion_record_count':vpocclusion_record_count, \
			'vpcase_count':vpcase_count}]
	else:
		if data[i]['video_id']!=vp_video[-1]['video_id']:

			vp_video[-1]['vpcase_index_count']=vpcase_index_count.copy()
			vp_video[-1]['vpcase_index']=vpcase_index.copy()
			vp_video[-1]['vpcase_index_pair']=vpcase_index_pair.copy()
			vp_video[-1]['vpocclusion_record']=vpocclusion_record.copy()
			vp_video[-1]['vpocclusion_record_count']=vpocclusion_record_count.copy()
			vp_video[-1]['vpcase_count']=vpcase_count.copy()
			occlusion_count_init=[0 for i in range(11)]
			occlusion_index_init=[[] for i in range(11)]
			occlusion_case_count_init=[[] for i in range(11)]
			occlusion_case_index_init=[[] for i in range(11)]
			occlusion_pair_index_init=[[] for i in range(11)]
			occlusion_record_init=[[] for i in range(11)]
			occlusion_record_count_init=[0 for i in range(11)]
			vpcase_index_count=occlusion_case_count_init.copy()
			vpcase_index=occlusion_case_index_init.copy()
			vpcase_index_pair=occlusion_pair_index_init.copy()
			vpocclusion_record=occlusion_record_init.copy()
			vpocclusion_record_count=occlusion_record_count_init.copy()
			vpcase_count=occlusion_record_count_init.copy()
			vp_video.append({'video_id':data[i]['video_id'],'length':1,'occluded':[],'vpcase_index_count':vpcase_index_count,'vpcase_index':vpcase_index, 'vpcase_index_pair':vpcase_index_pair,\
				'vpocclusion_record':vpocclusion_record,'vpocclusion_record_count':vpocclusion_record_count, \
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
		vpoccluded,vp_pair,pvoccluded,pv_pair=bbox_occluded(vbox,pbox,same=False)
		vcount,vindex=ratio_check(vpoccluded)
		pcount,pindex=ratio_check(pvoccluded)
		for j in range(11):
			if vcount[j]+pcount[j]>0:
				vpocclusion_record[j].append(i)
				vpcase_index[j].append(np.array(vbindex[vindex[j]].tolist() + pbindex[pindex[j]].tolist()))
				vpcase_index_count[j].append(vcount[j]+pcount[j])
				vpcase_index_pair[j].append(np.array(pbindex[vp_pair[vindex[j]]].tolist()+vbindex[pv_pair[pindex[j]]].tolist()))
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
	print('vp count of bbox')
	print(vpcase_count)
	print('vp count of frame')
	print(vpocclusion_record_count)
	if len(vp_video[i]['occluded'])/vp_video[i]['length']>0.5 or np.sum(vpcase_count[1:])>3*vp_video[i]['length']:
		testvp.append([vp_video[i]['video_id'],len(vp_video[i]['occluded']),vp_video[i]['length']])

test_all=testv+testp+testvp
test_fuse=[]
for i in range(len(test_all)):
	if test_all[i][0] not in test_fuse:
		test_fuse.append(test_all[i][0])
test_fuse.sort()
# print(test_fuse,len(test_fuse))

test_data_all=[]
train_data_all=[]
for i in range(len(data)):
	if data[i]['video_id'] in test_fuse:
		data[i]['filename']=os.path.join(data_path,data[i]['filename'])
		test_data_all.append(data[i])
	else:
		data[i]['filename']=os.path.join(data_path,data[i]['filename'])
		train_data_all.append(data[i])

test_data_v=[[] for i in range(10)]	
for i in range(len(v_video)):
	if v_video[i]['video_id'] in test_fuse:
		vvcase_index=v_video[i]['vvcase_index']
		vvcase_index_pair=v_video[i]['vvcase_index_pair']
		vvocclusion_record=v_video[i]['vvocclusion_record']
		for j in range(1,11):
			for m in range(len(vvcase_index[j])):
				data_index=vvocclusion_record[j][m]
				vvcase_index_t=vvcase_index[j][m]
				data_t=data[data_index].copy()
				data_t['filename']=os.path.join(data_path,data_t['filename'])
				data_t['occ_index']=vvcase_index_t.tolist()
				vvcase_index_pair_t=vvcase_index_pair[j][m]
				data_t['occ_index_pair']=vvcase_index_pair_t.tolist()
				data_t['occ_count']=len(data_t['occ_index'])
				data_t['data_index']=data_index
				test_data_v[j-1].append(data_t)


test_data_p=[[] for i in range(10)]	
for i in range(len(p_video)):
	if p_video[i]['video_id'] in test_fuse:
		ppcase_index=p_video[i]['ppcase_index']
		ppcase_index_pair=p_video[i]['ppcase_index_pair']
		ppocclusion_record=p_video[i]['ppocclusion_record']
		for j in range(1,11):
			for m in range(len(ppcase_index[j])):
				data_index=ppocclusion_record[j][m]
				ppcase_index_t=ppcase_index[j][m]
				data_t=data[data_index].copy()
				data_t['filename']=os.path.join(data_path,data_t['filename'])
				data_t['occ_index']=ppcase_index_t.tolist()
				ppcase_index_pair_t=ppcase_index_pair[j][m]
				data_t['occ_index_pair']=ppcase_index_pair_t.tolist()
				data_t['occ_count']=len(data_t['occ_index'])
				data_t['data_index']=data_index
				test_data_p[j-1].append(data_t)


test_data_vp=[[] for i in range(10)]	
for i in range(len(vp_video)):
	if vp_video[i]['video_id'] in test_fuse:
		vpcase_index=vp_video[i]['vpcase_index']
		vpcase_index_pair=vp_video[i]['vpcase_index_pair']
		vpocclusion_record=vp_video[i]['vpocclusion_record']
		for j in range(1,11):
			for m in range(len(vpcase_index[j])):
				data_index=vpocclusion_record[j][m]
				vpcase_index_t=vpcase_index[j][m]
				data_t=data[data_index].copy()
				data_t['filename']=os.path.join(data_path,data_t['filename'])
				data_t['occ_index']=vpcase_index_t.tolist()
				vpcase_index_pair_t=vpcase_index_pair[j][m]
				data_t['occ_index_pair']=vpcase_index_pair_t.tolist()
				data_t['occ_count']=len(data_t['occ_index'])
				data_t['data_index']=data_index
				test_data_vp[j-1].append(data_t)
mmcv.dump(test_data_all,os.path.join(data_path,'occlusion_test.json'))
mmcv.dump(train_data_all,os.path.join(data_path,'occlusion_train.json'))
mmcv.dump(test_data_v,os.path.join(data_path,'occlusion_test_v.json'))
mmcv.dump(test_data_p,os.path.join(data_path,'occlusion_test_p.json'))
mmcv.dump(test_data_vp,os.path.join(data_path,'occlusion_test_vp.json'))

print('frame count')
print('train_data_all:',len(train_data_all),'test_data_all:',len(test_data_all))
for i in range(10):
	print('ratio:',occlusion_ratio[i],'-',occlusion_ratio[i+1],'vehecle and vehicle:',len(test_data_v[i]),'person and person:',len(test_data_p[i]),'vehicle and person:',len(test_data_vp[i]))


print('bbox count')
for i in range(10):
	v_count=0
	p_count=0
	vp_count=0
	for j in range(len(test_data_v[i])):
		v_count+=len(test_data_v[i][j]['occ_index'])
	for j in range(len(test_data_p[i])):
		p_count+=len(test_data_p[i][j]['occ_index'])
	for j in range(len(test_data_vp[i])):
		vp_count+=len(test_data_vp[i][j]['occ_index'])
	print('ratio:',occlusion_ratio[i],'-',occlusion_ratio[i+1],'vehecle and vehicle:',v_count,'person and person:',p_count,'vehicle and person:',vp_count)
# frame count
# train_data_all: 8081 test_data_all: 3653
# ratio: 0.0 - 0.1 vehecle and vehicle: 2646 person and person: 290 vehicle and person: 443
# ratio: 0.1 - 0.2 vehecle and vehicle: 2128 person and person: 261 vehicle and person: 432
# ratio: 0.2 - 0.3 vehecle and vehicle: 2230 person and person: 207 vehicle and person: 313
# ratio: 0.3 - 0.4 vehecle and vehicle: 2148 person and person: 141 vehicle and person: 250
# ratio: 0.4 - 0.5 vehecle and vehicle: 1858 person and person: 110 vehicle and person: 217
# ratio: 0.5 - 0.6 vehecle and vehicle: 1363 person and person: 73  vehicle and person: 195
# ratio: 0.6 - 0.7 vehecle and vehicle: 1251 person and person: 50  vehicle and person: 241
# ratio: 0.7 - 0.8 vehecle and vehicle: 971  person and person: 43  vehicle and person: 228
# ratio: 0.8 - 0.9 vehecle and vehicle: 1304 person and person: 64  vehicle and person: 739
# ratio: 0.9 - 1.0 vehecle and vehicle: 0    person and person: 0   vehicle and person: 0
# bbox count
# ratio: 0.0 - 0.1 vehecle and vehicle: 4468 person and person: 505 vehicle and person: 564
# ratio: 0.1 - 0.2 vehecle and vehicle: 3855 person and person: 446 vehicle and person: 546
# ratio: 0.2 - 0.3 vehecle and vehicle: 3917 person and person: 315 vehicle and person: 363
# ratio: 0.3 - 0.4 vehecle and vehicle: 3661 person and person: 184 vehicle and person: 286
# ratio: 0.4 - 0.5 vehecle and vehicle: 2960 person and person: 131 vehicle and person: 242
# ratio: 0.5 - 0.6 vehecle and vehicle: 2092 person and person: 88  vehicle and person: 223
# ratio: 0.6 - 0.7 vehecle and vehicle: 1669 person and person: 57  vehicle and person: 261
# ratio: 0.7 - 0.8 vehecle and vehicle: 1138 person and person: 46  vehicle and person: 242
# ratio: 0.8 - 0.9 vehecle and vehicle: 1944 person and person: 69  vehicle and person: 1039
# ratio: 0.9 - 1.0 vehecle and vehicle: 0    person and person: 0   vehicle and person: 0
