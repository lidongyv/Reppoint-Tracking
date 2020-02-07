import os
import numpy as np
import json
import shutil
import cv2
# kitti:
# {'filename': 'training/image_02/0000/000000.png', 'flow_name': 'training/Flow/0000/000000.flo', 'inv_flow_name': 'training/Inv_Flow/0000/000000.flo', 'width': 1242, 'height': 375, \
# 	'video_id': '0000', 'frame_id': 0, 'ann': {'bboxes': [[296.744956, 161.752147, 455.226042, 292.372804], [737.619499, 161.531951, 931.112229, 374.0], [1106.137292, 166.576807, 1204.470628, 323.876144]], \
# 		'labels': [1, 3, 2], 'occluded': [0, 0, 0], 'truncated': [0, 0, 0], 'alpha': [-1.793451, -1.936993, -2.523309], 'dimensions': [[2.0, 1.823255, 4.433886], [1.739063, 0.824591, 1.785241], \
# 			[1.714062, 0.767881, 0.972283]], 'location': [[-4.552284, 1.858523, 13.410495], [1.6404, 1.67566, 5.776261], [6.301919, 1.652419, 8.455685]], 'ry': [-2.115488, -1.675458, -1.900245], \
# 				'intrinsic': [[721.5377, 0.0, 609.5593, 44.85728, 0.0, 721.5377, 172.854, 0.2163791, 0.0, 0.0, 1.0, 0.002745884], [721.5377, 0.0, 609.5593, 44.85728, 0.0, 721.5377, 172.854, 0.2163791, \
# 					0.0, 0.0, 1.0, 0.002745884], [721.5377, 0.0, 609.5593, 44.85728, 0.0, 721.5377, 172.854, 0.2163791, 0.0, 0.0, 1.0, 0.002745884]], 'bboxes_ignore': [[219.31, 188.49, 245.5, 218.56], [47.56, \
# 						195.28, 115.48, 221.48]], 'track_id': [0, 1, 2], 'track_id_ignore': [-1, -1]}}
# 'filename','width','height','video_id','frame_id','ann':{'bboxes','labels','track_id'}
# bdd:
# {'videoName': '004ea016-0b1932a7', 'name': '004ea016-0b1932a7-0000001.jpg', 'index': 0, 'timestamp': 0, 'labels': [{'category': 'truck', 'poly2d': None, 'box3d': None, \
# 'box2d': {'x2': 87.24030445052892, 'y1': 180.58743021259488, 'x1': 0, 'y2': 428.34989485209707}, 'manualShape': True, 'attributes': {'Truncated': True, 'Occluded': True}, \
# 'id': 191}, {'category': 'car', 'poly2d': None, 'box3d': None, 'box2d': {'x2': 247.18557270077517, 'y1': 336.1120897089809, 'x1': 10.550603712837965, 'y2': 460.4584906102855}, \
# 'manualShape': True, 'attributes': {'Truncated': False, 'Occluded': False}, 'id': 597}, {'category': 'car', 'poly2d': None, 'box3d': None, 'box2d': {'x2': 850.0207623837421, 'y1': \
# 321.7802886047864, 'x1': 636.7405710927083, 'y2': 476.1604270683544}, 'manualShape': True, 'attributes': {'Truncated': False, 'Occluded': False}, 'id': 1206}, {'category': 'bus', 'poly2d': None, \
# 'box3d': None, 'box2d': {'x2': 430.45752893456137, 'y1': 335.39077700251147, 'x1': 390.8955886895468, 'y2': 379.67653100812475}, 'manualShape': True, 'attributes': {'Truncated': False, 'Occluded': False}, \
# 'id': 1409}, {'category': 'car', 'poly2d': None, 'box3d': None, 'box2d': {'x2': 510.56177851710845, 'y1': 356.56740764182456, 'x1': 476.55671239322874, 'y2': 384.2572471998409}, 'manualShape': True, \
# 'attributes': {'Truncated': False, 'Occluded': False}, 'id': 1815}, {'category': 'car', 'poly2d': None, 'box3d': None, 'box2d': {'x2': 449.35265949412496, 'y1': 355.59583432399944, 'x1': 423.60596657175887, \
# 'y2': 375.99887399832727}, 'manualShape': True, 'attributes': {'Truncated': False, 'Occluded': True}, 'id': 2424}, {'category': 'car', 'poly2d': None, 'box3d': None, 'box2d': {'x2': 524.649591625573, 'y1': \
# 359.94478155616906, 'x1': 503.36750942559377, 'y2': 375.6750162257189}, 'manualShape': True, 'attributes': {'Truncated': False, 'Occluded': True}, 'id': 3236}], 'attributes': None}
# bdd class
classes=['other', 'truck', 'car', 'bus', 'person', 'rider', 'motor', 'bike']
vehicle=['bus','car','truck']
person=['person','rider']
cyclist=['bike','motor']
new_classes=['other','vehicle','person','cyclist']
id=[i for i in range(len(new_classes))]
class_count=[0 for i in range(len(new_classes))]
class_count=dict(zip(new_classes,class_count))
path='/backdata01/bdd/'
label=path+'bdd100k_tracking_cvpr2019_val.json'
with open(label) as f:
	label=json.load(f)

data=[]

video_name=['004ea016-0b1932a7', '003baca5-70c87fc6', '003c4a61-52588960', '004071a4-4e8a363a', '00390995-49d9a01e', '00391a82-8be5b76d', '003baca5-d6cd84e5', '00423717-0ef3c8dc', '003baca5-ad660439', \
'004071a4-a45d905f', '004071a4-ef4bf541', '0035afff-47378fa3', '0035afff-bd191d6a', '00378858-c5f802ac', '00391a82-d1428e56', '003baca5-aab2e274', '0035afff-572b2d4e', '003e23ee-07d32feb', '003e23ee-67d25f19', \
'004071a4-049b7b85', '00417c23-220bbc98', '007da0eb-8cca23d7', '007eddfc-528c4da4', '00810e80-37641274', '007aeb45-eef41701', '007c01ea-a02f29ef', '007c11bf-f6da335c', '007da0eb-e1f588e8', '007eddfc-f8a80310', \
'007aeb45-9330e852', '007aeb45-c601742b', '007b11e5-1033ff33', '007c01ea-63aa326c', '007c01ea-ad9f940b', '007eddfc-bcaeb35b', '007aeb45-56d1aed9', '007aeb45-f9f5ac8c', '007da0eb-1bad8468', '0080b637-55f7930e', \
'007aeb45-96ce245e', '007b11e5-c22ddae8', '0070bc56-49cf077c', '0070bc56-8d8cfd82', '0071d9c5-0f52d539', '0075a5b0-9a8d5dbb', '007ae77f-79995643', '0070bc56-4c0bb2d5', '0071d9c5-be7394cc', '007693e6-2535e7bf', \
'007693e6-c2a8b9a7', '00779058-517a4591', '00714cd3-48d6b290', '0075b179-8e09869a', '0077ccb8-d5778190', '007aeb45-3e75ce0e', '00787a90-b350f376', '0070bc56-29bcc943', '0070bc56-401304be', '0070bc56-7d01076a', \
'00721168-56efa5c2', '007693e6-bc55f0e4']
new_video_name=["{0:04d}".format(i) for i in range(len(video_name))]
out_name=dict(zip(video_name,new_video_name))
# for i in range(len(new_video_name)):
# 	os.mkdir(os.path.join(path,'training',new_video_name[i]))
video_length=[0 for i in range(len(video_name))]
video_class_count=[class_count.copy() for i in range(len(video_name))]
video_class_count=dict(zip(video_name,video_class_count.copy()))
video_count=dict(zip(video_name,video_length))

for i in range(len(label)):
	video_count[label[i]['videoName']]+=1
	if label[i]['labels'] is not None:
		tmp_label=[0 for m in range(len(label[i]['labels'])) if label[i]['labels'][m]['category']!='other']
		tmp_bbox=[[0,0,0,0] for m in range(len(label[i]['labels'])) if label[i]['labels'][m]['category']!='other']
		tmp_track=[0 for m in range(len(label[i]['labels'])) if label[i]['labels'][m]['category']!='other']
		tmp_count=0
		for j in range(len(label[i]['labels'])):
			if label[i]['labels'][j]['category']!='other':
				tmp_bbox[tmp_count][0]=label[i]['labels'][j]['box2d']['x1']
				tmp_bbox[tmp_count][1]=label[i]['labels'][j]['box2d']['y1']
				tmp_bbox[tmp_count][2]=label[i]['labels'][j]['box2d']['x2']
				tmp_bbox[tmp_count][3]=label[i]['labels'][j]['box2d']['y2']
				tmp_track[tmp_count]=label[i]['labels'][j]['id']
				if label[i]['labels'][j]['category'] in vehicle:
					label[i]['labels'][j]['category']='vehicle'
					tmp_label[tmp_count]=1
				elif label[i]['labels'][j]['category'] in person:
					label[i]['labels'][j]['category']='person'
					tmp_label[tmp_count]=2
				elif label[i]['labels'][j]['category'] in cyclist:
					label[i]['labels'][j]['category']='cyclist'
					tmp_label[tmp_count]=3
				tmp_count+=1

	else:
		tmp_label=[]
		tmp_bbox=[]
		tmp_track=[]

	# for j in range(len(label[i]['labels'])):
	# 	if label[i]['labels'][j]['category'] not in classes:
	# 		classes.append(label[i]['labels'][j]['category'])
	# if video_check!=data_t['videoName']:
	# 	os.mkdir(path+'video/'+data_t['videoName'])
	# 	video_check=data_t['videoName']
	old_path=path+'images/val/'+label[i]['name']
	new_path=path+'training/'+out_name[label[i]['videoName']]+'/'+label[i]['name'].split('-')[-1]
	# shutil.copy2(old_path,new_path)
	img=cv2.imread(new_path)
	width=img.shape[1]
	height=img.shape[0]
	data.append({
		'filename':'training/'+out_name[label[i]['videoName']]+'/'+label[i]['name'].split('-')[-1],
		'width':width,
		'height':height,
		'video_id':out_name[label[i]['videoName']],
		'frame_id':video_count[label[i]['videoName']],
		'ann':{
			'bboxes':tmp_bbox,
			'labels':tmp_label,
			'track_id':tmp_track,
			'track_id_ignore':[],
			'bboxes_ignore':[],
			
			}
	})
	print(data[-1]['filename'])
	# exit()

# classes=['other','vehicle','person','cyclist']
# id=[i for i in range(len(classes))]
# class_count=[0 for i in range(len(classes))]
# class_count=dict(zip(classes,class_count))


# for i in range(len(video_name)):
# 	# print(video_name[i])
# 	# print(video_count[video_name[i]])
# 	# print(video_class_count[video_name[i]])
# 	for j in range(len(classes)):
# 		class_count[classes[j]]+=video_class_count[video_name[i]][classes[j]]
# print(class_count)
# count=[0,0,0]
# for i in range(len(classes)):
# 	if classes[i] in vehicle:
# 		count[0]+=class_count[classes[i]]
# 	if classes[i] in person:
# 		count[1]+=class_count[classes[i]]
# 	if classes[i] in cyclist:
# 		count[2]+=class_count[classes[i]]
# print(count)
# seed=np.random.randint(61)
# test_count=[0,0,0]

jsonfile_name='bdd_all_3class.json'
with open(os.path.join(path,jsonfile_name),'w+',encoding='utf-8') as f:
	data=json.dump(data,f)


# video_name=['004ea016-0b1932a7', '003baca5-70c87fc6', '003c4a61-52588960', '004071a4-4e8a363a', '00390995-49d9a01e', '00391a82-8be5b76d', '003baca5-d6cd84e5', '00423717-0ef3c8dc', '003baca5-ad660439', '004071a4-a45d905f', '004071a4-ef4bf541', '0035afff-47378fa3', '0035afff-bd191d6a', '00378858-c5f802ac', '00391a82-d1428e56', '003baca5-aab2e274', '0035afff-572b2d4e', '003e23ee-07d32feb', '003e23ee-67d25f19', '004071a4-049b7b85', '00417c23-220bbc98', '007da0eb-8cca23d7', '007eddfc-528c4da4', '00810e80-37641274', '007aeb45-eef41701', '007c01ea-a02f29ef', '007c11bf-f6da335c', '007da0eb-e1f588e8', '007eddfc-f8a80310', '007aeb45-9330e852', '007aeb45-c601742b', '007b11e5-1033ff33', '007c01ea-63aa326c', '007c01ea-ad9f940b', '007eddfc-bcaeb35b', '007aeb45-56d1aed9', '007aeb45-f9f5ac8c', '007da0eb-1bad8468', '0080b637-55f7930e', '007aeb45-96ce245e', '007b11e5-c22ddae8', '0070bc56-49cf077c', '0070bc56-8d8cfd82', '0071d9c5-0f52d539', '0075a5b0-9a8d5dbb', '007ae77f-79995643', '0070bc56-4c0bb2d5', '0071d9c5-be7394cc', '007693e6-2535e7bf', '007693e6-c2a8b9a7', '00779058-517a4591', '00714cd3-48d6b290', '0075b179-8e09869a', '0077ccb8-d5778190', '007aeb45-3e75ce0e', '00787a90-b350f376', '0070bc56-29bcc943', '0070bc56-401304be', '0070bc56-7d01076a', '00721168-56efa5c2', '007693e6-bc55f0e4']
#video_count={'004ea016-0b1932a7': 202, '003baca5-70c87fc6': 202, '003c4a61-52588960': 203, '004071a4-4e8a363a': 202, '00390995-49d9a01e': 202, '00391a82-8be5b76d': 203, '003baca5-d6cd84e5': 203, '00423717-0ef3c8dc': 202, '003baca5-ad660439': 202, '004071a4-a45d905f': 203, '004071a4-ef4bf541': 203, '0035afff-47378fa3': 202, '0035afff-bd191d6a': 202, '00378858-c5f802ac': 202, '00391a82-d1428e56': 202, '003baca5-aab2e274': 202, '0035afff-572b2d4e': 203, '003e23ee-07d32feb': 202, '003e23ee-67d25f19': 202, '004071a4-049b7b85': 202, '00417c23-220bbc98': 203, '007da0eb-8cca23d7': 203, '007eddfc-528c4da4': 202, '00810e80-37641274': 202, '007aeb45-eef41701': 202, '007c01ea-a02f29ef': 197, '007c11bf-f6da335c': 202, '007da0eb-e1f588e8': 202, '007eddfc-f8a80310': 202, '007aeb45-9330e852': 202, '007aeb45-c601742b': 212, '007b11e5-1033ff33': 203, '007c01ea-63aa326c': 202, '007c01ea-ad9f940b': 103, '007eddfc-bcaeb35b': 202, '007aeb45-56d1aed9': 202, '007aeb45-f9f5ac8c': 202, '007da0eb-1bad8468': 202, '0080b637-55f7930e': 202, '007aeb45-96ce245e': 202, '007b11e5-c22ddae8': 202, '0070bc56-49cf077c': 202, '0070bc56-8d8cfd82': 102, '0071d9c5-0f52d539': 202, '0075a5b0-9a8d5dbb': 102, '007ae77f-79995643': 203, '0070bc56-4c0bb2d5': 102, '0071d9c5-be7394cc': 202, '007693e6-2535e7bf': 202, '007693e6-c2a8b9a7': 197, '00779058-517a4591': 202, '00714cd3-48d6b290': 202, '0075b179-8e09869a': 202, '0077ccb8-d5778190': 202, '007aeb45-3e75ce0e': 203, '00787a90-b350f376': 202, '0070bc56-29bcc943': 102, '0070bc56-401304be': 103, '0070bc56-7d01076a': 201, '00721168-56efa5c2': 202, '007693e6-bc55f0e4': 202}