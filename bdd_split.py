import os
import numpy as np
import json
import shutil
import cv2

classes=['other','vehicle','person','cyclist']
label=[i for i in range(len(classes))]
class_count=[0 for i in range(len(classes))]
class_count=dict(zip(classes,class_count))
path='/backdata01/bdd/'
jsonfile_name='bdd_all_3class.json'
with open(os.path.join(path,jsonfile_name),'r',encoding='utf-8') as f:
	data=json.load(f)
video_name=os.listdir(os.path.join(path,'training'))
video_name.sort()
# video_count=[]
# for i in range(len(video_name)):
# 	video_count.append(class_count.copy())
# video_count=dict(zip(video_name,video_count))
# for i in range(len(data)):
# 	for j in range(len(data[i]['ann']['labels'])):
# 		video_count[data[i]['video_id']][classes[data[i]['ann']['labels'][j]]]+=1

# test=['vehicle':30000, 'person':3000, 'cyclist':300]
# test_count=[0,0,0]
# seed=np.random.randint(61)
# test_video=[seed]
# while(test_count[0]<30000 or test_count[1]<2500 or test_count[2]<200 ):
# 	test_count[0]+=video_count[video_name[seed]][classes[1]]
# 	test_count[1]+=video_count[video_name[seed]][classes[2]]
# 	test_count[2]+=video_count[video_name[seed]][classes[3]]
# 	seed=np.random.randint(61)
# 	while(seed in test_video):
# 		seed=np.random.randint(61)
# 	test_video.append(seed)
# 	print(test_count)
# 	print(test_video)
# print(test_count)
# print(test_video)

test_count=[30112, 3297, 308]
test_video=[4, 48, 56, 17, 12, 39, 31, 60, 11, 36, 15, 10, 45, 42, 0, 9, 25, 52, 41]

test_name=[]
for i in range(len(test_video)):
	test_name.append(video_name[test_video[i]])
# print(test_name)
# exit()
train=[]
test=[]

for i in range(len(data)):
	video=data[i]['video_id']
	if video in test_name:
		test.append(data[i])
	else:
		train.append(data[i])
jsonfile_name='bdd_val_3class.json'
with open(os.path.join(path,jsonfile_name),'w+',encoding='utf-8') as f:
	test=json.dump(test,f)
jsonfile_name='bdd_train_3class.json'
with open(os.path.join(path,jsonfile_name),'w+',encoding='utf-8') as f:
	train=json.dump(train,f)