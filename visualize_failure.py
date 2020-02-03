import os
import shutil
path='/home/ld/RepPoints/analyze/fuse_c_result/epoch9_thres0.1_nms0.3_with10/baseline'
videos=os.listdir(path)
if not os.path.exists(os.path.join(path,'all')):
	os.mkdir(os.path.join(path,'all'))
for i in range(len(videos)):
	images=os.listdir(os.path.join(path,videos[i]))
	for j in range(len(images)):
		src_path=os.path.join(path,videos[i],images[j])
		det_path=os.path.join(path,'all',videos[i]+'_'+images[j])
		shutil.copy2(src_path,det_path)
		print(det_path)

	