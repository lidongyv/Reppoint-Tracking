# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-11-25 19:24:06  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-11-25 19:24:06

from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import time
import ffmpeg
#config_file = '/home/ld/RepPoints/mmdetection/configs/retinanet_r101_fpn_1x.py'
#config_file ='/home/ld/RepPoints/configs/reppoints_minmax_r50_fpn_1x.py'
#checkpoint_file = '/home/ld/RepPoints/trained/retinanet_r101_fpn_1x_20181129-f016f384.pth'
#checkpoint_file='/home/ld/RepPoints/trained/reppoints_minmax_r50_fpn_1x.pth'
config_file ='/home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_kitti.py'
checkpoint_file='/home/ld/RepPoints/work_dirs/reppoints_moment_r101_dcn_fpn_kitti/epoch_50.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')



# test a single image and show the results

# img_name = '/home/ld/RepPoints/kitti/0000/000000.png'  # or img = mmcv.imread(img), which will only load it once
# img = mmcv.imread(img_name)
# result = inference_detector(model, img)
# # visualize the results in a new window
# # show_result(img, result, model.CLASSES)
# # or save the visualization results to image files
# out_path='/home/ld/RepPoints/out/0000/'
# out_name=out_path+img_name.split('/')[-1]
# show_result(img, result, model.CLASSES, out_file=out_name,show=False)
# print(out_name)


# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
input_dir='/home/ld/RepPoints/kitti/0000/'
sequence=os.listdir(input_dir)
sequence.sort()
out_path='/home/ld/RepPoints/out/retinanet/0000/'
compute_time=0
for frame in sequence:
	start=time.time()
	img_name=input_dir+frame
	print(img_name)
	img = mmcv.imread(img_name)
	result = inference_detector(model, img)
	print(time.time()-start)
	compute_time+=time.time()-start
	out_name=out_path+frame
	show_result(img, result, model.CLASSES, show=True,out_file=out_name)
print('compute_time:',compute_time/len(sequence))

out_path='/home/ld/RepPoints/out/retinanet/0000'
(
    ffmpeg
    .input(os.path.join(out_path+'/*.png'), pattern_type='glob', framerate=10)
    .output(os.path.join(out_path+'.mp4'))
    .run()
)