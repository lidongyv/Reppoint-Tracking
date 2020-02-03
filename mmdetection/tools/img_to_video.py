import ffmpeg
import os
out_path='/home/ld/RepPoints/MOT_VIS_JSON_second/result/vis/'
video_name=os.listdir(out_path)
for i in range(len(video_name)):
	video_path=os.path.join(out_path,video_name[i])
	(
		ffmpeg
		.input(os.path.join(video_path,'/*.png'), pattern_type='glob', framerate=10)
		.output(os.path.join(out_path,video_name[i]+'.mp4'))
		.run()
	)