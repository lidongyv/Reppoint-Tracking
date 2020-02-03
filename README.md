run ./mmdetection/tools/eval_demo.py
get detection result xx.json in ./result
copy the json to /backdata1/KITTI/kitti/tracking/
run warp.py
get tracking result xx.json in /backdata1/KITTI/kitti/tracking/xx.json
visualize the result by 
/home/ld/RepPoints/MOT_VIS_JSON_second/run.sh
convert the image to video by 
/home/ld/RepPoints/img_to_video.py