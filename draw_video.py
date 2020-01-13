import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import ffmpeg

path='/home/ld/RepPoints/debug/feature_change/1'

# (
#     ffmpeg
#     .input(os.path.join(path,'resnet/*.jpg'), pattern_type='glob', framerate=10)
#     .output(os.path.join(path,'resnet.mp4'))
#     .run()
# )
# (
#     ffmpeg
#     .input(os.path.join(path,'stsn_r/*.jpg'), pattern_type='glob', framerate=10)
#     .output(os.path.join(path,'stsn_r.mp4'))
#     .run()
# )
# (
#     ffmpeg
#     .input(os.path.join(path,'stsn_s/*.jpg'), pattern_type='glob', framerate=10)
#     .output(os.path.join(path,'stsn_s.mp4'))
#     .run()
# )
# (
#     ffmpeg
#     .input(os.path.join(path,'init_rep/*.jpg'), pattern_type='glob', framerate=10)
#     .output(os.path.join(path,'init_rep.mp4'))
#     .run()
# )
# (
#     ffmpeg
#     .input(os.path.join(path,'refine_rep/*.jpg'), pattern_type='glob', framerate=10)
#     .output(os.path.join(path,'refine_rep.mp4'))
#     .run()
# )
(
    ffmpeg
    .input(os.path.join(path,'agg_f/*.jpg'), pattern_type='glob', framerate=10)
    .output(os.path.join(path,'agg_f.mp4'))
    .run()
)
(
    ffmpeg
    .input(os.path.join(path,'support_f/*.jpg'), pattern_type='glob', framerate=10)
    .output(os.path.join(path,'support_f.mp4'))
    .run()
)
