# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-12-11 16:41:28  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-12-11 16:41:28

import matplotlib.pyplot as plt
import numpy as np
import cv2
from mmcv.image import imread, imwrite
from mmcv.utils import is_str
from enum import Enum
import mmcv
import matplotlib
import os
import shutil
import json
import torch
def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    cv2.waitKey(wait_time)
class Color(Enum):
    """An enum that defines common colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def color_val(color):
    """Convert various input to color tuples.

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if is_str(color):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert channel >= 0 and channel <= 255
        return color
    elif isinstance(color, int):
        assert color >= 0 and color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError('Invalid type for color: {}'.format(type(color)))

def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    # assert bboxes.ndim == 2
    # assert labels.ndim == 1
    # assert bboxes.shape[0] == labels.shape[0]
    # assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    # print(labels.shape)
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)

def show_result(img,
                result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                show=True,
                out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    # print(len(result))
    bbox_result = result
    bboxes = np.vstack(bbox_result)
    # print(bboxes.shape)
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # print(labels.shape)
    imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=class_names,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    if not (show or out_file):
        return img


if __name__=='__main__':
    # torch.Size([2, 256, 48, 156])
    # torch.Size([2, 256, 24, 78])
    # torch.Size([2, 256, 12, 39])
    # torch.Size([2, 256, 6, 20])
    # torch.Size([2, 256, 3, 10])
    data=mmcv.load('/home/ld/RepPoints/temp_for_ransac_score0.3/temp_5.json')
    loc_record=mmcv.load('/home/ld/RepPoints/temp_for_ransac_score0.3/loc_result_5_all.pkl')
    img1=cv2.imread(data[2]['filename'])
    img2=cv2.imread(data[2]['filename'])
    show_img=img1+0
    show_img2=img2+0
    print(img1.shape)
    loc1=torch.from_numpy(np.vstack(loc_record[1]))
    loc2=torch.from_numpy(np.vstack(loc_record[2]))
    offset1_ori=torch.from_numpy(mmcv.load('./offset1_ori.pkl'))
    offset2_ori=torch.from_numpy(mmcv.load('./offset2_ori.pkl'))
    offset1_sample=torch.from_numpy(mmcv.load('./offset1_sample.pkl'))
    offset2_sample=torch.from_numpy(mmcv.load('./offset2_sample.pkl'))
    ransac=torch.from_numpy(mmcv.load('./ransac2.pkl'))
    correspondense=torch.from_numpy(mmcv.load('./correspondense2.pkl'))
    det_record=mmcv.load('./det_result_5_all.pkl')

    scale=[8]
    det1=det_record[1]
    det2=det_record[2]
    ps=8
    classes=['car','person']
    if not os.path.exists('./visualize'):
        os.mkdir('./visualize')
    out_path='./visualize'

    for m in range(len(loc1)):
        v_im=show_img+0
        
        x=(loc1[m,1]*1.78).long().clamp(min=ps+1,max=1280-ps-1)
        y=(loc1[m,0]*1.78).long().clamp(min=ps+1,max=1920-ps-1)

        v_im[x-ps:x+ps+1, y-ps:y+ps+1, :] = \
            np.tile(np.reshape([0, 255, 255], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
        v_im_center=v_im+0
        index_x=(loc1[m,1]//8).long().clamp(min=0,max=96)
        index_y=(loc1[m,0]//8).long().clamp(min=0,max=160)
        for n in range(9):
            bias=offset1_ori[0,2*n:2*n+2,index_x,index_y]*8*1.78
            bias=np.floor(bias).long()
            x=(loc1[m,1]*1.78+bias[1]).clamp(min=ps+1,max=1280-ps-1).int()
            y=(loc1[m,0]*1.78+bias[0]).clamp(min=ps+1,max=1920-ps-1).int()
            
            v_im[x-1:x+10, y,:]
            v_im[x-ps:x+ps+1, y-ps:y+ps+1,:] = \
                np.tile(np.reshape([0, 0, 255], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
        x=(loc1[m,1]*1.78).long().clamp(min=ps+1,max=1280-ps-1)
        y=(loc1[m,0]*1.78).long().clamp(min=ps+1,max=1920-ps-1)

        v_im[x-ps:x+ps+1, y-ps:y+ps+1, :] = \
            np.tile(np.reshape([0, 255, 255], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
        v_im_refer=v_im+0
        v_im=v_im_center+0
        for n in range(18):
            bias=offset1_sample[0,2*n:2*n+2,index_x,index_y]*8*1.78
            bias=np.floor(bias).long()
            x=(loc1[m,1]*1.78+bias[1]).clamp(min=ps+1,max=1280-ps-1).int()
            y=(loc1[m,0]*1.78+bias[0]).clamp(min=ps+1,max=1920-ps-1).int()
            
            v_im[x-1:x+10, y,:]
            v_im[x-ps:x+ps+1, y-ps:y+ps+1,:] = \
                np.tile(np.reshape([255, 0, 255], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
        x=(loc1[m,1]*1.78).long().clamp(min=ps+1,max=1280-ps-1)
        y=(loc1[m,0]*1.78).long().clamp(min=ps+1,max=1920-ps-1)

        v_im[x-ps:x+ps+1, y-ps:y+ps+1, :] = \
            np.tile(np.reshape([0, 255, 255], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
        v_im_sample=v_im+0
        v_im=v_im_center+0
        for n in range(18):
            bias=ransac[0,2*n:2*n+2,index_x,index_y]*8*1.78
            bias=np.floor(bias).long()
            x=(loc1[m,1]*1.78+bias[1]).clamp(min=ps+1,max=1280-ps-1).int()
            y=(loc1[m,0]*1.78+bias[0]).clamp(min=ps+1,max=1920-ps-1).int()
            
            v_im[x-1:x+10, y,:]
            v_im[x-ps:x+ps+1, y-ps:y+ps+1,:] = \
                np.tile(np.reshape([0, 255, 0], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
        x=(loc1[m,1]*1.78).long().clamp(min=ps+1,max=1280-ps-1)
        y=(loc1[m,0]*1.78).long().clamp(min=ps+1,max=1920-ps-1)

        v_im[x-ps:x+ps+1, y-ps:y+ps+1, :] = \
            np.tile(np.reshape([0, 255, 255], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
        v_im_ransac=v_im+0
        v_im=v_im_center+0
        for n in range(18):
            bias=correspondense[0,2*n:2*n+2,index_x,index_y]*8*1.78
            bias=np.floor(bias).long()
            x=(loc1[m,1]*1.78+bias[1]).clamp(min=ps+1,max=1280-ps-1).int()
            y=(loc1[m,0]*1.78+bias[0]).clamp(min=ps+1,max=1920-ps-1).int()
            
            v_im[x-1:x+10, y,:]
            v_im[x-ps:x+ps+1, y-ps:y+ps+1,:] = \
                np.tile(np.reshape([0, 255, 0], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
        x=(loc1[m,1]*1.78).long().clamp(min=ps+1,max=1280-ps-1)
        y=(loc1[m,0]*1.78).long().clamp(min=ps+1,max=1920-ps-1)

        v_im[x-ps:x+ps+1, y-ps:y+ps+1, :] = \
            np.tile(np.reshape([0, 255, 255], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
        v_im_correspondense=v_im+0
        v_im=v_im_center+0
        if not os.path.exists(os.path.join(out_path,'refer')):
            os.mkdir(os.path.join(out_path,'refer'))
        show_result(v_im_refer, det1, classes, show=False,out_file=os.path.join(out_path,'refer',str(m)+data[1]['filename'].split('/')[-1]))
        if not os.path.exists(os.path.join(out_path,'sample')):
            os.mkdir(os.path.join(out_path,'sample'))
        show_result(v_im_sample, det1, classes, show=False,out_file=os.path.join(out_path,'sample',str(m)+data[1]['filename'].split('/')[-1]))
        if not os.path.exists(os.path.join(out_path,'ransac')):
            os.mkdir(os.path.join(out_path,'ransac'))
        show_result(v_im_ransac, det1, classes, show=False,out_file=os.path.join(out_path,'ransac',str(m)+data[1]['filename'].split('/')[-1]))
        if not os.path.exists(os.path.join(out_path,'corr')):
            os.mkdir(os.path.join(out_path,'corr'))
        show_result(v_im_correspondense, det1, classes, show=False,out_file=os.path.join(out_path,'corr',str(m)+data[1]['filename'].split('/')[-1]))
    for m in range(len(loc2)):
        v_im=show_img2+0
        
        x=(loc2[m,1]*1.78).long().clamp(min=ps+1,max=1280-ps-1)
        y=(loc2[m,0]*1.78).long().clamp(min=ps+1,max=1920-ps-1)

        v_im[x-ps:x+ps+1, y-ps:y+ps+1, :] = \
            np.tile(np.reshape([0, 255, 255], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
        v_im_center=v_im+0
        index_x=(loc2[m,1]//8).long().clamp(min=0,max=96)
        index_y=(loc2[m,0]//8).long().clamp(min=0,max=160)
        for n in range(9):
            bias=offset2_ori[0,2*n:2*n+2,index_x,index_y]*8*1.78
            bias=np.floor(bias).long()
            x=(loc2[m,1]*1.78+bias[1]).clamp(min=ps+1,max=1280-ps-1).int()
            y=(loc2[m,0]*1.78+bias[0]).clamp(min=ps+1,max=1920-ps-1).int()
            
            v_im[x-1:x+10, y,:]
            v_im[x-ps:x+ps+1, y-ps:y+ps+1,:] = \
                np.tile(np.reshape([0, 0, 255], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
        v_im_refer2=v_im+0
        v_im=v_im_center+0
        for n in range(18):
            bias=offset2_sample[0,2*n:2*n+2,index_x,index_y]*8*1.78
            bias=np.floor(bias).long()
            x=(loc2[m,1]*1.78+bias[1]).clamp(min=ps+1,max=1280-ps-1).int()
            y=(loc2[m,0]*1.78+bias[0]).clamp(min=ps+1,max=1920-ps-1).int()
            
            v_im[x-1:x+10, y,:]
            v_im[x-ps:x+ps+1, y-ps:y+ps+1,:] = \
                np.tile(np.reshape([255, 0, 255], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
        v_im_sample2=v_im+0

        if not os.path.exists(os.path.join(out_path,'refer2')):
            os.mkdir(os.path.join(out_path,'refer2'))
        show_result(v_im_refer2, det1, classes, show=False,out_file=os.path.join(out_path,'refer2',str(m)+data[1]['filename'].split('/')[-1]))
        if not os.path.exists(os.path.join(out_path,'sample2')):
            os.mkdir(os.path.join(out_path,'sample2'))
        show_result(v_im_sample2, det1, classes, show=False,out_file=os.path.join(out_path,'sample2',str(m)+data[1]['filename'].split('/')[-1]))
        show_result(v_im_refer2, det1, classes, show=False,out_file=os.path.join(out_path,'refer2',str(m)+data[1]['filename'].split('/')[-1]))

       