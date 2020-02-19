# -*- coding: utf-8 -*-
# @Author: Lidong Yu
# @Date: 2019-12-11 16:41:28
# @Last Modified by: Lidong Yu
# @Last Modified time: 2019-12-11 16:41:28

import matplotlib.pyplot as plt
import numpy as np
import cv2
import mmcv
import matplotlib
import os
import shutil
import glob


def kernel_inv_map(vis_attr, target_point, map_h, map_w):
    pos_shift = [vis_attr['dilation'] * 0 - vis_attr['pad'],
                 vis_attr['dilation'] * 1 - vis_attr['pad'],
                 vis_attr['dilation'] * 2 - vis_attr['pad']]
    source_point = []
    for idx in range(vis_attr['filter_size'] ** 2):
        cur_source_point = np.array([target_point[0] + pos_shift[idx // 3],
                                     target_point[1] + pos_shift[idx % 3]])
        if cur_source_point[0] < 0 or cur_source_point[1] < 0 \
                or cur_source_point[0] > map_h - 1 or cur_source_point[1] > map_w - 1:
            continue
        source_point.append(cur_source_point.astype('f'))
    return source_point


def offset_inv_map(source_points, offset):
    # print(source_points,offset)
    for idx, _ in enumerate(source_points):
        source_points[idx][0] += offset[2 * idx]
        source_points[idx][1] += offset[2 * idx + 1]
    return source_points


def get_bottom_position(vis_attr, top_points, all_offset):
    map_h = all_offset[0].shape[2]
    map_w = all_offset[0].shape[3]
    # level 这个作用是什么？
    for level in range(vis_attr['plot_level']):
        source_points = []
        for idx, cur_top_point in enumerate(top_points):
            cur_top_point = np.round(cur_top_point)
            if cur_top_point[0] < 0 or cur_top_point[1] < 0 \
                    or cur_top_point[0] > map_h - 1 or cur_top_point[1] > map_w - 1:
                continue
            cur_source_point = kernel_inv_map(vis_attr, cur_top_point, map_h, map_w)
            cur_offset = np.squeeze(all_offset[level][:, :, int(cur_top_point[0]), int(cur_top_point[1])])
            # print(cur_offset)
            cur_source_point = offset_inv_map(cur_source_point, cur_offset)
            source_points = source_points + cur_source_point
        top_points = source_points
    return source_points


def plot_according_to_point(vis_attr, im, source_points, map_h, map_w, loc_offset_scale, color=[0, 0, 255]):
    plot_area = vis_attr['plot_area']
    for idx, cur_source_point in enumerate(source_points):
        # y = np.round((cur_source_point[0] + 0.5) * im.shape[0] / map_h).astype('i')
        # x = np.round((cur_source_point[1] + 0.5) * im.shape[1] / map_w).astype('i')

        y = np.round((cur_source_point[0] + 0.5) * 8 * loc_offset_scale).astype('i')
        x = np.round((cur_source_point[1] + 0.5) * 8 * loc_offset_scale).astype('i')

        if x < 0 or y < 0 or x > im.shape[1] - 1 or y > im.shape[0] - 1:
            continue
        y = min(y, im.shape[0] - vis_attr['plot_area'] - 1)
        x = min(x, im.shape[1] - vis_attr['plot_area'] - 1)
        y = max(y, vis_attr['plot_area'])
        x = max(x, vis_attr['plot_area'])
        # print('75 debug')
        # from IPython import embed
        # embed()
        im[y - plot_area:y + plot_area + 1, x - plot_area:x + plot_area + 1, :] = np.tile(
            np.reshape(color, (1, 1, 3)), (2 * plot_area + 1, 2 * plot_area + 1, 1)
        )
    return im


def show_dconv_offset_by_loc(im, all_offset, path, loc, loc_offset_scale, step=[2, 2], filter_size=3,
                             dilation=1, pad=1, plot_area=1, plot_level=1, stride=8, refer=None, ):
    #这个好像卷积时候的一些参数
    vis_attr = {'filter_size': filter_size, 'dilation': dilation, 'pad': pad,
                'plot_area': plot_area, 'plot_level': plot_level, 'stride': stride}
    # print(all_offset[0])

    map_h = all_offset[0].shape[2]  #48
    map_w = all_offset[0].shape[3]  #156
    scale = 1

    # fig = plt.figure(figsize=(10, 3))  #为什么是10，3
    count = 0
    for (im_w, im_h) in loc:

        source_y = im_h  #227
        source_x = im_w  #33
        # 这行代码的意义是什么？
        # (im.shape[1] / map_w)  可以算出来缩放的比例
        # 原始图像的高度  / 比例 可以得到在feature map上的位置
        # im_w = (im_w /loc_offset_scale) / (1280 / map_w)
        # im_h = (im_h/loc_offset_scale) / (736 / map_h)
        im_w = (im_w /loc_offset_scale) / 8
        im_h = (im_h /loc_offset_scale) / 8
        # im_w = im_w / (im.shape[1] / map_w)  #im_w:33 im.shape[1]:1242  map_w:156
        # im_h = im_h / (im.shape[0] / map_h)  #im_h:227 im.shape[1]:374  map_h:48
        target_point = np.array([im_h, im_w]).astype(np.int)  #[3,0] 计算这个的目的是什么？
        cur_im = np.copy(im)
        # plt.imshow(cur_im)
        # fig.savefig(path, dpi=150)

        # 1280  736
        source_points = get_bottom_position(vis_attr, [target_point], all_offset)
        source_points = [each for each in source_points]



        cur_im = plot_according_to_point(vis_attr, cur_im, source_points, map_h, map_w, loc_offset_scale)
        # plt.imshow(cur_im)
        # fig.savefig(path + '1.jpg', dpi=150)

        source_y_back = int(source_y )
        source_x_back = int(source_x )
        if source_y_back < plot_area or source_x_back < plot_area \
                or source_y_back >= im.shape[0] - plot_area or source_x_back >= im.shape[1] - plot_area:
            print('out of image')
            continue

        cur_im[source_y - plot_area:source_y + plot_area + 1, source_x - plot_area:source_x + plot_area + 1, :] = \
            np.tile(np.reshape([0, 255, 255], (1, 1, 3)), (2 * plot_area + 1, 2 * plot_area + 1, 1))

        im = np.copy(cur_im)
        # plt.imshow(cur_im)
        # fig.savefig(path, dpi=150)
    print('showing', im_h, im_w)
    # print('139 debug')
    # from IPython import embed
    # embed()
    cv2.imwrite(path, cur_im)
    # plt.axis("off")
    # plt.imshow(cur_im)
    #
    # fig.savefig(path, dpi=150)
    # plt.clf()
    # plt.close('all')


if __name__ == '__main__':
    # torch.Size([2, 256, 48, 156])
    # torch.Size([2, 256, 24, 78])
    # torch.Size([2, 256, 12, 39])
    # torch.Size([2, 256, 6, 20])
    # torch.Size([2, 256, 3, 10])
    #加载某一个 Reppoints
    reppoints = mmcv.load(
        '/home/zxn/RepPoints/zxn_result/debug/repoint_baseline_do3/kittibddwayo_epoch20_thres03_nms05_baseline_rerun_02142013_sub13/baseline/reppoints_13_all.pkl')
    #这个path是用来干什么的？
    path = '/home/zxn/RepPoints/zxn_result/debug/repoint_baseline_do3/kittibddwayo_epoch20_thres03_nms05_baseline_rerun_02142013_sub13/baseline'
    classes = ['Car', 'Person', 'Cyclist']
    split = ['not_detected', 'wrong_detected'] # 这个是在什么时候弄好的？
    # split = ['wrong_detected'] # 这个是在什么时候弄好的？

    # print('140 debug')
    # from IPython import embed
    # embed()
    for i in range(len(classes)):
        class_path = os.path.join(path, classes[i])
        video_name = os.listdir(class_path)
        video_name.sort()
        for j in range(len(video_name)):
            video_path = os.path.join(class_path, video_name[j])
            for m in range(len(split)):
                video_split = os.path.join(video_path, split[m])

                pkls = glob.glob(video_split + '/*.pkl')
                pkls.sort()
                # 这里是对几个文件进行循环
                # print('155 debug')
                # from IPython import embed
                # embed()
                for n in range(len(pkls)):
                    pkl_path = os.path.join(video_split, pkls[n]) #给出pkl的路径
                    # 这个是事先筛选好 比较适合用于分析的case么？
                    loc, index, name = mmcv.load(pkl_path) #加载进来，pkl里面应该包含了用于绘制offset所用的信息

                    # print(len(reppoints))
                    # print(len(reppoints[0]))
                    # print(reppoints[0][index].shape)
                    # len(reppoints[0]) = 5 应该是5个分辨率下的
                    # print('167 debug')
                    # from IPython import embed
                    # embed()
                    reppoint_frame = reppoints[0][index][0]  #提取出来的shape是：1 * 18 * 48 * 156  这个像是某一层的特征
                    img = cv2.imread(name)
                    # img = matplotlib.image.imread(name)  # matplotlib 读取进来图像之后，就是0到1之间的
                    loc_offset_scale_sets = [0.97, 1, 1.7777]
                    # target size = 736
                    if name.split('/')[-3].split('_')[0] == 'kitti':
                        loc_offset_scale = loc_offset_scale_sets[0]
                    #     print('176 debug')
                    #     from IPython import embed
                    #     embed()
                    #     img, scale_factor = mmcv.imrescale(img, (1280, 720), return_scale=True)
                    #     temp_img = np.zeros([736, 1280, 3])
                    #     temp_img[:img.shape[0], :, :] = img
                    #     img = temp_img
                    if name.split('/')[-3].split('_')[0] == 'bdd':
                        loc_offset_scale = loc_offset_scale_sets[1]
                    #     img, scale_factor = mmcv.imrescale(img, (1280, 720), return_scale=True)
                    #     temp_img = np.zeros([736, 1280, 3])
                    #     temp_img[:img.shape[0], :, :] = img
                    if name.split('/')[-3].split('_')[0] == 'waymo':
                        loc_offset_scale = loc_offset_scale_sets[2]
                    #     img, scale_factor = mmcv.imrescale(img, (1280, 720), return_scale=True)
                    #     temp_img = np.zeros([736, 1280, 3])
                    #     temp_img[:img.shape[0], :img.shape[1], :] = img

                    # img = cv2.resize(img)  #对图像进行缩放一下
                    loc = [np.array(p).astype(np.int)[:2] for p in loc] #这里的loc是如何获取的？ 将其他类型的转换为int型的
                    # loc=[loc.astype(np.int)[:2]]

                    scale = [8, 16, 32, 64, 128]

                    name = name.split('/')
                    offset_path = video_split + '/offset'
                    if not os.path.exists(offset_path):
                        os.mkdir(offset_path)
                    offset_path = os.path.join(offset_path, name[-1])
                    loc = [(each_loc * loc_offset_scale).astype(np.int) for each_loc in loc]
                    show_dconv_offset_by_loc(img, [reppoint_frame],  offset_path, loc, loc_offset_scale, plot_level=1, plot_area=3)
            # exit()
