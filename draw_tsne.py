import pickle
import os
import numpy as np
import cv2
import argparse
from fire import Fire
from matplotlib import pyplot as plt
from tsnecuda import TSNE
from tqdm import tqdm

from IPython import embed


def main():
    parser = argparse.ArgumentParser(description='draw feature tsne results')
    parser.add_argument('--video_name', '-vn', dest='video_name')
    args = parser.parse_args()

    root_path = '/home/jn/codes/radmin234/script_python/disentangle/Siamese-RPN/test_results/'+args.video_name
    save_path = root_path + '/tsne_results/'
    box_path = root_path + '/draw_boxes/'
    pos_heat_path = root_path + '/heat_imgs/pos/'
    f_feature_path = root_path + '/f-features/'
    b_feature_path = root_path + '/b-features/'
    #neg_heat_path = root_path + '/heat_imgs/neg/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # pos_path = root_path + '/save_features/pos_features/'
    # neg_path = root_path + '/save_features/neg_features/'
    # middle_path = root_path + '/save_features/middle_features/'
    # tem_path = root_path + '/save_features/tem_features/'
    # #filenames = os.listdir(box_gt_path)
    # posnames = os.listdir(pos_path)
    # negnames = os.listdir(neg_path)
    # temnames = os.listdir(tem_path)
    # middlenames = os.listdir(middle_path)
    # # filenames.sort(key=lambda x:int(x[4:-4]))
    # # print(np.array(filenames[:10]))
    # pos_feature_all = []
    # neg_feature_all = []
    # middle_feature_all = []
    # feature_result_all = []
    # tem_feature = pickle.load(open(tem_path+'template.pth', 'rb'))
    # feature_result_all += tem_feature.tolist()[0]
    # #embed()
    # for idx in range(1,len(posnames)+1):
    #     pos_feature = pickle.load(open(pos_path+'pos_{}.pth'.format(idx), 'rb'))
    #     pos_feature = pos_feature.reshape(-1)
    #     pos_feature_all += pos_feature.tolist()#[:256]
    #     # print(idx,len(pos_feature_all)/256)
    #     # pos_feature_all.append(pos_feature)
    # feature_result_all.extend(pos_feature_all)
    # for idx in range(1,len(negnames)+1):
    #     neg_feature = pickle.load(open(neg_path+'neg_{}.pth'.format(idx), 'rb'))
    #     neg_feature = neg_feature.reshape(-1)
    #     neg_feature_all += neg_feature.tolist()
    #     # neg_feature_all.append(neg_feature)
    # feature_result_all.extend(neg_feature_all)
    # for idx in range(1,len(middlenames)+1):
    #     middle_feature = pickle.load(open(middle_path+'middle_{}.pth'.format(idx), 'rb'))
    #     middle_feature = middle_feature.reshape(-1)
    #     middle_feature_all += middle_feature.tolist()
    #     # neg_feature_all.append(neg_feature)
    # feature_result_all.extend(middle_feature_all)
    # feature_result_all = np.array(feature_result_all).reshape(-1,256)
    #
    # point2D = TSNE().fit_transform(feature_result_all)
    # x_min = point2D[:, 0].min()
    # x_max = point2D[:, 0].max()
    # y_min = point2D[:, 1].min()
    # y_max = point2D[:, 1].max()
    # point2D_normal = (point2D - [x_min, y_min]) / [x_max - x_min, y_max - y_min]
    # color_pos = [[[0, 0, x]] for x in np.array(range(len(posnames))) / len(posnames) * 1]
    # color_neg = [[[1, 1, x]] for x in np.array(range(len(negnames))) / len(negnames) * 1]
    #embed()
    # for idx in tqdm(np.array(range(len(posnames))) + 1):
    #     fig = plt.figure(figsize=(8, 8))
    #     ax = fig.add_subplot(221)
    #     #plt.figure()
    #     plt.xlim((0, 1))
    #     plt.ylim((0, 1))
    #
    #     for i in range(idx):
    #         plt.scatter(point2D_normal[0, 0], point2D_normal[0, 1], c='r')
    #         plt.scatter(point2D_normal[i + 1, 0], point2D_normal[i+1, 1], c='m',marker='*')
    #         plt.scatter(point2D_normal[i + 1 + len(posnames) - 1, 0],
    #                     point2D_normal[i + 1 + len(posnames) - 1, 1], c='c',marker='x')
    #         plt.scatter(point2D_normal[i + 1 + 2*len(posnames)-1, 0],
    #                     point2D_normal[i + 1 + 2*len(posnames)-1, 1], c='y',marker='^')
    #     box_img = cv2.imread(box_path+str(idx)+'.jpg')
    #     pos_heat_img = cv2.imread(pos_heat_path+'posheatmap_{}.jpg'.format(idx))
    #     neg_heat_img = cv2.imread(neg_heat_path + 'negheatmap_{}.jpg'.format(idx))
    #     bx = fig.add_subplot(222)
    #     plt.imshow(box_img[:, :, ::-1])
    #     dx= fig.add_subplot(223)
    #     plt.imshow(pos_heat_img[:, :, ::-1])
    #     cx = fig.add_subplot(224)
    #     plt.imshow(neg_heat_img[:, :, ::-1])
    #
    #     plt.savefig(save_path + str(idx) + '.jpg')
    # plt.close()
    namelist = os.listdir(f_feature_path)
    for i in range(1,len(namelist)+1):
        box_img = cv2.imread(box_path+str(i)+'.jpg')
        pos_heat_img = cv2.imread(pos_heat_path+'posheatmap_{}.jpg'.format(i))
        f_feature_img = cv2.imread(f_feature_path+'fheatmap{}.jpg'.format(i))
        b_feature_img = cv2.imread(b_feature_path + 'bheatmap{}.jpg'.format(i))
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(221)
        plt.imshow(box_img[:,:,::-1])
        bx = fig.add_subplot(222)
        plt.imshow(pos_heat_img[:,:,::-1])
        cx = fig.add_subplot(223)
        plt.imshow(f_feature_img[:,:,::-1])
        dx = fig.add_subplot(224)
        plt.imshow(b_feature_img[:, :, ::-1])
        plt.savefig(save_path+str(i)+'.jpg')
    plt.close()

if __name__ == '__main__':
    Fire(main)
