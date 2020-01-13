import os.path as osp

import mmcv
import numpy as np
import torch
from torch.utils.data import Dataset
from IPython import embed

from .pipelines import Compose, FlowAug
from .registry import DATASETS

@DATASETS.register_module
class VideoCustomDataset(Dataset):
    """
    Custom dataset for video detection/tracking

    Annotation format:
    [
        {
            'video_name': '0000',
            'img_size': (370, 1224),
            'flow_size': (370, 1224),
            'quantization': False,
            'frames': [
                {
                    'filename': 'training/image_02/0000/000000.png',
                    'flow_name': 'training/Flow/0000/000000.flo',
                    'inv_flow_name': 'training/Inv_Flow/0000/000000.flo',
                    'is_annotated': True,
                    'intrinsic': (optional),
                    'ann': {
                        'bboxes': (n, 4),
                        'labels': (n, ),
                        'track_id': (n, ),
                        ...
                    }
                },
                ...
            ]
        },
        ...
    ]
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 seq_len=5,
                 padding=False,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,):
        super(VideoCustomDataset, self).__init__()
        self.seq_len = seq_len
        self.padding = padding
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations
        self.video_infos = self.load_annotations(self.ann_file)
        self.sample_list = []
        for vid, video_info in enumerate(self.video_infos):
            n_frames = len(video_info['frames'])
            if self.padding:
                for i in range(n_frames):
                    self.sample_list.append((vid, i))
            else:
                for i in range(n_frames - self.seq_len + 1):
                    self.sample_list.append((vid, i))
        if not self.test_mode:
            self.flag = np.zeros(len(self), dtype=np.uint8)
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.perpare_clip_test(idx)
        while True:
            data = self.prepare_clip_train(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _rand_another(self, idx):
        pool = range(len(self))
        ret = np.random.choice(pool)
        while ret == idx:
            ret = np.random.choice(pool)
        return ret

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def prepare_clip_test(self):
        pass

    def prepare_clip_train(self, idx):
        vid, fid = self.sample_list[idx]
        video = self.video_infos[vid]
        frames = video['frames']
        quantize = video['quantize']
        if self.padding:
            pass
        else:
            frames = frames[fid: fid + self.seq_len]
            # img_names = [x['filename'] for x in frames]
            flows = [mmcv.flowread(osp.join(self.img_prefix, x['flow_name']),
                                   quantize=quantize) for x in frames]
            inv_flows = [mmcv.flowread(osp.join(self.img_prefix, x['inv_flow_name']),
                                       quantize=quantize) for x in frames]
            img_results = [self.prepare_img_train(frame) for frame in frames]
            # results encode the transformation info of img
            # augment the flow/inv_flow accordingly, pack it into results
            # resize => flip => pad
            for i, x in enumerate(img_results):
                if x is None:
                    return None
            aug_meta = img_results[0]['img_meta'].data
            resize = aug_meta['img_shape'][:2]
            pad = aug_meta['pad_shape'][:2]
            flip = aug_meta['flip']
            flow_aug = FlowAug(resize, pad, flip)
            flows = [torch.tensor(x).permute(2, 0, 1) for x in flows]
            inv_flows = [torch.tensor(x).permute(2, 0, 1) for x in inv_flows]
            flows = torch.stack(flows)
            inv_flows = torch.stack(inv_flows)
            flows, inv_flows = flow_aug(flows, inv_flows)
            for idx, img_result in enumerate(img_results):
                # if not isinstance(img_result, dict):
                #     print("Debug")
                #     print(img_result)
                #     print(frames[idx])
                #     print(frames)
                img_result['flow'] = flows[idx]
                img_result['inv_flow'] = inv_flows[idx]
            # embed()
            return img_results

    def get_ann_info(self, frame):
        ann = frame['ann']
        for k, v in ann.items():
            if isinstance(v, list):
                if 'bbox' in k:
                    ann[k] = np.array(v, dtype=np.float32).reshape(-1, 4)
                else:
                    ann[k] = np.array(v, dtype=np.int64)
        return ann

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []


    def prepare_img_train(self, frame):
        img_info = frame
        ann_info = self.get_ann_info(frame)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        return results

if __name__ == '__main__':
    data_root = '/databack1/KITTI/kitti/tracking/'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ]
    train = dict(
        ann_file=data_root + 'training/kitti_train_tracking.json',
        img_prefix=data_root,
        pipeline=train_pipeline)
    dataset = VideoCustomDataset(**train)
    embed()
