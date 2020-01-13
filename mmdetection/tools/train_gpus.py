from __future__ import division
import argparse
import os
import tqdm
import random
import torch
from IPython import embed
from collections import OrderedDict
import mmcv
from mmcv import Config
from mmcv.runner import DistSamplerSeedHook, Runner, obj_from_dict
from mmcv.runner import get_dist_info, load_checkpoint
from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
						train_detector)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.datasets import build_dataset, DATASETS, build_dataloader
from mmdet.models import build_detector


def parse_args():
	parser = argparse.ArgumentParser(description='Train a detector')
	parser.add_argument('config', help='train config file path')
	parser.add_argument('--work_dir', help='the dir to save logs and models')
	parser.add_argument(
		'--resume_from', help='the checkpoint file to resume from')
	parser.add_argument('--vis', action='store_true',
					help='whether visualzie result')
	parser.add_argument(
		'--validate',
		action='store_true',
		help='whether to evaluate the checkpoint during training')
	parser.add_argument(
		'--gpus',
		type=int,
		default=1,
		help='number of gpus to use '
		'(only applicable to non-distributed training)')
	parser.add_argument('--seed', type=int, default=None, help='random seed')
	parser.add_argument(
		'--launcher',
		choices=['none', 'pytorch', 'slurm', 'mpi'],
		default='none',
		help='job launcher')
	parser.add_argument('--local_rank', type=int, default=0)
	parser.add_argument(
		'--autoscale-lr',
		action='store_true',
		help='automatically scale lr with the number of gpus')
	args = parser.parse_args()
	if 'LOCAL_RANK' not in os.environ:
		os.environ['LOCAL_RANK'] = str(args.local_rank)

	return args
def parse_losses(losses):
	log_vars = OrderedDict()
	for loss_name, loss_value in losses.items():
		if isinstance(loss_value, torch.Tensor):
			log_vars[loss_name] = loss_value.mean()
		elif isinstance(loss_value, list):
			log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
		else:
			raise TypeError(
				'{} is not a tensor or list of tensors'.format(loss_name))

	loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

	log_vars['loss'] = loss
	for name in log_vars:
		log_vars[name] = log_vars[name].item()

	return loss, log_vars

def weights_to_cpu(state_dict):
	"""Copy a model state_dict to cpu.

	Args:
		state_dict (OrderedDict): Model weights on GPU.

	Returns:
		OrderedDict: Model weights on GPU.
	"""
	state_dict_cpu = OrderedDict()
	for key, val in state_dict.items():
		state_dict_cpu[key] = val.cpu()
	return state_dict_cpu

def main():
	args = parse_args()

	cfg = Config.fromfile(args.config)
	# set cudnn_benchmark
	if cfg.get('cudnn_benchmark', False):
		torch.backends.cudnn.benchmark = True
	# update configs according to CLI args
	if args.resume_from is not None:
		cfg.resume_from = args.resume_from
	cfg.gpus = args.gpus


	cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

	# init distributed env first, since logger depends on the dist info.
	if args.launcher == 'none':
		distributed = False
	else:
		distributed = True
		init_dist(args.launcher, **cfg.dist_params)

	# init logger before other steps
	logger = get_root_logger(cfg.log_level)
	logger.info('Distributed training: {}'.format(distributed))

	# set random seeds
	if args.seed is not None:
		logger.info('Set random seed to {}'.format(args.seed))
		set_random_seed(args.seed)
	datasets = [build_dataset(cfg.data.train)]
	model = build_detector(
		cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
	if cfg.load_from:
		checkpoint = load_checkpoint(model, cfg.load_from, map_location='cpu')
		model.CLASSES = datasets[0].CLASSES
	if cfg.load_from:
		checkpoint = load_checkpoint(model, cfg.load_from, map_location='cpu')
		model.CLASSES = datasets[0].CLASSES
	if cfg.checkpoint_config is not None:
		# save mmdet version, config file content and class names in
		# checkpoints as meta data
		cfg.checkpoint_config.meta = dict(
			mmdet_version=__version__,
			config=cfg.text,
			CLASSES=datasets[0].CLASSES)
	
	data_loader = build_dataloader(
		datasets[0],
		imgs_per_gpu=cfg.data.imgs_per_gpu,
		workers_per_gpu=cfg.data.workers_per_gpu,
		num_gpus=cfg.gpus,
		dist=False,
		shuffle=False)
	# put model on gpus
	model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
	model.train()
	if hasattr(model, 'module'):
		model_load = model.module
	optimizer = obj_from_dict(cfg.optimizer, torch.optim,
							 dict(params=model_load.parameters()))

	check_video=None
	start_epoch=0
	meta=None
	epoch=start_epoch
	for e in range(cfg.total_epochs):
		i=0
		print(data_loader.__len__())
		for i, data in enumerate(data_loader):
			reference_id=(data['img_meta'].data[0][0]['filename'].split('/')[-1]).split('.')[0]
			video_id=data['img_meta'].data[0][0]['filename'].split('/')[-2]
			losses=model(return_loss=True, **data)
			losses, log_vars = parse_losses(losses)
			optimizer.zero_grad()
			losses.backward()
			optimizer.step()
			# print('transform kernel check',model.module.agg.trans_kernel.sum().item())
			print('epoch:',epoch,'i:',i,'video_id:',video_id,'reference_id:',reference_id, \
					'loss_cls:',log_vars['loss_cls'],'loss_init_box:',log_vars['loss_pts_init'], \
						'loss_refine_box:',log_vars['loss_pts_refine'])
			if i % 1000 == 0:
				if meta is None:
					meta = dict(epoch=epoch + 1, iter=i)
				else:
					meta.update(epoch=epoch + 1, iter=i)
				checkpoint = {
					'meta': meta,
					'state_dict': weights_to_cpu(model.state_dict())
				}

				if optimizer is not None:
					checkpoint['optimizer'] = optimizer.state_dict()
				if not os.path.exists(cfg.work_dir):
					os.mkdir(cfg.work_dir)
				filename=os.path.join(cfg.work_dir,'epoch_{}_{}.pth'.format(epoch,i))
				torch.save(checkpoint,filename)

		if epoch % 1 == 0:
			if meta is None:
				meta = dict(epoch=epoch + 1, iter=i)
			else:
				meta.update(epoch=epoch + 1, iter=i)
			checkpoint = {
				'meta': meta,
				'state_dict': weights_to_cpu(model.state_dict())
			}

			if optimizer is not None:
				checkpoint['optimizer'] = optimizer.state_dict()
			if not os.path.exists(cfg.work_dir):
				os.mkdir(cfg.work_dir)
			filename=os.path.join(cfg.work_dir,'epoch_{}.pth'.format(epoch))
			torch.save(checkpoint,filename)

		epoch+=1
		
	# train_detector(
	#	 model,
	#	 datasets,
	#	 cfg,
	#	 distributed=distributed,
	#	 validate=args.validate,
	#	 logger=logger)


if __name__ == '__main__':
	main()
