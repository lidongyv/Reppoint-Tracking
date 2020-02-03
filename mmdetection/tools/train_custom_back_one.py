from __future__ import division
import argparse
import os
# import tqdm
import random
import torch
# #from IPython import embed
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
import visdom

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
	# torch.autograd.set_detect_anomaly(True)
	cfg = Config.fromfile(args.config)
	# set cudnn_benchmark
	if cfg.get('cudnn_benchmark', False):
		torch.backends.cudnn.benchmark = True
	# update configs according to CLI args
	if args.resume_from is not None:
		cfg.resume_from = args.resume_from
	cfg.gpus = args.gpus

	if args.autoscale_lr:
		# apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
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
	optimizer_rep = obj_from_dict(cfg.optimizer, torch.optim,
							 dict(params=model_load.bbox_head.parameters()))
	optimizer = obj_from_dict(cfg.optimizer, torch.optim,
							 dict(params=model_load.agg.parameters()))
	optimizer_all=obj_from_dict(cfg.optimizer, torch.optim,
							 dict(params=model_load.parameters()))
	check_video=None
	# start_epoch=checkpoint['epoch']+1
	start_epoch=0
	meta=None
	epoch=start_epoch
	vis = visdom.Visdom(env='stsnone')
	loss_cls_window = vis.line(X=torch.zeros((1,)).cpu(),
						Y=torch.zeros((1)).cpu(),
						opts=dict(xlabel='minibatches',
									ylabel='Loss of classification',
									title='Loss of classification ',
									legend=['Loss of classification']))

	loss_init_window = vis.line(X=torch.zeros((1,)).cpu(),
						Y=torch.zeros((1)).cpu(),
						opts=dict(xlabel='minibatches',
									ylabel='Loss of init reppoint',
									title='Loss of init reppoint',
									legend=['Loss of init reppoint']))
	loss_refine_window = vis.line(X=torch.zeros((1,)).cpu(),
						Y=torch.zeros((1)).cpu(),
						opts=dict(xlabel='minibatches',
									ylabel='Loss of refine reppoint',
									title='Loss of refine reppoint',
									legend=['Loss of refine reppoint']))
	loss_total_window = vis.line(X=torch.zeros((1,)).cpu(),
						Y=torch.zeros((1)).cpu(),
						opts=dict(xlabel='minibatches',
									ylabel='Loss all',
									title='Loss all',
									legend=['Loss all']))
	# loss_trans_window = vis.line(X=torch.zeros((1,)).cpu(),
	# 					Y=torch.zeros((1)).cpu(),
	# 					opts=dict(xlabel='minibatches',
	# 								ylabel='Loss trans',
	# 								title='Loss trans',
	# 								legend=['Loss trans']))
	training_sample=0
	for e in range(cfg.total_epochs):
		i=0
		if epoch % 1 == 0:
			if meta is None:
				meta = dict(epoch=epoch + 1, iter=i)
			else:
				meta.update(epoch=epoch + 1, iter=i)
			checkpoint = {
				'meta': meta,
				'state_dict': weights_to_cpu(model.state_dict()),
				'epoch':epoch
			}
			if not os.path.exists(cfg.work_dir):
				os.mkdir(cfg.work_dir)
			filename=os.path.join(cfg.work_dir,'epoch_{}.pth'.format(epoch))
			torch.save(checkpoint,filename)
		for i, data in enumerate(data_loader):
			# if len(data['gt_bboxes'].data[0][0]) == 0:
			#	 continue
			optimizer_all.zero_grad()
			reference_id=(data['img_meta'].data[0][0]['filename'].split('/')[-1]).split('.')[0]
			video_id=data['img_meta'].data[0][0]['filename'].split('/')[-2]
			print('start image:',data['img_meta'].data[0][0]['filename'])
			print('end image:',data['img_meta'].data[-1][-1]['filename'])
			# print(len(data['img'].data),len(data['img'].data[0]))
			# exit()
			for m in range(len(data['img_meta'].data)):
				start_name=data['img_meta'].data[m][0]['filename'].split('/')[-2]
				# print(data['img_meta'].data[m][0]['filename'])
				for n in range(len(data['img_meta'].data[m])):
					check_name=data['img_meta'].data[m][n]['filename'].split('/')[-2]
					# print(data['img_meta'].data[m][n]['filename'])
					if start_name!=check_name:
						print('end of video')
						data['img_meta'].data[m][n]=data['img_meta'].data[m][0]
						data['gt_bboxes'].data[m][n]=data['gt_bboxes'].data[m][0]
						data['gt_labels'].data[m][n]=data['gt_labels'].data[m][0]
						data['img'].data[m][n]=data['img'].data[m][0]
			
			losses=model(return_loss=True, **data)
			# losses,loss_check=model(return_loss=True, **data)
			# loss_check=loss_check.mean()
			# print('loss_check',loss_check.item())
			# print(losses)
			if isinstance(losses, list):
				
				loss_all=[]
				log=[]
				for p in range(len(losses)):
					# print(p)
					# print(losses[p])
					loss, log_var = parse_losses(losses[p])
					loss_all.append(loss)
					log.append(log_var)
			else:
				losses, log_vars = parse_losses(losses)
			# if isinstance(losses, list):
			# 	losses=loss_all[0]+0.5*loss_all[1]+0.5*loss_all[2]+0.5*loss_all[3]
			# 	losses=losses/2.5
			# print(loss_trans.shape)
			# loss_trans=torch.mean(loss_trans)*0.1
			# losses=losses+loss_trans
			# if losses.item()>10:
			#	 losses.backward(retain_graph=False)
			#	 optimizer.zero_grad()
			#	 continue

			if epoch<15:
				losses=0
				for l in range(2,len(loss_all)):
					losses+=loss_all[l]
				losses=losses/(len(loss_all)-2)
				losses.backward()
				optimizer.step()
				# optimizer_rep.step()
			else:
				losses=0
				for l in range(len(loss_all)):
					if l<=1:
						losses+=loss_all[l]
					else:
						losses+=loss_all[l]/(len(loss_all)-2)
				losses=losses/3
				losses.backward()
				optimizer.step()
				optimizer_rep.step()

			# if training_sample<700:
			# 	optimizer.step()
			# else:
			# 	optimizer_rep.step()
			# print('transform kernel check',model.module.agg.trans_kernel.sum().item())
			log_vars=log[0]


			print('refer')
			print('epoch:',epoch,'index:',i,'video_id:',video_id,'reference_id:',reference_id, \
					'loss_cls:',log_vars['loss_cls'],'loss_init_box:',log_vars['loss_pts_init'], \
						'loss_refine_box:',log_vars['loss_pts_refine'])
			log_vars=log[1]
			print('agg')
			print('epoch:',epoch,'index:',i,'video_id:',video_id,'reference_id:',reference_id, \
					'loss_cls:',log_vars['loss_cls'],'loss_init_box:',log_vars['loss_pts_init'], \
						'loss_refine_box:',log_vars['loss_pts_refine'])


			log_vars=log[2]
			print('support')
			print('epoch:',epoch,'index:',i,'video_id:',video_id,'reference_id:',reference_id, \
					'loss_cls:',log_vars['loss_cls'],'loss_init_box:',log_vars['loss_pts_init'], \
						'loss_refine_box:',log_vars['loss_pts_refine'])	 
			vis.line(
			X=torch.ones(1).cpu() *training_sample,
			Y=(log_vars['loss_cls']) * torch.ones(1).cpu(),
			win=loss_cls_window,
			update='append')
			vis.line(
				X=torch.ones(1).cpu() * training_sample,
				Y=(log_vars['loss_pts_init']) * torch.ones(1).cpu(),
				win=loss_init_window,
				update='append')
			vis.line(
				X=torch.ones(1).cpu() * training_sample,
				Y=(log_vars['loss_pts_refine']) * torch.ones(1).cpu(),
				win=loss_refine_window,
				update='append')
			vis.line(
				X=torch.ones(1).cpu() * training_sample,
				Y=(losses).item() * torch.ones(1).cpu(),
				win=loss_total_window,
				update='append')
			# vis.line(
			# 		 X=torch.ones(1).cpu() * training_sample,
			# 		 Y=loss_check.item() * torch.ones(1).cpu(),
			# 		 win=loss_trans_window,
			# 		 update='append')
			# log_vars=log[-1]
			# print('new_agg')
			# print('epoch:',epoch,'index:',i,'video_id:',video_id,'reference_id:',reference_id, \
			# 		'loss_cls:',log_vars['loss_cls'],'loss_init_box:',log_vars['loss_pts_init'], \
			# 			'loss_refine_box:',log_vars['loss_pts_refine'])	 
			training_sample+=1
			# if i % 300 == 0:
			# 	if meta is None:
			# 		meta = dict(epoch=epoch + 1, iter=i)
			# 	else:
			# 		meta.update(epoch=epoch + 1, iter=i)
			# 	checkpoint = {
			# 		'meta': meta,
			# 		'state_dict': weights_to_cpu(model.state_dict())
			# 	}

			# 	if optimizer_rep is not None:
			# 		checkpoint['optimizer'] = optimizer_rep.state_dict()
			# 	if not os.path.exists(cfg.work_dir):
			# 		os.mkdir(cfg.work_dir)
			# 	filename=os.path.join(cfg.work_dir,'epoch_{}_{}.pth'.format(epoch,i))
			# 	torch.save(checkpoint,filename)
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
