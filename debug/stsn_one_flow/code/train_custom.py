from __future__ import division
import argparse
import os
# import tqdm
import random
import torch
import torch.nn.functional as F
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
from IO import *
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
	optimizer_fuse = obj_from_dict(cfg.optimizer, torch.optim,
							 dict(params=list(model_load.agg.weight_feature1.parameters())+list(model_load.agg.weight_feature2.parameters())+ \
								 list(model_load.agg.weight_feature3.parameters())+list(model_load.agg.weight_feature4.parameters())+list(model_load.agg.weight_feature5.parameters())))
	optimizer_all=obj_from_dict(cfg.optimizer, torch.optim,
							 dict(params=model_load.parameters()))
	l2loss=torch.nn.MSELoss()
	check_video=None
	try:
		start_epoch=checkpoint['epoch']+1
	except:
		start_epoch=0
	# start_epoch=0
	meta=None
	epoch=start_epoch
	vis = visdom.Visdom(env='stsnflow')
	loss_flow_window = vis.line(X=torch.zeros((1,)).cpu(),
						Y=torch.zeros((1)).cpu(),
						opts=dict(xlabel='minibatches',
									ylabel='Loss of flow',
									title='Loss of flow ',
									legend=['Loss of flow']))
	pre_window = vis.image(
		np.random.rand(375, 1242),
		opts=dict(title='predict!', caption='predict.'),
	)
	ground_window = vis.image(
		np.random.rand(375, 1242),
		opts=dict(title='groud!', caption='groud.'),
	)
	# loss_cls_window = vis.line(X=torch.zeros((1,)).cpu(),
	# 					Y=torch.zeros((1)).cpu(),
	# 					opts=dict(xlabel='minibatches',
	# 								ylabel='Loss of classification',
	# 								title='Loss of classification ',
	# 								legend=['Loss of classification']))
	# loss_cls_window2 = vis.line(X=torch.zeros((1,)).cpu(),
	# 					Y=torch.zeros((1)).cpu(),
	# 					opts=dict(xlabel='minibatches',
	# 								ylabel='Loss of classification2',
	# 								title='Loss of classification2 ',
	# 								legend=['Loss of classification2']))
	# loss_cls_window3 = vis.line(X=torch.zeros((1,)).cpu(),
	# 					Y=torch.zeros((1)).cpu(),
	# 					opts=dict(xlabel='minibatches',
	# 								ylabel='Loss of classification3',
	# 								title='Loss of classification3 ',
	# 								legend=['Loss of classification3']))
	# loss_cls_window4 = vis.line(X=torch.zeros((1,)).cpu(),
	# 					Y=torch.zeros((1)).cpu(),
	# 					opts=dict(xlabel='minibatches',
	# 								ylabel='Loss of classification4',
	# 								title='Loss of classification4 ',
	# 								legend=['Loss of classification4']))
	# loss_init_window = vis.line(X=torch.zeros((1,)).cpu(),
	# 					Y=torch.zeros((1)).cpu(),
	# 					opts=dict(xlabel='minibatches',
	# 								ylabel='Loss of init reppoint',
	# 								title='Loss of init reppoint',
	# 								legend=['Loss of init reppoint']))
	# loss_refine_window = vis.line(X=torch.zeros((1,)).cpu(),
	# 					Y=torch.zeros((1)).cpu(),
	# 					opts=dict(xlabel='minibatches',
	# 								ylabel='Loss of refine reppoint',
	# 								title='Loss of refine reppoint',
	# 								legend=['Loss of refine reppoint']))
	# loss_total_window = vis.line(X=torch.zeros((1,)).cpu(),
	# 					Y=torch.zeros((1)).cpu(),
	# 					opts=dict(xlabel='minibatches',
	# 								ylabel='Loss all',
	# 								title='Loss all',
	# 								legend=['Loss all']))
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
			# print(data['img_meta'])
			# exit()
			optimizer_all.zero_grad()
			optimizer.zero_grad()
			reference_id=(data['img_meta'].data[0][0]['filename'].split('/')[-1]).split('.')[0]
			video_id=data['img_meta'].data[0][0]['filename'].split('/')[-2]
			print('start image:',data['img_meta'].data[0][0]['filename'])
			print('end image:',data['img_meta'].data[-1][-1]['filename'])
			with torch.no_grad():
				flow_data=[]
				for m in range(len(data['img_meta'].data)):
					one_card=[]
					for n in range(len(data['img_meta'].data[m])):
						flowname=data['img_meta'].data[m][0]['filename'].split('/')
						flowname[-3]='Flow'
						flowname='/'.join(flowname)
						flowname=flowname.split('.')
						flowname[-1]='flo'
						flowname='.'.join(flowname)
						flow=torch.from_numpy(np.array(readFlow(flowname))).cuda().permute(2,0,1).unsqueeze(0)
						flow=F.interpolate(flow,[375,1242],mode='bilinear')
						one_card.append(flow)
					one_card[1:]=one_card[:-1]
					one_card[0]=torch.zeros_like(flow)
					one_card=torch.cat(one_card,dim=0)
					flow_data.append(one_card)
				
				inv_flow_data=[]
				for m in range(len(data['img_meta'].data)):
					one_card=[]
					for n in range(len(data['img_meta'].data[m])):
						flowname=data['img_meta'].data[m][0]['filename'].split('/')
						flowname[-3]='Inv_Flow'
						flowname='/'.join(flowname)
						flowname=flowname.split('.')
						flowname[-1]='flo'
						flowname='.'.join(flowname)
						flow=torch.from_numpy(np.array(readFlow(flowname))).cuda().permute(2,0,1).unsqueeze(0)
						flow=F.interpolate(flow,[375,1242],mode='bilinear')
						one_card.append(flow)
					one_card[:-1]=one_card[1:]
					one_card[-1]=torch.zeros_like(flow)
					one_card=torch.cat(one_card,dim=0)
					inv_flow_data.append(one_card)
					# (375, 1242, 2)

				for m in range(len(data['img_meta'].data)):
					start_name=data['img_meta'].data[m][0]['filename'].split('/')[-2]
					end_check=False
					for n in range(len(data['img_meta'].data[m])):
						check_name=data['img_meta'].data[m][n]['filename'].split('/')[-2]
						if start_name!=check_name:
							print('end of video')
							end_check=True
							break
					if end_check:
						for n in range(len(data['img_meta'].data[m])):
							data['img_meta'].data[m][n]=data['img_meta'].data[m-1][n]
							data['gt_bboxes'].data[m][n]=data['gt_bboxes'].data[m-1][n]
							data['gt_labels'].data[m][n]=data['gt_labels'].data[m-1][n]
							data['img'].data[m][n]=data['img'].data[m-1][n]
							flow_data[m][n,...]=flow_data[m-1][n,...]
							inv_flow_data[m][n,...]=inv_flow_data[m-1][n,...]

				inv_flow_data=torch.cat(inv_flow_data,dim=0)
				flow_data=torch.cat(flow_data,dim=0)
				self_flow=torch.zeros_like(flow_data)
				flow_ground=[flow_data,inv_flow_data,self_flow]
				# print(flow_data.shape,inv_flow_data.shape)
				# torch.Size([16, 2, 375, 1242]) torch.Size([16, 2, 375, 1242])
				# if i==0:
				# 	data0=data
				# else:
				# 	data=data0
			# if i==0:
			# 	data0=data
			# 	flow0=flow_ground
			# else:
			# 	data=data0
			# 	flow_ground=flow0
			flow_pre=model(return_loss=True, **data)
			flow_loss=0
			for m in range(len(flow_pre)):
				#m=3
				check_flow=flow_ground[m]
				for n in range(len(flow_pre[m])):
					#n=5
					if n==0:
						# print(flow_pre[m][n].shape)
						scale_flow=F.interpolate(check_flow,[flow_pre[m][n].shape[-2],flow_pre[m][n].shape[-1]],mode='bilinear')
						flow_loss=flow_loss+(l2loss(scale_flow[:,0,...]/(check_flow.shape[-1]/flow_pre[m][n].shape[-1]),flow_pre[m][n][:,1,...])+ \
											l2loss(scale_flow[:,1,...]/(check_flow.shape[-2]/flow_pre[m][n].shape[-2]),flow_pre[m][n][:,0,...]))
											# *(check_flow.shape[-1]/flow_pre[m][n].shape[-1])
			flow_loss=flow_loss/(len(flow_pre))
			flow_loss.backward()
			print('epoch:',epoch,'index:',i,'video_id:',video_id,'reference_id:',reference_id,'flow_loss:',flow_loss.item())
			# optimizer_fuse.zero_grad()
			# optimizer.step()
			if flow_loss.item()<50:
				optimizer_all.step()
			else:
				optimizer_all.zero_grad()
				print('error grad')
			if flow_loss.item()>1e3:
				print('grad inf')
				exit()
			
			vis.line(
				X=torch.ones(1).cpu() * training_sample,
				Y=torch.ones(1).cpu() *flow_loss.item(),
				win=loss_flow_window,
				update='append')
			pre=flow_pre[0][0][3,...]
			# print(pre[0,...].shape,flow_pre[0][0].shape)
			pre[0,...]=flow_pre[0][0][3,1,...]
			pre[1,...]=flow_pre[0][0][3,0,...]
			pre=F.interpolate(pre.unsqueeze(0),[375,1242],mode='bilinear').squeeze(0)*(1242/pre.shape[-1])
			pre=pre.permute(1,2,0)
			pre=pre.data.cpu().numpy()
			# print(pre.shape)
			pre=flow_to_color(pre,convert_to_bgr=False)
			vis.image(
				pre.transpose(2,0,1),
				opts=dict(title='predict!', caption='predict.'),
				win=pre_window,
			)
			ground=flow_ground[0][3,...]
			ground=F.interpolate(ground.unsqueeze(0),[375,1242],mode='bilinear').squeeze(0)*(1242/ground.shape[-1])
			ground=ground.permute(1,2,0)
			ground=ground.data.cpu().numpy()
			ground=flow_to_color(ground,convert_to_bgr=False)
			vis.image(
				ground.transpose(2,0,1),
				opts=dict(title='ground!', caption='ground.'),
				win=ground_window,
			)
			training_sample+=1
		epoch+=1

"""

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

			# if epoch<15:
			# 	losses=(loss_all[1]+loss_all[2]+loss_all[3]+loss_all[4])/4
			# 	losses.backward()
			# 	optimizer.step()
			# 	# optimizer_rep.step()
			# else:
			# 	losses=loss_all[0]+(loss_all[1]+loss_all[2]+loss_all[3]+loss_all[4])/4
			# 	losses.backward()
			# 	# optimizer.step()
			# 	# optimizer_rep.step()
			# 	optimizer_all.step()


			# log_vars=log[0]
			# print('refer')
			# print('epoch:',epoch,'index:',i,'video_id:',video_id,'reference_id:',reference_id, \
			# 		'loss_cls:',log_vars['loss_cls'],'loss_init_box:',log_vars['loss_pts_init'], \
			# 			'loss_refine_box:',log_vars['loss_pts_refine'])
			# log_vars=log[1]

			# vis.line(
			# 	X=torch.ones(1).cpu() *training_sample,
			# 	Y=(log_vars['loss_cls']) * torch.ones(1).cpu(),
			# 	win=loss_cls_window,
			# 	update='append')


			# vis.line(
			# 	X=torch.ones(1).cpu() * training_sample,
			# 	Y=(log_vars['loss_pts_init']) * torch.ones(1).cpu(),
			# 	win=loss_init_window,
			# 	update='append')
			# vis.line(
			# 	X=torch.ones(1).cpu() * training_sample,
			# 	Y=(log_vars['loss_pts_refine']) * torch.ones(1).cpu(),
			# 	win=loss_refine_window,
			# 	update='append')
			# vis.line(
			# 	X=torch.ones(1).cpu() * training_sample,
			# 	Y=(losses).item() * torch.ones(1).cpu(),
			# 	win=loss_total_window,
			# 	update='append')
			# print('agg')
			# print('epoch:',epoch,'index:',i,'video_id:',video_id,'reference_id:',reference_id, \
			# 		'loss_cls:',log_vars['loss_cls'],'loss_init_box:',log_vars['loss_pts_init'], \
			# 			'loss_refine_box:',log_vars['loss_pts_refine'])
			log_vars=log[2]
			print('support1')
			print('epoch:',epoch,'index:',i,'video_id:',video_id,'reference_id:',reference_id, \
					'loss_cls:',log_vars['loss_cls'],'loss_init_box:',log_vars['loss_pts_init'], \
						'loss_refine_box:',log_vars['loss_pts_refine'])
			vis.line(
				X=torch.ones(1).cpu() *training_sample,
				Y=(log_vars['loss_cls']) * torch.ones(1).cpu(),
				win=loss_cls_window2,
				update='append')

			log_vars=log[3]
			print('support2')
			print('epoch:',epoch,'index:',i,'video_id:',video_id,'reference_id:',reference_id, \
					'loss_cls:',log_vars['loss_cls'],'loss_init_box:',log_vars['loss_pts_init'], \
						'loss_refine_box:',log_vars['loss_pts_refine'])	 
			vis.line(
				X=torch.ones(1).cpu() *training_sample,
				Y=(log_vars['loss_cls']) * torch.ones(1).cpu(),
				win=loss_cls_window3,
				update='append')
			log_vars=log[4]
			print('warp refer')
			print('epoch:',epoch,'index:',i,'video_id:',video_id,'reference_id:',reference_id, \
					'loss_cls:',log_vars['loss_cls'],'loss_init_box:',log_vars['loss_pts_init'], \
						'loss_refine_box:',log_vars['loss_pts_refine'])	 
			vis.line(
				X=torch.ones(1).cpu() *training_sample,
				Y=(log_vars['loss_cls']) * torch.ones(1).cpu(),
				win=loss_cls_window4,
				update='append')
			training_sample+=1
		epoch+=1
"""

if __name__ == '__main__':
	main()
