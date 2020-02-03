# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-10-14 01:22:42  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-10-14 01:22:42



def train(epoch,vae_dir,training_sample):
	""" One training epoch """
	model_view.train()

	# bce = torch.nn.BCELoss()
	# real_label=torch.ones(args.batch_size,1,1).cuda(0).detach()
	# fake_label=torch.zeros(args.batch_size,1,1).cuda(0).detach()
	for batch_idx, [image,depth,voxel,pose] in enumerate(train_loader):
		image=image.squeeze().cuda().view(-1,image.shape[-3],image.shape[-2],image.shape[-1])
		depth=depth.squeeze().cuda().view(-1,depth.shape[-3],depth.shape[-2],depth.shape[-1])
		voxel=voxel.squeeze().cuda().view(-1,1,voxel.shape[-3],voxel.shape[-2],voxel.shape[-1])
		pose=pose.squeeze().cuda().view(-1,pose.shape[-1])
		# print(torch.max(image),torch.min(image))
		optimizer_view.zero_grad()
		#reconstruction view1
		recon,front, back,loss_vae,loss_kl,loss_front,loss_back,loss_voxel = \
			model_view(image,depth,voxel,pose)
		loss_vae=loss_vae.mean()
		loss_front=loss_front.mean()
		loss_back=loss_back.mean()
		loss_voxel=loss_voxel.mean()
		loss_kl=loss_kl.mean()
		loss=loss_vae+loss_front+loss_back+loss_voxel+loss_kl
		loss.backward()
		optimizer_view.step()
		#visualization with visdom
		print('front error:', (torch.sum(torch.abs(front-depth[:,0,...]))/torch.sum((front>0).float())).item(),
		'back error:', (torch.sum(torch.abs(back-depth[:,1,...]))/torch.sum((back>0).float())).item(),)
		ground = image[0,...].data.cpu().numpy().astype('float32')
		ground = np.reshape(ground, [3,192,192])
		vis.image(
			ground,
			opts=dict(title='ground!', caption='ground.'),
			win=current_window,
		)
		image=recon[0,...].detach()
		image = image.data.cpu().numpy().astype('float32')
		image = np.reshape(image, [3, 192,192])
		vis.image(
			np.abs(image),
			opts=dict(title='Reconstruction!', caption='Reconstruction.'),
			win=recon_window,
		)
		image=front[0,...].detach().view(1,1,48,48)/48
		image=F.interpolate(image,[192,192],mode='bilinear',align_corners=True)
		image = image.data.cpu().numpy().astype('float32')
		image = np.reshape(image, [1, 192,192])
		vis.image(
			np.abs(image),
			opts=dict(title='front!', caption='front.'),
			win=front_window,
		)
		image=back[0,...].detach().view(1,1,48,48)/48
		image=F.interpolate(image,[192,192],mode='bilinear',align_corners=True)
		image = image.data.cpu().numpy().astype('float32')
		image = np.reshape(image, [1, 192,192])
		vis.image(
			np.abs(image),
			opts=dict(title='back!', caption='back.'),
			win=back_window,
		)

		image=depth[0,0,...].detach().view(1,1,48,48).float()/48
		image=F.interpolate(image,[192,192],mode='bilinear',align_corners=True)
		image = image.data.cpu().numpy().astype('float32')
		image = np.reshape(image, [1, 192,192])
		vis.image(
			np.abs(image),
			opts=dict(title='front!', caption='front.'),
			win=front_g_window,
		)
		image=depth[0,1,...].detach().view(1,1,48,48).float()/48
		image=F.interpolate(image,[192,192],mode='bilinear',align_corners=True)
		image = image.data.cpu().numpy().astype('float32')
		image = np.reshape(image, [1, 192,192])
		vis.image(
			np.abs(image),
			opts=dict(title='back!', caption='back.'),
			win=back_g_window,
		)

		vis.line(
			X=torch.ones(1).cpu() *training_sample,
			Y=(loss_vae).item() * torch.ones(1).cpu(),
			win=loss_vae_window,
			update='append')
		vis.line(
			X=torch.ones(1).cpu() * training_sample,
			Y=(loss_front).item() * torch.ones(1).cpu(),
			win=loss_df_window,
			update='append')
		vis.line(
			X=torch.ones(1).cpu() * training_sample,
			Y=(loss_back).item() * torch.ones(1).cpu(),
			win=loss_db_window,
			update='append')
		vis.line(
			X=torch.ones(1).cpu() * training_sample,
			Y=(loss_voxel).item() * torch.ones(1).cpu(),
			win=loss_voxel_window,
			update='append')
		training_sample+=1
		if batch_idx % 3 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}% training_sample:{:.0f}] \
loss_vae: {:.4f} loss_front: {:.4f} loss_back: {:.4f} loss_voxel: {:.4f}' \
       		.format(
				epoch, batch_idx * args.batch_size, len(dataset_train),
				args.batch_size * batch_idx / len(dataset_train)*100,training_sample, \
					loss_vae.item(),loss_front.item(),loss_back.item(),loss_voxel.item()
				))
	if (epoch)%1==0:

		filename = join(vae_dir, 'nvs_'+str(epoch)+'_'+str(training_sample)+'.pkl')
		torch.save({
			'epoch': epoch//10,
			'state_dict': model_view.state_dict(),
			'samples':training_sample
		},filename)

	vis.save(['3dnvs'])
	return training_sample

if __name__=='__main__':

	#vae recontruction
	model_view=VAE(3, LSIZE)
	model_view=torch.nn.DataParallel(model_view,device_ids=range(gpu_counts))
	model_view.cuda()
	optimizer_view = optim.Adam(model_view.parameters(),lr=learning_rate, betas=(0.5, 0.95))

	vis = visdom.Visdom(env='3dnvs')

	current_window = vis.image(
		np.random.rand(192,192),
		opts=dict(title='current!', caption='current.'),
	)
	recon_window = vis.image(
		np.random.rand(192,192),
		opts=dict(title='Reconstruction!', caption='Reconstruction.'),
	)
	front_window = vis.image(
		np.random.rand(192,192),
		opts=dict(title='front!', caption='front.'),
	)
	back_window = vis.image(
		np.random.rand(192,192),
		opts=dict(title='back!', caption='back.'),
	)
	front_g_window = vis.image(
		np.random.rand(192,192),
		opts=dict(title='front ground!', caption='front ground.'),
	)
	back_g_window = vis.image(
		np.random.rand(192,192),
		opts=dict(title='back ground!', caption='back ground.'),
	)
	loss_vae_window = vis.line(X=torch.zeros((1,)).cpu(),
						Y=torch.zeros((1)).cpu(),
						opts=dict(xlabel='minibatches',
									ylabel='Loss Reconstruction image',
									title='Reconstruction image Loss ',
									legend=['Reconstruction image Loss']))

	loss_df_window = vis.line(X=torch.zeros((1,)).cpu(),
						Y=torch.zeros((1)).cpu(),
						opts=dict(xlabel='minibatches',
									ylabel='Loss depth of front',
									title='Loss depth of front',
									legend=['Loss depth of front']))
	loss_db_window = vis.line(X=torch.zeros((1,)).cpu(),
						Y=torch.zeros((1)).cpu(),
						opts=dict(xlabel='minibatches',
									ylabel='Loss depth of back',
									title='Loss depth of back',
									legend=['Loss depth of back']))
	loss_voxel_window = vis.line(X=torch.zeros((1,)).cpu(),
						Y=torch.zeros((1)).cpu(),
						opts=dict(xlabel='minibatches',
									ylabel='Loss voxel',
									title='Loss voxel',
									legend=['Loss voxel']))
	# check vae dir exists, if not, create it
	vae_dir = join(args.logdir,'all_category_random_training')
	if not exists(vae_dir):
		mkdir(vae_dir)
	training_sample=0
	model_view.load_state_dict(torch.load('./log/all_category_split_training/nvs_76_84238.pkl')['state_dict'])
	print('load success!')
	trained=-1
	# dataset_test = ShapeNetDataset(root='/home/ld/3dvs/training_data',transform=transform_train, train=False)
	# test_loader = torch.utils.data.DataLoader(
	# 		dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=32,drop_last=True)
	data_load=1
	dataset_train = ShapeNetDataset(root='/data/3d/ShapeNetVox32/',transform=transform_train, train=True,data_load=data_load)
	train_loader = torch.utils.data.DataLoader(
			dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8,drop_last=True)
	for epoch in range(trained+1, args.epochs + 1):
		print('training')
		training_sample=train(epoch,vae_dir,training_sample)

		# data_load+=1
		if epoch%3==0:
			dataset_train=0
			train_loader=0
			data_load+=1
			dataset_train = ShapeNetDataset(root='/data/3d/ShapeNetVox32/',transform=transform_train, train=True,data_load=data_load)
			train_loader = torch.utils.data.DataLoader(
					dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=32,drop_last=True)

		if data_load==5:
			data_load=1

