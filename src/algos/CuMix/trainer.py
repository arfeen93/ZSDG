import os
import time
import numpy as np
#import math
#import pickle
#import paramiko

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn import MSELoss
import sys
sys.path.append('/home/arfeen/ZSDG_cumix_gpu1/src/')
from data.dataloaders import CuMixloader
from data.sampler import BalancedSampler
from data.dataloaders import BaselineDataset
from models.cumix import CuMix
from utils import utils
from utils.metrics import compute_dg_acc
from utils.logger import AverageMeter
from data.PACS import pacs
from data.Domainnet.dg_splits import domainnet_dg
from data.Domainnet.zsl_splits import domainnet_zsl

class Trainer:
	
	def __init__(self, args, device):

		self.args = args
		self.device = device
		print('root_path:', args.root_path)
		if not args.transfer2remote:
			args.root_path = args.root_path_remote

			args.checkpoint_path = args.checkpoint_path_remote
		else:
			print("transfer2remote is true")
			if not args.checkpoint_path_remote:
				print("checkpoint_path_remote not available, creating it now")
				os.makedirs(args.checkpoint_path_remote)

		if not args.checkpoint_path:
			print("checkpoint_path not available, creating it now")
			os.makedirs(args.checkpoint_path)
		print('\nLoading data...')
		#import pdb;pdb.set_trace()
		self.tr_classes, self.va_classes, self.te_classes, semantic_vec, data_splits = domainnet_zsl.create_trval_splits(args)

		#data = domainnet_dg.create_trvalte_splits(args)
		if not args.trainvalid:
			self.seen_classes = self.tr_classes
			self.va_classes = self.va_classes
			self.unseen_classes = self.te_classes
			self.all_classes = self.seen_classes + self.va_classes + self.unseen_classes
		else:
			self.seen_classes = self.tr_classes + self.va_classes
			self.unseen_classes = self.te_classes
			self.all_classes = self.seen_classes + self.unseen_classes
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		use_gpu = torch.cuda.is_available()

		if use_gpu:
			cudnn.benchmark = True
			torch.cuda.manual_seed_all(args.seed)
		
		# Imagenet standards
		im_mean = [0.485, 0.456, 0.406]
		im_std = [0.229, 0.224, 0.225]

		# Image transformations
		image_transforms = {
			'train': {
				'PACS': transforms.Compose([
					transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
					transforms.RandomHorizontalFlip(0.5),
					transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
					transforms.ToTensor(),
					transforms.Normalize(im_mean, im_std)
				]),
				'DomainNet': transforms.Compose([
					transforms.Resize(256),
					transforms.RandomCrop(args.image_size),
					transforms.RandomHorizontalFlip(0.5),
					transforms.ToTensor(),
					transforms.Normalize(im_mean, im_std)
				])
			},

			'eval': transforms.Compose([
				transforms.Resize((args.image_size, args.image_size)),
				transforms.ToTensor(),
				transforms.Normalize(im_mean, im_std)
			])
		}
		#data_splits = data[1]
		#import pdb;pdb.set_trace()
		if args.trainvalid:
			fls_tr = np.array(data_splits['tr'] + data_splits['va'])
		else:
			fls_tr = np.array(data_splits['tr'])

		cls_tr = np.array([f.split('/')[-2] for f in fls_tr])
		dom_tr = np.array([f.split('/')[-3] for f in fls_tr])
		tr_domains_unique = np.unique(dom_tr)

		# class dictionary
		self.dict_clss_tr = utils.create_dict_texts(self.seen_classes)
		if self.va_classes is not None:
			self.dict_clss_va = utils.create_dict_texts(self.va_classes)
		else:
			self.dict_clss_va = self.dict_clss_tr
		# doamin dictionary
		self.dict_doms = utils.create_dict_texts(tr_domains_unique)
		print(self.dict_doms)

		domain_ids = utils.numeric_classes(dom_tr, self.dict_doms)
		data_train = CuMixloader(fls_tr, cls_tr, dom_tr, self.dict_doms, transforms=image_transforms['train']['DomainNet'])
		train_sampler = BalancedSampler(domain_ids, args.batch_size//len(tr_domains_unique), domains_per_batch=len(tr_domains_unique))
		self.train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, 
									   pin_memory=True)
		
		print(f'#Tr samples:{len(data_train)}\n')
		if not args.trainvalid:
			data_val = BaselineDataset(data_splits['va'], transforms=image_transforms['eval'])
			self.val_loader = DataLoader(data_val, batch_size=args.batch_size*5, shuffle=False, num_workers=args.num_workers, pin_memory=True)
			print(f'#Val samples:{len(data_val)}.')
		print('Loading Done\n')

		self.RG = np.random.default_rng()
		#semantic_vec = data[0]
		# Model
		self.model = CuMix(args.backbone, self.seen_classes, self.va_classes, self.unseen_classes, semantic_vec, input_dim=2048,
						   semantic_dim=args.semantic_emb_size, dg_only=args.dg_only).to(device)
		
		self.criterion = nn.CrossEntropyLoss()
		self.mixup_criterion = self.soft_cce

		self.optimizer_net = optim.SGD(self.model.backbone.parameters(), weight_decay=args.l2_reg, momentum=0.9, nesterov=True, lr=args.lr_net)
		self.optimizer_clf = optim.SGD(self.get_classifier_params(), weight_decay=args.l2_reg, momentum=0.9, nesterov=True, lr=args.lr_clf)
		
		if args.dataset=='DomainNet':
			save_folder_name = args.holdout_domain

		if args.dataset=='PACS':
			save_folder_name = args.holdout_domain

		self.path_cp = os.path.join(args.checkpoint_path, args.dataset, save_folder_name)

		if args.transfer2remote:
			print("Transfer2remotr")
			self.path_cp_remote = os.path.join(args.checkpoint_path_remote, args.dataset, save_folder_name)
			if not self.path_cp_remote:
				os.makedirs(self.path_cp_remote)
		
		self.suffix = '_wcce-'+str(args.wcce)+'_wimg-'+str(args.mixup_w_img)+'_wfeat-'+str(args.mixup_w_feat)+'_e-'+str(args.epochs)+\
					  '_bs-'+str(args.batch_size)+'_lrnet-'+str(args.lr_net)+'_lrclf-'+str(args.lr_clf)+'_l2-'+str(args.l2_reg)+\
					  '_beta-'+str(args.mixup_beta)+'_warmup-'+str(args.mixup_step)+'_seed-'+str(args.seed)+'_tv-'+str(args.trainvalid)
		
		# exit(0)
		path_log = os.path.join('./logs', args.dataset, save_folder_name, self.suffix)
		# Logger
		print('Setting logger...', end='')
		self.logger = SummaryWriter(path_log)
		print('Done\n')

		self.start_epoch = 0
		self.best_acc = 0
		self.early_stop_counter = 0
		self.last_chkpt_name='init'

		self.resume_from_checkpoint(args.resume_dict)


	def get_classifier_params(self):
		
		if self.args.dg_only:
			return self.model.train_classifier.parameters()
		else: 
			return self.model.semantic_projector.parameters()


	def soft_cce(self, y_pred, y_true):
		
		loss = -torch.sum(y_true*torch.log_softmax(y_pred, dim=1), dim=1)		
		return loss.mean()
	
	
	def adjust_learning_rate(self):

		scale_lr = 0.1**(self.current_epoch//6) # //10
		
		lr_net = self.args.lr_net*scale_lr
		lr_clf = self.args.lr_clf*scale_lr
		
		for param_group in self.optimizer_net.param_groups:
			param_group['lr'] = lr_net

		for param_group in self.optimizer_clf.param_groups:
			param_group['lr'] = lr_clf


	# Create one hot labels
	def create_one_hot(self, y):
		
		y_onehot = torch.LongTensor(y.size(0), len(self.seen_classes)).to(self.device)
		y_onehot.zero_()
		y_onehot.scatter_(1, y.view(-1, 1), 1)
		
		return y_onehot


	def resume_from_checkpoint(self, resume_dict):

		if resume_dict is not None:
			print('==> Resuming from checkpoint: ',resume_dict)
			checkpoint = torch.load(os.path.join(self.path_cp_remote, resume_dict+'.pth'))
			self.start_epoch = checkpoint['epoch']+1
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict'])
			self.optimizer_clf.load_state_dict(checkpoint['optimizer_clf_state_dict'])
			if not self.args.trainvalid:
				self.best_acc = checkpoint['best_acc']
			self.last_chkpt_name = resume_dict


	def swap(self, xs, a, b):
		xs[a], xs[b] = xs[b], xs[a]

	
	def derange(self, xs):
		x_new = [] + xs
		for a in range(1, len(x_new)):
			b = self.RG.choice(range(0, a))
			self.swap(x_new, a, b)
		return x_new

	def off_diagonal(self, x):
		# return a flattened view of the off-diagonal elements of a square matrix
		n, m = x.shape
		assert n == m
		return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


	def get_mixed_samples(self, X, y, domain_ids, mixup_level='img'):

		batch_ratios = self.RG.beta(self.mixup_beta, self.mixup_beta, size=X.size(0))
		if mixup_level=='feat':
			ratio = np.expand_dims(batch_ratios, axis=1)
		elif mixup_level=='img':
			ratio = np.expand_dims(batch_ratios, axis=(1, 2, 3))
		ratio = torch.from_numpy(ratio).float().to(self.device)

		doms = list(range(len(torch.unique(domain_ids))))
		bs = X.size(0) // len(doms)
		selected = self.derange(doms)
		
		permuted_across_dom = torch.cat([(torch.randperm(bs) + selected[i]*bs) for i in range(len(doms))])
		permuted_within_dom = torch.cat([(torch.randperm(bs) + i*bs) for i in range(len(doms))])
		
		ratio_within_dom = torch.from_numpy(self.RG.binomial(1, self.mixup_domain, size=X.size(0)))
		#import pdb;pdb.set_trace()
		mixed_indices = ratio_within_dom*permuted_within_dom + (1. - ratio_within_dom)*permuted_across_dom
		mixed_indices = mixed_indices.long()

		X_mix = ratio*X + (1 - ratio)*X[mixed_indices]

		batch_ratios = np.expand_dims(batch_ratios, axis=1)
		batch_ratios = torch.from_numpy(batch_ratios).float().to(self.device)
		y_mix = batch_ratios*y + (1 - batch_ratios)*y[mixed_indices]
		self.clss_perm = self.clss[mixed_indices]
		self.batch_ratios = batch_ratios
		return X_mix, y_mix
	
	
	def do_epoch(self):

		self.model.train()

		batch_time = AverageMeter()
		cce_loss = AverageMeter()
		mixup_img_loss = AverageMeter()
		mixup_feat_loss = AverageMeter()
		total_loss = AverageMeter()
		#import pdb;pdb.set_trace()
		# Start counting time
		time_start = time.time()

		for i, (im, cl, domain_ids) in enumerate(self.train_loader):

			# Transfer im to cuda
			im = im.float().to(self.device)
			# Get numeric classes
			num_clss = utils.numeric_classes(cl, self.dict_clss_tr)
			#import pdb;pdb.set_trace()
			cls_numeric = torch.from_numpy(num_clss).long().to(self.device)
			self.clss = cls_numeric
			one_hot_labels = self.create_one_hot(cls_numeric)

			self.optimizer_net.zero_grad()
			self.optimizer_clf.zero_grad()

			clf_out, features = self.model(im, mode='train')
			cce_l = self.criterion(clf_out, cls_numeric)

			feat_mix, label_mix = self.get_mixed_samples(features, one_hot_labels.float(), domain_ids, mixup_level='feat')

			clf_out_mix = self.model.train_classifier(self.model.semantic_projector(feat_mix))
			feat_mix_l = self.mixup_criterion(clf_out_mix, label_mix)

			im_mix, label_mix = self.get_mixed_samples(im, one_hot_labels.float(), domain_ids, mixup_level='img')
			clf_out_mix, _ = self.model(im_mix, mode='train')
			img_mix_l = self.mixup_criterion(clf_out_mix, label_mix)

			"Barlow twin losses"
			#if self.mixup_domain !=0:
			# mixed_semantic_proj = self.model.semantic_projector(feat_mix)
			# orig_semantic_proj = self.model.semantic_projector(features)
			# sizes = 300  # semantic dimension
			# self.bn = nn.BatchNorm1d(sizes, affine=False).cuda()
			# c = torch.t(self.bn(mixed_semantic_proj)) @ self.bn(orig_semantic_proj)
			# c.div_(self.args.batch_size)
			# on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
			# off_diag = self.off_diagonal(c).pow_(2).sum()
			# barlow_l = on_diag + self.args.lambd * off_diag
			# loss = self.args.wcce * cce_l + self.args.mixup_w_feat * feat_mix_l + self.args.mixup_w_img * img_mix_l + \
			# 	   self.args.mixup_w_barlow * barlow_l

			# loss = self.args.wcce * cce_l + self.args.mixup_w_feat * feat_mix_l + self.args.mixup_w_img * img_mix_l + \
			# 	   self.args.mixup_w_barlow * barlow_l
			
			"Mixup ratio prediction"

			# if self.mixup_domain == 0:
			# 	orig_mixed_cat_feat = torch.cat((feat_mix, features), 1)
			# 	ratio_pred = self.model.mixup_ratio(orig_mixed_cat_feat)
			# 	ratio_pred = torch.sigmoid(ratio_pred)
			# 	#import pdb;pdb.set_trace()
			# 	lamdba_gt = [1 if self.clss[j] == self.clss_perm[j] else self.batch_ratios[j]  for j in range(len(self.batch_ratios))]
			# 	lamdba_gt = torch.Tensor(lamdba_gt)
			# 	lamdba_gt = lamdba_gt.reshape(lamdba_gt.shape[0], 1).cuda()
			# 	mix_ratio_l = MSELoss()(lamdba_gt, ratio_pred)
			loss = self.args.wcce * cce_l + self.args.mixup_w_feat * feat_mix_l + self.args.mixup_w_img * img_mix_l
			loss.backward()
			
			self.optimizer_net.step()
			self.optimizer_clf.step()

			# Store losses for visualization
			cce_loss.update(cce_l.item(), im.size(0))
			mixup_feat_loss.update(feat_mix_l.item(), im.size(0))
			mixup_img_loss.update(img_mix_l.item(), im.size(0))
			total_loss.update(loss.item(), im.size(0))


			# time
			time_end = time.time()
			batch_time.update(time_end - time_start)
			time_start = time_end

			if (i + 1) % self.args.log_interval == 0:
				print('[Train] Epoch: [{0}/{1}][{2}/{3}]\t'
					  # 'lr:{3:.6f}\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'cce {cce.val:.4f} ({cce.avg:.4f})\t'
					  'feat {feat.val:.4f} ({feat.avg:.4f})\t'
					  'img {img.val:.4f} ({img.avg:.4f})\t'
					  'net {net.val:.4f} ({net.avg:.4f})\t'
					  .format(self.current_epoch+1, self.args.epochs, i+1, len(self.train_loader), batch_time=batch_time, 
							  cce=cce_loss, feat=mixup_feat_loss, img=mixup_img_loss, net=total_loss))

			# if (i+1)==50:
			#     break

		return {'cce':cce_loss.avg, 'mixup_img':mixup_img_loss.avg, 'mixup_feat':mixup_feat_loss.avg, 'net':total_loss.avg}

	
	def do_training(self):

		print('***Train***')
		for self.current_epoch in range(self.start_epoch, self.args.epochs):

			start = time.time()

			self.mixup_beta = min(self.args.mixup_beta, max(self.args.mixup_beta*(self.current_epoch)/self.args.mixup_step, 0.1))
			self.mixup_domain = min(1.0, max((2*self.args.mixup_step - self.current_epoch)/self.args.mixup_step, 0.0))
			#self.mixup_domain = min(0.0,max((2 * self.args.mixup_step - self.current_epoch) / self.args.mixup_step, 0.0))
			print(f'\nAcross Class Mix Coeff:{self.mixup_beta}; Within Domain Mix Coeff:{self.mixup_domain}.\n')

			self.adjust_learning_rate()
	
			loss = self.do_epoch()


			"**************************"
			if not self.args.trainvalid:
				print('\n***Validation***')
				valid_res = evaluate(self.val_loader, self.model, self.dict_clss_va, self.va_classes,
									 self.current_epoch + 1, self.args, 'val')
				acc = valid_res['overall_acc']

				end = time.time()
				elapsed = end - start

				print(
					f"Epoch Time:{elapsed // 60:.0f}m{elapsed % 60:.0f}s lr_clf:{utils.get_lr(self.optimizer_clf):.7f} lr_net:{utils.get_lr(self.optimizer_net):.7f} Acc:{acc:.4f}\n")

				if acc > self.best_acc:
					self.best_acc = acc
					self.early_stop_counter = 0

					model_save_name = 'val_acc-' + '{0:.4f}'.format(acc) + '_ep-' + str(
						self.current_epoch + 1) + self.suffix
					utils.save_checkpoint({
						'epoch': self.current_epoch + 1,
						'model_state_dict': self.model.state_dict(),
						'optimizer_net_state_dict': self.optimizer_net.state_dict(),
						'optimizer_clf_state_dict': self.optimizer_clf.state_dict(),
						'best_acc': self.best_acc
					}, directory=self.path_cp, save_name=model_save_name, last_chkpt=self.last_chkpt_name)
					self.last_chkpt_name = model_save_name
				else:
					self.early_stop_counter += 1
					if self.args.early_stop == self.early_stop_counter:
						print(f"Validation Performance did not improve for {self.args.early_stop} epochs."
							  f"Early stopping by {self.args.epochs - self.current_epoch - 1} epochs.")

					print(f"Val Acc hasn't improved from {self.best_acc:.4f} for {self.early_stop_counter} epoch(s)!\n")

			else:
				end = time.time()
				elapsed = end-start

				# print(f"Epoch Time:{elapsed//60:.0f}m{elapsed%60:.0f}s lr:{utils.get_lr(self.optimizer):.6f}\n")
				print(f"Epoch Time:{elapsed//60:.0f}m{elapsed%60:.0f}s lr_net:{utils.get_lr(self.optimizer_net):.6f} lr_clf:{utils.get_lr(self.optimizer_clf):.6f}\n")

				model_save_name = 'ep-'+str(self.current_epoch+1)+self.suffix
				utils.save_checkpoint({
										'epoch':self.current_epoch+1,
										'model_state_dict':self.model.state_dict(),
										'optimizer_net_state_dict':self.optimizer_net.state_dict(),
										'optimizer_clf_state_dict':self.optimizer_clf.state_dict()
										}, directory=self.path_cp, save_name=model_save_name, last_chkpt=self.last_chkpt_name)
				self.last_chkpt_name = model_save_name

			# Logger step
			self.logger.add_scalar('Train/CE loss', loss['cce'], self.current_epoch)
			self.logger.add_scalar('Train/Mixup Img CE loss', loss['mixup_img'], self.current_epoch)
			self.logger.add_scalar('Train/Mixup Feat CE loss', loss['mixup_feat'], self.current_epoch)
			self.logger.add_scalar('Train/total loss', loss['net'], self.current_epoch)

		self.logger.close()

		# if self.args.transfer2remote:
		# 	ssh_client = paramiko.SSHClient()
		# 	ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		# 	ssh_client.connect(hostname='10.64.34.33', username='arfeen', password='arfeen')
		#
		# 	ftp_client = ssh_client.open_sftp()
		#
		# 	ftp_client.put(os.path.join(self.path_cp, self.last_chkpt_name+'.pth'), os.path.join(self.path_cp_remote, self.last_chkpt_name+'.pth'))
		# 	ftp_client.close()
		#
		# 	best_model_local = os.path.join(self.path_cp, self.last_chkpt_name+'.pth')
		# 	if os.path.isfile(best_model_local):
		# 		os.remove(best_model_local)
		# 	else:
		# 		print("Error: {} file not found".format(best_model_local))

		print('\n***Training and Validation complete***')


def evaluate(loader_image, model, dict_clss, tgt_classes, epoch, args, mode):

	model.eval()

	# Start counting time
	start = time.time()
	use_gpu = torch.cuda.is_available()
	device = torch.device("cuda:1" if use_gpu else "cpu")
	for i, (im, cls_im) in enumerate(loader_image):

		im = im.float().to(device)
		# Get numeric classes
		cls_numeric = utils.numeric_classes(cls_im, dict_clss)

		with torch.no_grad():
			
			_, feat = model(im)
			sem_out = model.semantic_projector(feat)
			
			if mode=='train' or args.dg_only:
				clf_out = model.train_classifier(sem_out)
			elif mode=='val':
				clf_out = model.va_classifier(sem_out)
			elif mode=='test':
				clf_out = model.test_classifier(sem_out)
			elif mode=='gzs':
				clf_out = model.all_classifier(sem_out)
			
			_, pred_labels = torch.max(clf_out.data, 1)

		if i == 0:
			cls_pred = pred_labels.cpu().data.numpy()
			cls_true = cls_numeric
		else:
			cls_pred = np.concatenate((cls_pred, pred_labels.cpu().data.numpy()), axis=0)
			cls_true = np.concatenate((cls_true, cls_numeric), axis=0)

		if (i+1) % args.log_interval == 0:
			print('[Eval] Epoch: [{0}][{1}/{2}]'.format(epoch, i+1, len(loader_image)))

	end = time.time()
	elapsed = end-start
	print(f"Time Taken:{elapsed//60:.0f}m{elapsed%60:.0f}s.\n")

	return compute_dg_acc(cls_true, cls_pred, dict_clss, tgt_classes, not args.dg_only)