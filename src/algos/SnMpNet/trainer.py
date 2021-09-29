import os
import time
import numpy as np
import math
import pickle
import paramiko

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from data.dataloaders import BaselineDataset, CuMixloader
from data.sampler import BalancedSampler
from models.snmpnet import SnMpNet
from losses.embedding_losses import Mixup_Cosine_CCE, Mixup_Euclidean_MSE
from utils import utils
from utils.metrics import compute_dg_acc
from utils.logger import AverageMeter


class Trainer:
	
	def __init__(self, args):

		self.args = args

		if not args.transfer2remote:
			args.root_path = args.root_path_remote
			args.checkpoint_path = args.checkpoint_path_remote

		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		use_gpu = torch.cuda.is_available()

		if use_gpu:
			cudnn.benchmark = True
			torch.cuda.manual_seed_all(args.seed)

		if args.dataset=='PACS':
			args.dg_only = 1

		if args.dg_only:
			args.trainvalid = 0

		print('\nLoading data...')

		if args.dataset=='DomainNet' and args.dg_only:
			from data.Domainnet.dg_splits.domainnet_dg import create_trvalte_splits
			data = create_trvalte_splits(args)

		elif args.dataset=='DomainNet' and not args.dg_only:
			from data.Domainnet.zsl_splits.domainnet_zsl import create_trval_splits
			data = create_trval_splits(args)

		elif args.dataset=='PACS':
			from data.PACS.pacs import create_trvalte_splits
			data = create_trvalte_splits(args)

		# Imagenet standards
		im_mean = [0.485, 0.456, 0.406]
		im_std = [0.229, 0.224, 0.225]

		# Image transformations
		image_transforms = {
			'train':{
				'PACS':transforms.Compose([
					transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
					transforms.RandomHorizontalFlip(0.5),
					transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
					transforms.ToTensor(),
					transforms.Normalize(im_mean, im_std)
				]),
				'DomainNet':transforms.Compose([
					transforms.Resize(256),
					transforms.RandomCrop(args.image_size),
					transforms.RandomHorizontalFlip(0.5),
					transforms.ToTensor(),
					transforms.Normalize(im_mean, im_std)
				])
			},
			
			'eval':transforms.Compose([
				transforms.Resize((args.image_size, args.image_size)),
				transforms.ToTensor(),
				transforms.Normalize(im_mean, im_std)
			])
		}

		fls_tr = np.array(data['splits']['tr'])
		if args.trainvalid:
			fls_tr = np.concatenate((fls_tr, np.array(data['splits']['va'])))
		
		cls_tr = np.array([f.split('/')[-2] for f in fls_tr])
		dom_tr = np.array([f.split('/')[-3] for f in fls_tr])
		tr_domains_unique = np.unique(dom_tr)

		if args.dg_only:
			self.seen_classes = sorted(list(data['semantic_vec'].keys()))
			self.va_classes = None
			self.unseen_classes = None
			self.all_classes = self.seen_classes
		elif not args.dg_only and not args.trainvalid:
			self.seen_classes = data['tr_classes']
			self.va_classes = data['va_classes']
			self.unseen_classes = data['te_classes']
			self.all_classes = self.seen_classes + self.va_classes + self.unseen_classes
		else:
			self.seen_classes = data['tr_classes'] + data['va_classes']
			self.va_classes = data['va_classes']
			self.unseen_classes = data['te_classes']
			self.all_classes = self.seen_classes + self.unseen_classes

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
		data_train = CuMixloader(fls_tr, cls_tr, dom_tr, self.dict_doms, transforms=image_transforms['train'][args.dataset])
		train_sampler = BalancedSampler(domain_ids, args.batch_size//len(tr_domains_unique), domains_per_batch=len(tr_domains_unique))
		self.train_loader = DataLoader(data_train, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)

		print(f'#Tr samples:{len(data_train)}.')

		if not args.trainvalid:
			data_val = BaselineDataset(data['splits']['va'], transforms=image_transforms['eval'])
			self.val_loader = DataLoader(data_val, batch_size=args.batch_size*5, shuffle=False, num_workers=args.num_workers, pin_memory=True)
			print(f'#Val samples:{len(data_val)}.')

		print('\nLoading Done\n')

		self.RG = np.random.default_rng()

		self.model = SnMpNet(args.backbone, tr_classes=self.seen_classes, va_classes=self.va_classes, te_classes=self.unseen_classes, 
							 all_classes=self.all_classes, attributes=data['semantic_vec'], semantic_dim=args.semantic_emb_size).cuda()

		self.semantic_vec = np.array([data['semantic_vec'].get(cl) for cl in self.seen_classes])
		self.clf_criterion = Mixup_Cosine_CCE(torch.from_numpy(self.semantic_vec).float().cuda())
		self.mse_criterion = Mixup_Euclidean_MSE(torch.from_numpy(self.semantic_vec).float().cuda(), args.alpha)

		if args.optimizer=='sgd':
			self.optimizer = optim.SGD(self.model.parameters(), weight_decay=args.l2_reg, momentum=args.momentum, nesterov=False, lr=args.lr)
		elif args.optimizer=='adam':
			self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.l2_reg)

		if args.dataset=='DomainNet':
			if args.dg_only: save_folder_name = os.path.join(args.holdout_domain, 'dg', args.backbone)
			else: save_folder_name = os.path.join(args.holdout_domain, 'zsl', args.backbone)
		elif args.dataset=='PACS':
			save_folder_name = os.path.join(args.holdout_domain, args.backbone)

		self.path_cp = os.path.join(args.checkpoint_path, args.dataset, save_folder_name)
		if args.transfer2remote:
			self.path_cp_remote = os.path.join(args.checkpoint_path_remote, args.dataset, save_folder_name)

		self.suffix = '_mixlevel-'+args.mixup_level+'_wcce-'+str(args.wcce)+'_wratio-'+str(args.wratio)+'_wmse-'+str(args.wmse)+\
					  '_clswts-'+str(args.alpha)+'_e-'+str(args.epochs)+'_es-'+str(args.early_stop)+'_opt-'+args.optimizer+\
					  '_bs-'+str(args.batch_size)+'_lr-'+str(args.lr)+'_l2-'+str(args.l2_reg)+'_beta-'+str(args.mixup_beta)+\
					  '_warmup-'+str(args.mixup_step)+'_seed-'+str(args.seed)+'_tv-'+str(args.trainvalid)

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


	def adjust_learning_rate(self, min_lr=1e-6):
		# lr = args.lr * 0.5 * (1.0 + math.cos(float(epoch) / args.epochs * math.pi))
		# epoch_curr = min(epoch, 20)
		# lr = args.lr * math.pow(0.001, float(epoch_curr)/ 20 )
		lr = self.args.lr * math.pow(1e-3, float(self.current_epoch)/20)
		lr = max(lr, min_lr)
		# print('epoch: {}, lr: {}'.format(epoch, lr))
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr


	def resume_from_checkpoint(self, resume_dict):

		if resume_dict is not None:
			print('==> Resuming from checkpoint: ',resume_dict)
			checkpoint = torch.load(os.path.join(self.path_cp, resume_dict+'.pth'))
			self.start_epoch = checkpoint['epoch']+1
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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

	
	def soft_cce(self, y_pred, y_true):
		loss = -torch.sum(y_true*torch.log_softmax(y_pred, dim=1), dim=1)
		return loss.mean()


	def get_mixed_samples(self, X, y, domain_ids, mixup_level='img'):

		batch_ratios = self.RG.beta(self.mixup_beta, self.mixup_beta, size=X.size(0))
		if mixup_level=='feat':
			ratio = np.expand_dims(batch_ratios, axis=1)
		elif mixup_level=='img':
			ratio = np.expand_dims(batch_ratios, axis=(1, 2, 3))
		ratio = torch.from_numpy(ratio).float().cuda()

		doms = list(range(len(torch.unique(domain_ids))))
		bs = X.size(0) // len(doms)
		selected = self.derange(doms)
		
		permuted_across_dom = torch.cat([(torch.randperm(bs) + selected[i] * bs) for i in range(len(doms))])
		permuted_within_dom = torch.cat([(torch.randperm(bs) + i * bs) for i in range(len(doms))])
		
		ratio_within_dom = torch.from_numpy(self.RG.binomial(1, self.mixup_domain, size=X.size(0)))
		mixed_indices = ratio_within_dom*permuted_within_dom + (1. - ratio_within_dom)*permuted_across_dom
		mixed_indices = mixed_indices.long()

		X_mix = ratio*X + (1-ratio)*X[mixed_indices]
		y_a, y_b = y, y[mixed_indices]

		ratio_vec_gt = torch.zeros([X.size()[0], len(self.seen_classes)]).cuda()
		for i in range(X.size()[0]):
			ratio_vec_gt[i, y_a[i]] += batch_ratios[i]
			ratio_vec_gt[i, y_b[i]] += 1-batch_ratios[i]

		return X_mix, y_a, y_b, torch.from_numpy(batch_ratios).float().cuda(), ratio_vec_gt


	def do_epoch(self):

		self.model.train()

		batch_time = AverageMeter()
		cce_loss = AverageMeter()
		mse_loss = AverageMeter()
		ratio_loss = AverageMeter()
		total_loss = AverageMeter()

		# Start counting time
		time_start = time.time()

		for i, (im, cl, domain_ids) in enumerate(self.train_loader):

			# Transfer im to cuda
			im = im.float().cuda()
			# Get numeric classes
			cls_numeric = torch.from_numpy(utils.numeric_classes(cl, self.dict_clss_tr)).long().cuda()

			self.optimizer.zero_grad()

			if self.args.mixup_level=='img':
				im, y_a, y_b, ratios, ratio_vec_gt = self.get_mixed_samples(im, cls_numeric, domain_ids, 'img')
			
			ratio_vec_pred, features = self.model(im)

			if self.args.mixup_level=='feat':
				features, y_a, y_b, ratios, ratio_vec_gt = self.get_mixed_samples(features, cls_numeric, domain_ids, 'feat')
			
			sem_out = self.model.semantic_projector(features)
			
			# Optimize parameters
			cce_l = self.clf_criterion(sem_out, y_a, y_b, ratios)
			mse_l = self.mse_criterion(sem_out, y_a, y_b, ratios)
			rat_l = self.soft_cce(ratio_vec_pred, ratio_vec_gt)
			loss = self.args.wcce*cce_l + self.args.wmse*mse_l + self.args.wratio*rat_l
			loss.backward()
			
			self.optimizer.step()

			# Store losses for visualization
			cce_loss.update(cce_l.item(), im.size(0))
			mse_loss.update(mse_l.item(), im.size(0))
			ratio_loss.update(rat_l.item(), im.size(0))
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
					  'mse {mse.val:.4f} ({mse.avg:.4f})\t'
					  'rat {rat.val:.4f} ({rat.avg:.4f})\t'
					  'net {net.val:.4f} ({net.avg:.4f})\t'
					  .format(self.current_epoch+1, self.args.epochs, i+1, len(self.train_loader), batch_time=batch_time, 
							  cce=cce_loss, mse=mse_loss, rat=ratio_loss, net=total_loss))

			# if (i+1)==50:
			#     break

		return {'cce':cce_loss.avg, 'mse':mse_loss.avg, 'ratio':ratio_loss.avg, 'net':total_loss.avg}


	def do_training(self):

		print('***Train***')
		for self.current_epoch in range(self.start_epoch, self.args.epochs):

			start = time.time()

			self.adjust_learning_rate()

			self.mixup_beta = min(self.args.mixup_beta, max(self.args.mixup_beta*(self.current_epoch)/self.args.mixup_step, 0.1))
			self.mixup_domain = min(1.0, max((2*self.args.mixup_step - self.current_epoch)/self.args.mixup_step, 0.0))
			print(f'\nAcross Class Mix Coeff:{self.mixup_beta}; Within Domain Mix Coeff:{self.mixup_domain}.\n')

			loss = self.do_epoch()

			if not self.args.trainvalid:
				print('\n***Validation***')
				valid_res = evaluate(self.val_loader, self.model, self.dict_clss_va, self.va_classes, self.current_epoch+1, self.args, 'val')			
				acc = valid_res['overall_acc']

				end = time.time()
				elapsed = end-start

				print(f"Epoch Time:{elapsed//60:.0f}m{elapsed%60:.0f}s lr:{utils.get_lr(self.optimizer):.7f} Acc:{acc:.4f}\n")
				
				if acc > self.best_acc:
					
					self.best_acc = acc
					self.early_stop_counter = 0
					
					model_save_name = 'val_acc-'+'{0:.4f}'.format(acc)+'_ep-'+str(self.current_epoch+1)+self.suffix
					utils.save_checkpoint({
											'epoch':self.current_epoch+1, 
											'model_state_dict':self.model.state_dict(),
											'optimizer_state_dict':self.optimizer.state_dict(), 
											'best_acc':self.best_acc
											}, directory=self.path_cp, save_name=model_save_name, last_chkpt=self.last_chkpt_name)
					self.last_chkpt_name = model_save_name
				
				else:
					self.early_stop_counter += 1
					if self.args.early_stop==self.early_stop_counter:
						print(f"Validation Performance did not improve for {self.args.early_stop} epochs."
							  f"Early stopping by {self.args.epochs-self.current_epoch-1} epochs.")
						break
					
					print(f"Val Acc hasn't improved from {self.best_acc:.4f} for {self.early_stop_counter} epoch(s)!\n")

			else:
				end = time.time()
				elapsed = end-start
				print(f"Epoch Time:{elapsed//60:.0f}m{elapsed%60:.0f}s lr:{utils.get_lr(self.optimizer):.7f}\n")

				model_save_name = 'ep-'+str(self.current_epoch+1)+self.suffix
				utils.save_checkpoint({
										'epoch':self.current_epoch+1, 
										'model_state_dict':self.model.state_dict(),
										'optimizer_state_dict':self.optimizer.state_dict()
										}, directory=self.path_cp, save_name=model_save_name, last_chkpt=self.last_chkpt_name)
				self.last_chkpt_name = model_save_name

			# Logger step
			self.logger.add_scalar('Train/CE loss', loss['cce'], self.current_epoch)
			self.logger.add_scalar('Train/MSE loss', loss['mse'], self.current_epoch)
			self.logger.add_scalar('Train/Mixup Ratio SoftCCE loss', loss['ratio'], self.current_epoch)
			self.logger.add_scalar('Train/total loss', loss['net'], self.current_epoch)
			if not self.args.trainvalid:
				self.logger.add_scalar('Val/acc', acc, self.current_epoch)

		self.logger.close()

		if self.args.transfer2remote:
			ssh_client = paramiko.SSHClient()
			ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
			ssh_client.connect(hostname='10.64.34.37', username='soumava', password='soumava')

			ftp_client = ssh_client.open_sftp()
			ftp_client.put(os.path.join(self.path_cp, self.last_chkpt_name+'.pth'), os.path.join(self.path_cp_remote, self.last_chkpt_name+'.pth'))
			ftp_client.close()

			best_model_local = os.path.join(self.path_cp, self.last_chkpt_name+'.pth')
			if os.path.isfile(best_model_local):
				os.remove(best_model_local)
			else:
				print("Error: {} file not found".format(best_model_local))

		print('\n***Training and Validation complete***')


def evaluate(loader_image, model, dict_clss, tgt_classes, epoch, args, mode):

	model.eval()

	# Start counting time
	start = time.time()

	for i, (im, cls_im) in enumerate(loader_image):

		im = im.float().cuda()
		# Get numeric classes
		cls_numeric = utils.numeric_classes(cls_im, dict_clss)

		with torch.no_grad():
			
			_, feat = model(im)
			sem_out = model.semantic_projector(feat)
			
			if mode=='train' or args.dg_only:
				clf_out = model.tr_classifier(sem_out)
			elif mode=='val':
				clf_out = model.va_classifier(sem_out)
			elif mode=='test':
				clf_out = model.te_classifier(sem_out)
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