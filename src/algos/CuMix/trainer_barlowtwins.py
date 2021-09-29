import sys
import os
import time
import numpy as np
import pickle
import glob
from datetime import datetime

# pytorch, torch vision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append('/home/soumava/TTT-ZSDG/src/')
from options.options_cumix import Options
from data.dataloaders import BaselineDataset
from models.cumix import CuMix
from algos.BarlowTwins import barlowtwins
from trainer import evaluate
from utils.logger import AverageMeter
from utils import utils
from utils import metrics


def barlow_ttt(loader, model, args):

	# projector
	sizes = [args.semantic_emb_size] + list(map(int, args.projector.split('-')))
	layers = []
	for i in range(len(sizes) - 2):
		layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
		layers.append(nn.BatchNorm1d(sizes[i + 1]))
		layers.append(nn.ReLU(inplace=True))
	layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
	projector = nn.Sequential(*layers).cuda()

	# normalization layer for the representations z1 and z2
	bn_layer = nn.BatchNorm1d(args.semantic_emb_size, affine=False).cuda()
	
	if args.dg_only:
		classifier_params = model.train_classifier.parameters()
	else:
		classifier_params = model.semantic_projector.parameters()

	opt_net = optim.SGD(model.backbone.parameters(), weight_decay=0, lr=1e-5)
	opt_clf = optim.SGD(list(classifier_params) + list(bn_layer.parameters()), weight_decay=0, lr=1e-4)
	# opt_clf = optim.SGD(list(classifier_params) + list(projector.parameters()) + list(bn_layer.parameters()), weight_decay=0, lr=1e-4)
	# opt_clf = optim.SGD(classifier_params, weight_decay=0, lr=1e-4)

	model.train()

	bt_loss = AverageMeter()

	for epoch in range(args.epochs):

		# Start counting time
		start = time.time()

		for i, (im1, im2, _) in enumerate(loader):
			
			im1 = im1.float().cuda()
			im2 = im2.float().cuda()

			opt_net.zero_grad()
			opt_clf.zero_grad()

			# z1 = projector(model.semantic_projector(model.backbone(im1)))
			# z2 = projector(model.semantic_projector(model.backbone(im2)))
			z1 = model.semantic_projector(model.backbone(im1))
			z2 = model.semantic_projector(model.backbone(im2))

			# empirical cross-correlation matrix
			# c = torch.t(z1) @ z2
			c = torch.t(bn_layer(z1)) @ bn_layer(z2)
			c.div_(args.batch_size)

			on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
			off_diag = barlowtwins.off_diagonal(c).pow_(2).sum()
			loss = on_diag + args.lambd*off_diag			
			loss.backward()

			opt_net.step()
			opt_clf.step()

			bt_loss.update(loss.item(), im1.size(0))

			if (i+1) % args.log_interval == 0:
				print('[Train] Epoch: [{0}][{1}/{2}]\t'
					  'BT loss: {bt.val:.4f} ({bt.avg:.4f})\t'
					  .format(epoch+1, i+1, len(loader), bt=bt_loss))

		end = time.time()
		elapsed = end-start
		print(f"Time Taken:{elapsed//60:.0f}m{elapsed%60:.0f}s.\n")
	
	return model


def main(args):

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	use_gpu = torch.cuda.is_available()

	if use_gpu:
		cudnn.benchmark = True
		torch.cuda.manual_seed_all(args.seed)

	device = torch.device("cuda:0" if use_gpu else "cpu")
	print('\nDevice:{}'.format(device))

	args.root_path = args.root_path_remote
	args.checkpoint_path = args.checkpoint_path_remote

	if args.dataset=='PACS':
		args.dg_only = 1

	if args.dg_only:
		args.trainvalid = 0

	print('\nLoading data...')

	if args.dataset=='DomainNet' and args.dg_only:
		from data.Domainnet.dg_splits.domainnet_dg import create_trvalte_splits
		data = create_trvalte_splits(args)

	elif args.dataset=='DomainNet' and not args.dg_only:
		from data.Domainnet.zsl_splits.domainnet_zsl import trvalte_per_domain
		data = trvalte_per_domain(args, args.holdout_domain)

	elif args.dataset=='PACS':
		from data.PACS.pacs import create_trvalte_splits
		data = create_trvalte_splits(args)

	# Imagenet standards
	im_mean = [0.485, 0.456, 0.406]
	im_std = [0.229, 0.224, 0.225]

	image_transforms_te = transforms.Compose([
		transforms.Resize((args.image_size, args.image_size)),
		transforms.ToTensor(),
		transforms.Normalize(im_mean, im_std)
	])

	if args.dg_only:
		seen_classes = sorted(list(data['semantic_vec'].keys()))
		va_classes = None
		unseen_classes = None
		all_classes = seen_classes
	elif not args.dg_only and not args.trainvalid:
		seen_classes = data['tr_classes']
		va_classes = data['va_classes']
		unseen_classes = data['te_classes']
		all_classes = seen_classes + va_classes + unseen_classes
	else:
		seen_classes = data['tr_classes'] + data['va_classes']
		va_classes = data['va_classes']
		unseen_classes = data['te_classes']
		all_classes = seen_classes + unseen_classes

	dict_clss_tr = utils.create_dict_texts(seen_classes)
	dict_clss_all = utils.create_dict_texts(all_classes)
	if unseen_classes is not None:
		dict_clss_te = utils.create_dict_texts(unseen_classes)
	else:
		dict_clss_te = dict_clss_tr

	transform, transform_prime = barlowtwins.Transform_BT()

	data_ttt = barlowtwins.BarlowDataset(data['splits']['te'], transform, transform_prime)
	data_te = BaselineDataset(data['splits']['te'], transforms=image_transforms_te)
	
	ttt_loader = DataLoader(data_ttt, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
	te_loader = DataLoader(data_te, batch_size=192, shuffle=False, num_workers=args.num_workers, pin_memory=True)
	
	print(f'#Test samples:{len(te_loader.dataset)}.')

	model = CuMix(args.backbone, seen_classes, unseen_classes, data['semantic_vec'], input_dim=2048, semantic_dim=args.semantic_emb_size, 
				  dg_only=args.dg_only).cuda()

	save_folder_name = args.holdout_domain
	path_cp = os.path.join(args.checkpoint_path, args.dataset, save_folder_name)
	today = datetime.now().strftime('%B %d, %Y')
	path_log = os.path.join('./results', args.dataset, save_folder_name)
	if not os.path.isdir(path_log):
		os.makedirs(path_log)

	if len(os.listdir(path_log))>0:
		# models_tested = [result_file.split('/')[-1][:-len('.txt')]+'.pth' for result_file in glob.glob(os.path.join(path_log, '*/*.*'))]
		models_tested = [result_file.split('/')[-1][:-len('.txt')]+'.pth' for result_file in glob.glob(os.path.join(path_log, '*.*'))]
	else:
		models_tested = []
	
	print('Total models tested before: ', len(models_tested))

	# path_log_save = os.path.join(path_log, today)
	path_log_save = path_log
	if not os.path.isdir(path_log_save):
		os.makedirs(path_log_save)

	for best_model_name in os.listdir(path_cp):

		# if best_model_name not in models_tested:

		best_model_file = os.path.join(path_cp, best_model_name)
		txt_save_name = best_model_name[:-len('.pth')]

		if os.path.isfile(best_model_file):
			
			print("\nLoading best model from '{}'".format(best_model_file))
			# load the best model yet
			checkpoint = torch.load(best_model_file)
			epoch = checkpoint['epoch']
			model.load_state_dict(checkpoint['model_state_dict'])
			print("Loaded best model '{0}' (epoch {1})\n".format(best_model_file, epoch))

			model_ttt = barlow_ttt(ttt_loader, model, args)

			if args.dg_only:
				test_res = evaluate(te_loader, model_ttt, dict_clss_te, all_classes, epoch, args, 'train')
				outstr = 'Overall Acc:'+'{0:.4f}'.format(test_res['overall_acc'])			
			else:
				test_res = evaluate(te_loader, model_ttt, dict_clss_te, unseen_classes, epoch, args, 'test')
				outstr = 'ZERO SHOT:\n\n'
				outstr += "\n".join("{!r}: {!r},".format(k, v) for k, v in test_res['per_class_acc'].items())
				outstr += '\n\nOverall:'+'{0:.4f}'.format(test_res['overall_acc'])
			
			print(outstr)
			result_file = open(os.path.join(path_log_save, 'barlow_'+txt_save_name+'.txt'), 'w')
			result_file.write(outstr)
			result_file.close()

			print('\nTest Results saved!')

		# else:
		# 	continue


if __name__ == '__main__':
	# Parse options
	args = Options().parse()
	print('Parameters:\t' + str(args))
	main(args)