import sys
import os
import time
import numpy as np
import pickle
import glob
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append('/home/soumava/TTT-ZSDG/src/')
from options.options_snmpnet import Options
from data.dataloaders import BaselineDataset
from models.snmpnet import SnMpNet
from utils import utils
from trainer import evaluate


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

	image_transforms = transforms.Compose([
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

	data_te = BaselineDataset(data['splits']['te'], transforms=image_transforms)
	te_loader = DataLoader(data_te, batch_size=args.batch_size*3, shuffle=False, num_workers=args.num_workers, pin_memory=True)
	print(f'#Test samples:{len(te_loader.dataset)}.')

	if not args.dg_only:
		data_te_seen_cls = BaselineDataset(data['splits']['tr']+data['splits']['va'], transforms=image_transforms)
		te_seen_cls_loader = DataLoader(data_te_seen_cls, batch_size=args.batch_size*3, shuffle=False, num_workers=args.num_workers, pin_memory=True)
		print(f'#Test Seen Class samples:{len(te_seen_cls_loader.dataset)}.')

	model = SnMpNet(args.backbone, tr_classes=seen_classes, va_classes=va_classes, te_classes=unseen_classes, 
					all_classes=all_classes, attributes=data['semantic_vec'], semantic_dim=args.semantic_emb_size).cuda()

	if args.dataset=='DomainNet':
		if args.dg_only: save_folder_name = os.path.join(args.holdout_domain, 'dg', args.backbone)
		else: save_folder_name = os.path.join(args.holdout_domain, 'zsl', args.backbone)
	elif args.dataset=='PACS':
		save_folder_name = os.path.join(args.holdout_domain, args.backbone)

	path_cp = os.path.join(args.checkpoint_path, args.dataset, save_folder_name)
	path_log = os.path.join('./results', args.dataset, save_folder_name)
	if not os.path.isdir(path_log):
		os.makedirs(path_log)

	if len(os.listdir(path_log))>0:
		models_tested = [result_file.split('/')[-1][:-len('.txt')]+'.pth' for result_file in glob.glob(os.path.join(path_log, '*.*'))]
	else:
		models_tested = []
	
	print('Total models tested before: ', len(models_tested))

	for best_model_name in os.listdir(path_cp):

		if best_model_name not in models_tested:

			best_model_file = os.path.join(path_cp, best_model_name)
			txt_save_name = best_model_name[:-len('.pth')]

			if os.path.isfile(best_model_file) and ('_tv-'+str(args.trainvalid) in best_model_name):
				
				print("\nLoading best model from '{}'".format(best_model_file))
				# load the best model yet
				checkpoint = torch.load(best_model_file)
				epoch = checkpoint['epoch']
				model.load_state_dict(checkpoint['model_state_dict'])
				print("Loaded best model '{0}' (epoch {1})\n".format(best_model_file, epoch))

				if args.dg_only:
					test_res = evaluate(te_loader, model, dict_clss_te, all_classes, epoch, args, 'train')
					outstr = 'Overall Acc:'+'{0:.4f}'.format(test_res['overall_acc'])
				
				else:
					test_res = evaluate(te_loader, model, dict_clss_te, unseen_classes, epoch, args, 'test')
					outstr = 'ZERO SHOT:\n\n'
					outstr += "\n".join("{!r}: {!r},".format(k, v) for k, v in test_res['per_class_acc'].items())
					outstr += '\n\nOverall:'+'{0:.4f}'.format(test_res['overall_acc'])

					print(outstr)
				
					test_res_unseen = evaluate(te_loader, model, dict_clss_all, unseen_classes, epoch, args, 'gzs')
					test_res_seen = evaluate(te_seen_cls_loader, model, dict_clss_all, np.setdiff1d(all_classes, unseen_classes), epoch, args, 'gzs')

					HM = 2*test_res_seen['overall_acc']*test_res_unseen['overall_acc']/(test_res_seen['overall_acc'] + test_res_unseen['overall_acc'])

					outstr += '\n\nGENERALIZED ZERO SHOT:\n\n'

					outstr += 'Unseen Classes:\n\n'
					outstr += "\n".join("{!r}: {!r},".format(k, v) for k, v in test_res_unseen['per_class_acc'].items())
					outstr += '\n\nOverall:'+'{0:.4f}'.format(test_res_unseen['overall_acc'])

					outstr += '\n\nSeen Classes:\n\n'
					outstr += "\n".join("{!r}: {!r},".format(k, v) for k, v in test_res_seen['per_class_acc'].items())
					outstr += '\n\nOverall:'+'{0:.4f}'.format(test_res_seen['overall_acc'])

					outstr += '\n\nHM:'+'{0:.4f}'.format(HM)

				print(outstr)
				result_file = open(os.path.join(path_log, txt_save_name+'.txt'), 'w')
				result_file.write(outstr)
				result_file.close()

				print('\nTest Results saved!')

		else:
			continue


if __name__ == '__main__':
	# Parse options
	args = Options().parse()
	print('Parameters:\t' + str(args))
	main(args)