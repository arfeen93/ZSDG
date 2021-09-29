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
from data.Domainnet import domainnet
from data.dataloaders import BaselineDataset
from models.cumix import CuMix
from algos.Tent import tent
from utils import utils
from utils import metrics


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

	tr_classes = np.load('/home/soumava/TTT-ZSDG/src/data/Domainnet/train_classes.npy').tolist()
	va_classes = np.load('/home/soumava/TTT-ZSDG/src/data/Domainnet/val_classes.npy').tolist()
	te_classes = np.load('/home/soumava/TTT-ZSDG/src/data/Domainnet/test_classes.npy').tolist()

	semantic_vec = np.load('/home/soumava/TTT-ZSDG/src/data/Domainnet/w2v_domainnet.npy', allow_pickle=True, 
                           encoding='latin1').item()

	seen_classes = tr_classes + va_classes
	unseen_classes = te_classes

	dict_clss_te = utils.create_dict_texts(unseen_classes)

	# Imagenet standards
	im_mean = [0.485, 0.456, 0.406]
	im_std = [0.229, 0.224, 0.225]

	# Image transformations
	image_transforms = transforms.Compose([
			transforms.Resize((args.image_size, args.image_size)),
			transforms.ToTensor(),
			transforms.Normalize(im_mean, im_std)
		])

	splits_unseen_domain = domainnet.trvalte_per_domain(args, args.holdout_domain, 0, tr_classes, va_classes, te_classes)
	data_te = BaselineDataset(np.array(splits_unseen_domain['te']), transforms=image_transforms)
	te_loader = DataLoader(dataset=data_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
	print(f'#Test samples:{len(te_loader.dataset)}.\n')

	# Model
	model = CuMix(args.backbone, seen_classes, unseen_classes, semantic_vec, input_dim=2048, semantic_dim=args.semantic_emb_size, 
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

		if os.path.isfile(best_model_file):
			
			print("\nLoading best model from '{}'".format(best_model_file))
			# load the best model yet
			checkpoint = torch.load(best_model_file)
			epoch = checkpoint['epoch']
			model.load_state_dict(checkpoint['model_state_dict'])
			print("Loaded best model '{0}' (epoch {1})\n".format(best_model_file, epoch))

			print('\nTenting the model...', end='')

			tented_model = tent.configure_model(model)
			params, param_names = tent.collect_params(tented_model)
			# params, param_names = tent.collect_params(model)
			opt_tent = optim.Adam(params, lr=args.lr_clf, betas=(0.9, 0.999), weight_decay=0)

			print('Done')

			per_class_acc, acc_overall = metrics.zsl_acc_w_tent(te_loader, tented_model, opt_tent, dict_clss_te, te_classes, epoch, args)

			outstr = "\n".join("{!r}: {!r},".format(k, v) for k, v in per_class_acc.items())
			outstr += '\n\nOverall:'+'{0:.4f}'.format(acc_overall)
			
			print(outstr)
			result_file = open(os.path.join(path_log_save, 'tent_'+best_model_name[:-len('.pth')]+'.txt'), 'w')
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