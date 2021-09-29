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

sys.path.append('/home/arfeen/ZSDG_domainnet/src/')
from options.options_cumix import Options
from data.Domainnet.zsl_splits import domainnet_zsl
from data.dataloaders import BaselineDataset
from models.cumix import CuMix
from utils import utils
from utils import metrics
from algos.CuMix.trainer import evaluate


def main(args):

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	use_gpu = torch.cuda.is_available()

	if use_gpu:
		cudnn.benchmark = True
		torch.cuda.manual_seed_all(args.seed)

	device = torch.device("cuda:0" if use_gpu else "cpu")
	print('\nDevice:{}'.format(device))

	#args.root_path = args.root_path_remote
	#args.checkpoint_path = args.checkpoint_path_remote

	tr_classes = np.load('/home/arfeen/ZSDG_domainnet/src/data/Domainnet/zsl_splits/train_classes.npy').tolist()
	va_classes = np.load('/home/arfeen/ZSDG_domainnet/src/data/Domainnet/zsl_splits/val_classes.npy').tolist()
	te_classes = np.load('/home/arfeen/ZSDG_domainnet/src/data/Domainnet/zsl_splits/test_classes.npy').tolist()

	semantic_vec = np.load('/home/arfeen/ZSDG_domainnet/src/data/Domainnet/w2v_domainnet.npy', allow_pickle=True,
                           encoding='latin1').item()
	if not args.trainvalid:
		seen_classes = tr_classes
		va_classes = va_classes
		unseen_classes = te_classes
		all_classes = seen_classes + va_classes + unseen_classes
	else:
		seen_classes = tr_classes + va_classes
		va_classes = va_classes
		unseen_classes = te_classes
		all_classes = seen_classes + unseen_classes
	dict_clss_tr = utils.create_dict_texts(seen_classes)
	dict_clss_all = utils.create_dict_texts(all_classes)
	if unseen_classes is not None:
		dict_clss_te = utils.create_dict_texts(unseen_classes)
	else:
		print("No unseen classes....check")

	# Imagenet standards
	im_mean = [0.485, 0.456, 0.406]
	im_std = [0.229, 0.224, 0.225]

	# Image transformations
	image_transforms = transforms.Compose([
			transforms.Resize((args.image_size, args.image_size)),
			transforms.ToTensor(),
			transforms.Normalize(im_mean, im_std)
		])

	splits_unseen_domain = domainnet_zsl.trvalte_per_domain(args, args.holdout_domain)
	data_te = BaselineDataset(np.array(splits_unseen_domain['te']), transforms=image_transforms)
	te_loader = DataLoader(dataset=data_te, batch_size=512, shuffle=False, num_workers=args.num_workers, pin_memory=True)
	print(f'#Test samples:{len(te_loader.dataset)}.\n')

	# Model
	model = CuMix(args.backbone, seen_classes, va_classes, unseen_classes, semantic_vec, input_dim=2048, semantic_dim=args.semantic_emb_size,
				  dg_only=args.dg_only).cuda()

	save_folder_name = args.holdout_domain
	path_cp = os.path.join(args.checkpoint_path, args.dataset, save_folder_name)
	today = datetime.now().strftime('%B %d, %Y')
	path_log = os.path.join('./results', args.dataset, save_folder_name, )
	#import pdb;pdb.set_trace()
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

		if best_model_name not in models_tested:

			best_model_file = os.path.join(path_cp, best_model_name)

			if os.path.isfile(best_model_file) and ('_tv-'+str(args.trainvalid) in best_model_name):
				
				print("\nLoading best model from '{}'".format(best_model_file))
				# load the best model yet
				checkpoint = torch.load(best_model_file)
				epoch = checkpoint['epoch']
				model.load_state_dict(checkpoint['model_state_dict'])
				print("Loaded best model '{0}' (epoch {1})\n".format(best_model_file, epoch))

				test_res = evaluate(te_loader, model, dict_clss_te, unseen_classes, epoch, args, 'test')
				outstr = 'ZERO SHOT DG:\n\n'
				outstr += "\n".join("{!r}: {!r},".format(k, v) for k, v in test_res['per_class_acc'].items())
				outstr += '\n\nOverall:' + '{0:.4f}'.format(test_res['overall_acc'])

				print(outstr)

				outstr = "\n".join("{!r}: {!r},".format(k, v) for k, v in test_res['per_class_acc'].items())
				outstr += '\n\nOverall:'+'{0:.4f}'.format(test_res['overall_acc'])
				
				print(outstr)
				result_file = open(os.path.join(path_log_save, best_model_name[:-len('.pth')]+'.txt'), 'w')
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