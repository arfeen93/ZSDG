import os
import numpy as np
import pickle
import sys
sys.path.append('/home/arfeen/ZSDG_main/src/')
from utils.utils import process_txt

BASE_PATH = '/home/arfeen/ZSDG_main/src/data/PACS'
DATA_PATH = 'PACS/kfold/'


def create_trvalte_splits(args):

	with open(os.path.join(BASE_PATH, 'PACS_glove300.pkl'), 'rb') as f:
		semantic_vec = pickle.load(f)

	all_domains = []
	for f in os.listdir(os.path.join(args.root_path, DATA_PATH)):
		if os.path.isdir(os.path.join(args.root_path, DATA_PATH, f)):
			all_domains.append(f)

	unseen_domain = args.holdout_domain
	seen_domains = np.setdiff1d(all_domains, [unseen_domain]).tolist()

	print('\nSeen:{}; Unseen:{}.'.format(seen_domains, unseen_domain))

	fls_train = []
	fls_va = []
	fls_train_cls = []
	fls_va_cls = []
	for dom in seen_domains:
		
		dom_tr_files = process_txt(os.path.join(BASE_PATH, dom + '_train_kfold.txt'), os.path.join(args.root_path, DATA_PATH))
		dom_va_files = process_txt(os.path.join(BASE_PATH, dom + '_crossval_kfold.txt'), os.path.join(args.root_path, DATA_PATH))
		
		fls_train += dom_tr_files['img_paths'].tolist()
		fls_train_cls += dom_tr_files['labels'].tolist()

		fls_va += dom_va_files['img_paths'].tolist()
		fls_va_cls += dom_va_files['labels'].tolist()
	unseen_domain_te_files = process_txt(os.path.join(BASE_PATH, unseen_domain + '_test_kfold.txt'), os.path.join(args.root_path, DATA_PATH))
	fls_te = unseen_domain_te_files['img_paths'].tolist()
	fls_te_cls = unseen_domain_te_files['labels']
	splits = {}
	cls = {}
	splits['tr'] = fls_train
	splits['va'] = fls_va
	splits['te'] = fls_te
	cls['tr'] = fls_train_cls
	cls['te'] = fls_te_cls
	cls['va'] = fls_va_cls
	#return {'semantic_vec':semantic_vec, 'splits':splits}
	return semantic_vec, splits, cls