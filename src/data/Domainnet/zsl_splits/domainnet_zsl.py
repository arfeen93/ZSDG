import os
import numpy as np
import glob

_BASE_PATH = '/home/arfeen/ZSDG_cumix_gpu1/src/data/Domainnet'


def create_trval_splits(args):

    tr_classes = np.load(os.path.join(_BASE_PATH, 'zsl_splits', 'train_classes.npy')).tolist()

    va_classes = np.load(os.path.join(_BASE_PATH, 'zsl_splits', 'val_classes.npy')).tolist()
    te_classes = np.load(os.path.join(_BASE_PATH, 'zsl_splits', 'test_classes.npy')).tolist()

    semantic_vec = np.load(os.path.join(_BASE_PATH, 'w2v_domainnet.npy'), allow_pickle=True, encoding='latin1').item()

    all_domains = []
    for f in os.listdir(os.path.join(args.root_path, 'DomainNet')):
        if os.path.isdir(os.path.join(args.root_path, 'DomainNet', f)):
            all_domains.append(f)

    unseen_domain = args.holdout_domain
    seen_domains = np.setdiff1d(all_domains, [unseen_domain]).tolist()

    print('\nSeen:{}; Unseen:{}.'.format(seen_domains, unseen_domain))

    fls_train = []
    fls_va = []

    for dom in seen_domains:
        
        for cl in tr_classes:
            domain_cl_path = os.path.join(args.root_path, 'DomainNet', dom, cl)
            fls_train += glob.glob(os.path.join(domain_cl_path, '*.*'))

        for cl in va_classes:
            domain_cl_path = os.path.join(args.root_path, 'DomainNet', dom, cl)
            fls_va += glob.glob(os.path.join(domain_cl_path, '*.*'))
    
    splits = {}
    splits['tr'] = fls_train
    splits['va'] = fls_va

    print('\n# Classes - Tr:{}; Va:{}; Te:{}'.format(len(tr_classes), len(va_classes), len(te_classes)))

    return tr_classes, va_classes, te_classes, semantic_vec, splits


def trvalte_per_domain(args, domain):

    tr_classes = np.load(os.path.join(_BASE_PATH, 'zsl_splits', 'train_classes.npy')).tolist()

    va_classes = np.load(os.path.join(_BASE_PATH, 'zsl_splits', 'val_classes.npy')).tolist()
    te_classes = np.load(os.path.join(_BASE_PATH, 'zsl_splits', 'test_classes.npy')).tolist()
    semantic_vec = np.load(os.path.join(_BASE_PATH, 'w2v_domainnet.npy'), allow_pickle=True, encoding='latin1').item()

    domain_path = os.path.join(args.root_path, 'DomainNet', domain)

    all_fls = np.array(glob.glob(os.path.join(domain_path, '*/*.*')))
    all_clss = np.array([f.split('/')[-2] for f in all_fls])

    fls_tr = []
    fls_va = []
    fls_te_unseen_cls = []

    for c in te_classes:
        #domain_cl_path = os.path.join(args.root_path, 'DomainNet', domain, c)
        #fls_te_unseen_cls += glob.glob(os.path.join(domain_cl_path, '*.*'))
        fls_te_unseen_cls += all_fls[np.where(all_clss==c)[0]].tolist()
    
    for c in tr_classes:
        #domain_cl_path = os.path.join(args.root_path, 'DomainNet', domain, c)
        #fls_tr += glob.glob(os.path.join(domain_cl_path, '*.*'))
        fls_tr += all_fls[np.where(all_clss==c)[0]].tolist()

    for c in va_classes:
        #domain_cl_path = os.path.join(args.root_path, 'DomainNet', domain, c)
        #fls_va += glob.glob(os.path.join(domain_cl_path, '*.*'))
        fls_va += all_fls[np.where(all_clss==c)[0]].tolist()
    #import pdb;pdb.set_trace()
    splits = {}
    splits['tr'] = fls_tr
    splits['va'] = fls_va
    splits['te'] = fls_te_unseen_cls

    return splits