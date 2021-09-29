"""
    Parse input arguments
"""
import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='CuMix for ZSDG.')
        
        parser.add_argument('-root_l', '--root_path', default='/mnt/c61a35bf-fc59-4aab-a996-b254f9ab9052/arfeen', type=str)
        parser.add_argument('-root_r', '--root_path_remote', default='/home/arfeen/datasets/', type=str)
        parser.add_argument('-path_cpl', '--checkpoint_path', default='/mnt/c61a35bf-fc59-4aab-a996-b254f9ab9052/arfeen/saved_models/CuMix/', type=str)
        parser.add_argument('-path_cpr', '--checkpoint_path_remote', default='/home/arfeen/ZSDG_domainnet/saved_models/CuMix/', type=str)
        parser.add_argument('-resume', '--resume_dict', default=None, type=str, help='checkpoint file to resume training from')
        parser.add_argument('-tr', '--transfer2remote', choices=[0, 1], default=1, type=int, help='use path_cpl/path_cpr for storing models.')

        parser.add_argument('-data', '--dataset', default='DomainNet', choices=['PACS', 'DomainNet'])
        
        # DomainNet specific arguments
        parser.add_argument('-tv', '--trainvalid', choices=[0, 1], default=0, type=int, help='whether to include val class samples during training.\
                            1 if hyperparameter tuning done with val set; 0 if dg_only=1')
        
        parser.add_argument('-hd', '--holdout_domain', default='quickdraw', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-dg_only', '--dg_only', choices=[0, 1], default=0, type=int, help='ZSDG (0) or just DG(1)')

        parser.add_argument('-arch', '--backbone', choices=['resnet18', 'resnet50'], default='resnet50', help='Backbone resnet model')

        # Loss weight & reg. parameters
        parser.add_argument('-wcce', '--wcce', default=1.0, type=float, help='Weight on Distance based CCE Loss for Sketch and Image Encoders')
        parser.add_argument('-wimg', '--mixup_w_img', type=float, default=0.001, help='Weight to image mixup CE loss')
        parser.add_argument('-wfeat', '--mixup_w_feat', type=float, default=2, help='Weight to feature mixup CE loss')
        parser.add_argument('-wbarlow', '--mixup_w_barlow', type=float, default=0.05, help='Weight to barlow twins loss')
        parser.add_argument('-wmix_ratio', '--mixup_w_mix_ratio', type=float, default=50, help='Weight to mix ratio prediction mse loss')
        parser.add_argument('-l2', '--l2_reg', default=5e-5, type=float, help='L2 Weight Decay for optimizer')

        # Size parameters
        parser.add_argument('-semsz', '--semantic_emb_size', choices=[200, 300], default=300, type=int, help='Glove vector dimension')
        parser.add_argument('-imsz', '--image_size', default=224, type=int, help='Input size for query/gallery domain sample')
        
        # Model parameters
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-beta', '--mixup_beta', type=float, default=1, help='mixup interpolation coefficient')
        parser.add_argument('-step', '--mixup_step', type=int, default=2, help='Initial warmup steps for domain and class mixing ratios.')
        parser.add_argument('-bs', '--batch_size', default=45, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=8, help='Number of workers in data loader')
        
        # Optimization parameters
        parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N', help='Number of epochs to train (default: 100)')
        parser.add_argument('-lrb', '--lr_net', type=float, default=0.0001, metavar='LR', help='Initial learning rate for backbone')
        parser.add_argument('-lrc', '--lr_clf', type=float, default=0.001, metavar='LR', help='Initial learning rate for classifier')

        # I/O parameters
        parser.add_argument('-log', '--log_interval', type=int, default=400, metavar='N', help='How many batches to wait before logging training status')

        # Tent parameters
        parser.add_argument('-tstep', '--tent_step', type=int, default=1, help='Entropy minm steps for Tent')
        parser.add_argument('-es', '--early_stop', type=int, default=15, help='Early stopping epochs.')
        # Barlow Twins parameters
        parser.add_argument('--projector', default='300-300', type=str, metavar='MLP', help='projector MLP')
        parser.add_argument('-lambd', '--lambd', type=float, default=0.0051, help='redundancy reduction loss weight')

        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()
