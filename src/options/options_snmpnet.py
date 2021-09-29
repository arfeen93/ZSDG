"""
    Parse input arguments
"""
import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='SnMpNet for DG/ZSDG.')
        
        parser.add_argument('-root_l', '--root_path', default='/mnt/17672a04-dd61-4677-b159-13ceafd6ba95/soumava/datasets/', type=str)
        parser.add_argument('-root_r', '--root_path_remote', default='/data/soumava/datasets/', type=str)
        parser.add_argument('-path_cpl', '--checkpoint_path', default='/mnt/17672a04-dd61-4677-b159-13ceafd6ba95/soumava/saved_models/ZSDG/SnMpnet/', 
                            type=str)
        parser.add_argument('-path_cpr', '--checkpoint_path_remote', default='/data/soumava/saved_models/ZSDG/SnMpnet/', type=str)
        parser.add_argument('-resume', '--resume_dict', default=None, type=str, help='checkpoint file to resume training from')
        parser.add_argument('-tr', '--transfer2remote', choices=[0, 1], default=1, type=int, help='use path_cpl/path_cpr for storing models.')

        parser.add_argument('-data', '--dataset', default='DomainNet', choices=['PACS', 'DomainNet'])

        parser.add_argument('-hd', '--holdout_domain', default='sketch', 
                            choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting', 'photo', 'art_painting', 'cartoon', 'real'])
        parser.add_argument('-dg', '--dg_only', choices=[0, 1], default=0, type=int, help='ZSDG (0) or just DG(1)')

        # DomainNet specific arguments
        parser.add_argument('-tv', '--trainvalid', choices=[0, 1], default=0, type=int, help='whether to include val class samples during training.\
                            1 if hyperparameter tuning done with val set; 0 if dg_only=1')
        # parser.add_argument('-gzs', '--generalized_zero_shot', choices=[0, 1], default=0, type=int, help='zsl or gzsl during DomainNet testing.')

        parser.add_argument('-arch', '--backbone', choices=['resnet18', 'resnet50'], default='resnet50', help='Backbone resnet model')
        parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adam'], default='sgd')

        # Loss weight & reg. parameters
        parser.add_argument('-wcce', '--wcce', default=1.0, type=float, help='Weight on Distance based CCE Loss')
        parser.add_argument('-wmse', '--wmse', default=0.0, type=float, help='Weight on MSE Loss')
        parser.add_argument('-wrat', '--wratio', default=0.0, type=float, help='Weight on Soft Crossentropy Loss for mixup ratio prediction')
        parser.add_argument('-alpha', '--alpha', default=0, type=float, help='Parameter to scale weights for Class Similarity Matrix')
        parser.add_argument('-l2', '--l2_reg', default=0.0, type=float, help='L2 Weight Decay for optimizer')

        # Size parameters
        parser.add_argument('-semsz', '--semantic_emb_size', choices=[200, 300], default=300, type=int, help='Glove/word2vec vector dimension')
        parser.add_argument('-imsz', '--image_size', default=224, type=int, help='Input size for query/gallery domain sample')
        
        # Model parameters
        parser.add_argument('-mixl', '--mixup_level', type=str, choices=['feat', 'img'], default='img', help='mixup at the image or feature level')
        parser.add_argument('-beta', '--mixup_beta', type=float, default=1, help='mixup interpolation coefficient')
        parser.add_argument('-step', '--mixup_step', type=int, default=2, help='Initial warmup steps for domain and class mixing ratios.')
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-bs', '--batch_size', default=60, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=6, help='Number of workers in data loader')
        
        # Optimization parameters
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N', help='Number of epochs to train (default: 100)')
        parser.add_argument('-lr', '--lr', type=float, default=0.001, metavar='LR', help='Initial learning rate for optimizer')
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')

        # Checkpoint parameters
        parser.add_argument('-es', '--early_stop', type=int, default=15, help='Early stopping epochs.')

        # I/O parameters
        parser.add_argument('-log', '--log_interval', type=int, default=400, metavar='N', help='How many batches to wait before logging training status')

        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()
