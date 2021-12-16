import sys
import torch

sys.path.append('/home/arfeen/ZSDG_cumix_gpu1/src/')
# user defined
from trainer import Trainer
from options.options_cumix import Options


def main(args):

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_gpu else "cpu")
    print('\nDevice:{}'.format(device))
    #import pdb;pdb.set_trace()
    print('root_path:', args.root_path)
    trainer = Trainer(args, device)
    #import pdb;pdb.set_trace()
    trainer.do_training()


if __name__ == '__main__':

    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))
    main(args)