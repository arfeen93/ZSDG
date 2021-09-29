import sys
import torch

sys.path.append('/home/arfeen/ZSDG_main/src/')
# user defined
from trainer import Trainer
from options.options_cumix import Options


def main(args):

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print('\nDevice:{}'.format(device))
    print('root_path:', args.root_path)
    trainer = Trainer(args)
    #import pdb;pdb.set_trace()
    trainer.do_training()


if __name__ == '__main__':

    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))
    main(args)