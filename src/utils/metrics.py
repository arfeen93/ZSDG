import time
import torch
import numpy as np

from algos.Tent.tent import softmax_entropy
from utils import utils
from utils.logger import AverageMeter


def compute_dg_acc(true, pred, dict_clss, tgt_classes, zsl):

	res = {}

	if zsl:
		per_class_acc = {}
		
		for clss in tgt_classes:
			
			is_class = true==dict_clss[clss]
			per_class_acc[clss] = ((pred[is_class]==true[is_class]).sum())/is_class.sum()

		cls_avg_top1_acc = np.mean(list(per_class_acc.values()))

		res = {'per_class_acc':per_class_acc, 'overall_acc':cls_avg_top1_acc}
	
	else:
		plain_acc = (pred==true).sum()/pred.shape[0]
		res = {'overall_acc':plain_acc}

	return res


def dg_acc_w_tent(loader_image, model, optimizer, dict_clss, tgt_classes, epoch, args, mode, zsl):

	model.train()

	# Start counting time
	start = time.time()

	entropy_l = AverageMeter()

	for i, (im, cls_im) in enumerate(loader_image):

		im = im.float().cuda()
			
		with torch.set_grad_enabled(True):
			
			for _ in range(args.tent_step):

				optimizer.zero_grad()
				
				clf_out, _ = model(im, mode=mode)
				
				loss = softmax_entropy(clf_out).mean(0)
				loss.backward()
				
				optimizer.step()

		if (i+1) % args.log_interval == 0:
			print('[Train] Epoch: [{0}][{1}/{2}]\t'
				  'entropy loss: {ent.val:.4f} ({ent.avg:.4f})\t'
				  .format(epoch, i+1, len(loader_image), ent=entropy_l))

	end = time.time()
	elapsed = end-start
	print(f"Time Taken:{elapsed//60:.0f}m{elapsed%60:.0f}s.\n")

	res = compute_zsl_acc(loader_image, model, dict_clss, tgt_classes, epoch, args, mode, zsl)

	return res