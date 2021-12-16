import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.resnet import resnet18, resnet50

from .utils import UnitClassifier, weights_init


class CuMix(nn.Module):

	def __init__(self, arch, seen_classes, va_classes, unseen_classes, attributes, input_dim=2048, semantic_dim=300, dg_only=0):
		super(CuMix, self).__init__()

		self.dg_only = dg_only
		self.backbone = eval(arch)(pretrained=True)
		self.backbone.fc = nn.Identity()

		if self.dg_only:
			self.semantic_projector = nn.Identity()
			self.train_classifier = nn.Linear(input_dim, len(seen_classes)+len(unseen_classes))
			self.train_classifier.apply(weights_init)
		else:
			dropout = True
			self.semantic_projector = nn.Linear(input_dim, semantic_dim)
			self.semantic_projector.apply(weights_init)
			#seen_classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
			#unseen_classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
			self.train_classifier = UnitClassifier(attributes, seen_classes, semantic_dim)
			self.va_classifier = UnitClassifier(attributes, va_classes, semantic_dim)
			self.test_classifier = UnitClassifier(attributes, unseen_classes, semantic_dim)

			# self.mixup_ratio = nn.Sequential((OrderedDict([
            # ("fc8", nn.Linear(4096, 1024)),
            # ("relu8", nn.ReLU(inplace=True)),
            # ("drop8", nn.Dropout()),
            # ("fc10", nn.Linear(1024, 1))])) )

	
	def forward(self, x, mode='train'):

		feat = self.backbone(x)
		sem_out = self.semantic_projector(feat)

		if mode=='train' or self.dg_only: clf_out = self.train_classifier(sem_out)
		else: clf_out = self.test_classifier(sem_out)

		return clf_out, feat