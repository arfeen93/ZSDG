import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

from .utils import UnitClassifier


class SnMpNet(nn.Module):

	def __init__(self, arch, tr_classes, va_classes, te_classes, all_classes, attributes, semantic_dim=300):
		super(SnMpNet, self).__init__()
		
		self.backbone = eval(arch)(pretrained=True)
		feat_dim = self.backbone.fc.in_features
		self.backbone.fc = nn.Identity()
		self.semantic_projector = nn.Linear(feat_dim, semantic_dim)
		self.ratio_predictor_tr = nn.Linear(feat_dim, len(tr_classes))

		self.tr_classifier = UnitClassifier(attributes, tr_classes, semantic_dim)
		if va_classes is not None:
			self.va_classifier = UnitClassifier(attributes, va_classes, semantic_dim)
		if te_classes is not None:
			self.ratio_predictor_te = nn.Linear(feat_dim, len(te_classes))
			self.te_classifier = UnitClassifier(attributes, te_classes, semantic_dim)

		self.all_classifier = UnitClassifier(attributes, all_classes, semantic_dim)
	
	def forward(self, x):

		feat = self.backbone(x)
		mixup_ratio = self.ratio_predictor_tr(feat)

		return mixup_ratio, feat