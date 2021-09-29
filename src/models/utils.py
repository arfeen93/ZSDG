import torch
import torch.nn as nn

# Init ZSL classifier with normalized embeddings
class UnitClassifier(nn.Module):
	
	def __init__(self, attributes, classes, semantic_dim):
		super(UnitClassifier, self).__init__()
		
		self.fc = nn.Linear(semantic_dim, len(classes), bias=False)

		for i, c in enumerate(classes):

			norm_attributes = torch.from_numpy(attributes[c]).float()
			norm_attributes /= torch.norm(norm_attributes, 2)
			self.fc.weight[i].data[:] = norm_attributes

	def forward(self, x):
		o = self.fc(x)
		return o


def weights_init(m):
	
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		try:
			m.bias.data.fill_(0)
		except:
			print('bias not present')
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)