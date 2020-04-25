import torch
from torch import nn
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from dataset import PosterDataset, Resize, ToTensor
import numpy as np
import warnings
warnings.filterwarnings("ignore")

bs = 32

transformed_dataset = PosterDataset(csv_file='./data.txt',
									root_dir='../data/fgsm/FGSM',
									transform=transforms.Compose([
										Resize(),
										ToTensor()
									]))
data_loader = DataLoader(transformed_dataset, batch_size=bs,shuffle=False)
print('train batches: ', len(data_loader))

device = torch.device('cuda')
org_model = torch.load('../data/models/origin_model.pkl').cuda()
fgsm_model = torch.load('../data/models/fgsm_model.pkl').cuda()
pgd_model = torch.load('../data/models/pgd_model.pkl').cuda()

def props_to_onehot(props):
	if isinstance(props, list):
		props = np.array(props)
	a = np.argmax(props, axis=1)
	b = np.zeros((len(a), props.shape[1]))
	b[np.arange(len(a)), a] = 1
	return b

def fgsm(model, X, y, epsilon=0.2):
	""" Construct FGSM adversarial examples on the examples X"""
	delta = torch.zeros_like(X, requires_grad=True)
	loss = nn.CrossEntropyLoss()(model(X + delta), y)
	loss.backward()
	return epsilon * delta.grad.detach().sign()

def epoch_adversarial(loader, model, **kwargs):
	"""Adversarial training/evaluation epoch over the dataset"""
	total_acc = 0.
	for idx, item in enumerate(loader):
		X,y,ys = item['image'].to(device), item['label'].to(device), item['labels']
		delta = fgsm(model, X, y)
		yp = model(X+delta)
			
		# 多标签acc算法
		pred = props_to_onehot(yp.detach().cpu().numpy()) # 输出转为onehot
		for i in range(X.size(0)): # 不能设置为tbx因为最后一个batch可能会越界
			a = torch.IntTensor(pred[i]).to(device)
			b = torch.IntTensor(list(map(int, ys[i]))).to(device)
			total_acc += (a*b).sum().item() # 向量点乘，若模型预测结果在电影类型中则点乘为1，即预测正确

	return total_acc / len(loader.dataset)
	


acc = epoch_adversarial(data_loader, org_model)
print('origin model acc on adv_example: ', acc)

acc = epoch_adversarial(data_loader, fgsm_model)
print('fgsm adv_train model acc on adv_example: ', acc)

acc = epoch_adversarial(data_loader, pgd_model)
print('pgd adv_train model acc on adv_example: ', acc)
