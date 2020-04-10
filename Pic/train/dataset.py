import os
import pandas as pd
from skimage import io, transform
import torch
from torch.utils.data import Dataset

class PosterDataset(Dataset):
	
	def __init__(self, csv_file, root_dir, transform=None):
		self.items = pd.read_csv(csv_file, sep='\t', header=None, converters={2: str}, encoding='latin-1') # 以字符串读取labels字段
		self.root_dir = root_dir
		self.transform = transform
		
	def __len__(self):
		return len(self.items)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		mid = self.items.iloc[idx, 0]
		image_name = os.path.join(self.root_dir, str(self.items.iloc[idx, 0]))+'.png'
		image = io.imread(image_name)
		label = self.items.iloc[idx, 1]
		labels = self.items.iloc[idx, 2]
		title = self.items.iloc[idx, 3]
		sample = {'image': image, 'label': label, 'labels': labels, 'title': title, 'id': mid}
		
		if self.transform:
			sample = self.transform(sample)
			
		return sample
		
class Resize(object):

	def __call__(self, sample):
		image, label, labels, title, mid = sample['image'], sample['label'], sample['labels'], sample['title'], sample['id']
		new_h, new_w = int(320), int(224)
		img = transform.resize(image, (new_h, new_w))
		return {'image': img, 'label': label, 'labels': labels, 'title': title, 'id': mid}
	
class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, label, labels, title, mid = sample['image'], sample['label'], sample['labels'], sample['title'], sample['id']
		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = image.transpose((2, 0, 1))
		return {'image': torch.from_numpy(image).type(torch.FloatTensor), 'label': label, 'labels': labels, 'title': title, 'id': mid}