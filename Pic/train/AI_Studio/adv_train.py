import torch
from torch import nn, optim
from torchvision import transforms, utils
from torch.utils.data import DataLoader, random_split
from models import resnet50
from dataset import PosterDataset, Resize, ToTensor
import matplotlib
import numpy as np
matplotlib.use('Agg')
from matplotlib import pyplot as plt

bs = 32
tbs = 8
epochs = 100
lr = 1e-3

transformed_dataset = PosterDataset(csv_file='./data.txt',
                                    root_dir='../data/posters/posters',
                                    transform=transforms.Compose([
                                        Resize(),
                                        ToTensor()
                                    ]))
train_size = int(0.8*len(transformed_dataset)+1)
test_size = int(0.2*len(transformed_dataset))
train_dataset, test_dataset = random_split(transformed_dataset, [train_size, test_size])
data_loader1 = DataLoader(train_dataset, batch_size=bs,shuffle=True)
data_loader2 = DataLoader(test_dataset, batch_size=tbs,shuffle=True)
print('train batches: ', len(data_loader1))
print('test batches: ', len(data_loader2))

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
    
def pgd_linf(model, X, y, epsilon=0.1, alpha=0.02, num_iter=10, randomize=False):
    """ Construct PGD adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def epoch_adversarial(loader, model, attack, opt=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_acc = 0.,0.
    for idx, item in enumerate(loader):
        X,y,ys = item['image'].to(device), item['label'].to(device), item['labels']
        delta = attack(model, X, y, **kwargs)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        # 多标签acc算法
        pred = props_to_onehot(yp.detach().cpu().numpy()) # 输出转为onehot
        for i in range(X.size(0)): # 不能设置为tbx因为最后一个batch可能会越界
            a = torch.IntTensor(pred[i]).to(device)
            b = torch.IntTensor(list(map(int, ys[i]))).to(device)
            total_acc += (a*b).sum().item() # 向量点乘，若模型预测结果在电影类型中则点乘为1，即预测正确

        total_loss += loss.item() * X.shape[0]
    return total_acc / len(loader.dataset), total_loss / len(loader.dataset)
	
device = torch.device('cuda')
model = resnet50().to(device)
criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
train_loss = []
train_acc = []

for epoch in range(epochs):
    # 对抗训练
    acc, loss = epoch_adversarial(data_loader1, model, pgd_linf, optimizer)
    train_loss.append(loss)
    train_acc.append(acc)
    
    print('epoch: [', epoch+1, '/', epochs, ']', '\tloss: ', loss, '\tacc: ', acc)

    # 保存神经网络
    if (epoch+1) % 10 == 0:
        torch.save(model, str(lr)+'_robust_net_'+str(epoch+1)+'.pkl')                      # 保存整个神经网络的结构和模型参数
        print('saved in '+str(lr)+'_robust_net_'+str(epoch+1)+'.pkl')
        
        plt.figure()
        plt.plot(train_loss, label='loss')   
        plt.legend()
        plt.savefig(f'./'+str(lr)+'_epoch'+str(epoch+1)+'_robust_loss.png')
        
        plt.figure()
        plt.plot(train_acc, label='acc')   
        plt.legend()
        plt.savefig(f'./'+str(lr)+'_epoch'+str(epoch+1)+'_robust_acc.png')
