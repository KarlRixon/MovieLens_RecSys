import torch
from torch import nn, optim
from torchvision import transforms, utils
from torch.utils.data import DataLoader, random_split
from models import resnet101
from dataset import PosterDataset, Resize, ToTensor
import matplotlib
import numpy as np
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#%matplotlib inline

bs = 32
tbs = 8
epochs = 50
lr = 1e-3

def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b

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

device = torch.device('cuda')
model = resnet101().to(device)
criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
train_loss = []
train_acc = []

for epoch in range(epochs):
    model.train()
    losses = []
    for idx, item in enumerate(data_loader1):
        x, label = item['image'].to(device), item['label'].to(device)
        logits = model(x)
        loss = criteon(logits, label)
        losses.append(loss.item())

        # backporp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('epoch: ', epoch, '[', idx, '/', len(data_loader1), ']', '\tloss: ', loss.item())

    print('epoch: ', epoch, '\tloss: ', np.mean(losses))
    train_loss.append(np.mean(losses))

    model.eval()
    with torch.no_grad():
        # test
        total_correct = 0
        total_num = 0
        for idx, item in enumerate(data_loader2):
            x, labels = item['image'].to(device), item['labels']
            
            # 改进
            logits = model(x)
            pred = props_to_onehot(logits.cpu().numpy()) # 输出转为onehot
            
            for i in range(x.size(0)): # 不能设置为tbx因为最后一个batch可能会越界
                a = torch.IntTensor(pred[i]).to(device)
                b = torch.IntTensor(list(map(int, labels[i]))).to(device)
                total_correct += (a*b).sum().item() # 向量点乘，若模型预测结果在电影类型中则点乘为1，即预测正确
                
            total_num += x.size(0)

        acc = total_correct / total_num
        train_acc.append(acc)
        print('epoch: ', epoch, '\tacc: ', acc, '\n')

    # 保存神经网络
    if (epoch+1) % 10 == 0:
        torch.save(model, str(lr)+'_net_'+str(epoch+1)+'.pkl')                      # 保存整个神经网络的结构和模型参数
        print('saved in '+str(lr)+'_net_'+str(epoch+1)+'.pkl')
        
        plt.figure()
        plt.plot(train_loss, label='loss')   
        plt.legend()
        plt.savefig(f'./'+str(lr)+'_epoch'+str(epoch+1)+'_loss.png')
        
        plt.figure()
        plt.plot(train_acc, label='acc')   
        plt.legend()
        plt.savefig(f'./'+str(lr)+'_epoch'+str(epoch+1)+'_acc.png')

# plt.figure()
# plt.plot(train_loss, label='loss')   
# plt.legend()
# plt.savefig(f'./'+str(lr)+'_final_loss.png')

# plt.figure()
# plt.plot(train_acc, label='acc')   
# plt.legend()
# plt.savefig(f'./'+str(lr)+'_final_acc.png')

# # 保存神经网络
# torch.save(model, 'net_final.pkl')                      # 保存整个神经网络的结构和模型参数
# #     torch.save(net.state_dict(), 'net_params.pkl')  # 只保存神经网络的模型参数
# print('saved in net_final.pkl')