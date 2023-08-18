from scipy.stats import normaltest
import numpy as np
import torch as t
import random, time

import torch.nn.functional as F
import torch.optim as optim
from torch import nn

# 定义一些超参数
lr = 0.001
momentum = 0.3
lambda1 = 0.00003
lambda2 = 0.00001
offset = 10

dtype = t.float

with open('rec','r+') as f:
    lines = f.readlines()
train_loader = []
x, y = [], []
for line in lines:
    l = line.strip('\n').split(' ')
    for i in range(6):
        l[i] = float(l[i])
    x.append(l[:3])
    y.append(l[3:])
for i in range(1, len(x) - 1):
    train_loader.append((t.tensor(x[(i - offset if i - offset > 0 else 0):i + 1], dtype=dtype).transpose(0,1), t.tensor(y[i], dtype=dtype)))
train_loader = train_loader[offset:-offset]

with open('test','r+') as f:
    lines = f.readlines()
test_loader = []
x, y = [], []
for line in lines:
    l = line.strip('\n').split(' ')
    for i in range(6):
        l[i] = float(l[i])
    x.append(l[:3])
    y.append(l[3:])
for i in range(1, len(x) - 1):
    test_loader.append((t.tensor(x[(i - offset if i - offset > 0 else 0):i + 1], dtype=dtype).transpose(0,1), t.tensor(y[i], dtype=dtype)))
test_loader = test_loader[offset:-offset]

def lossCalc(a, b):
    return t.sum(t.pow(a - b, 2)) / 2

class Net(nn.Module):
    """
    使用sequential构建网络，Sequential()函数的功能是将网络的层组合到一起
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),nn.BatchNorm1d(n_hidden_3))
        #self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),nn.BatchNorm1d(n_hidden_4))
        #self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),nn.BatchNorm1d(n_hidden_5))
        self.layer6 = nn.Linear(n_hidden_3, out_dim)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        #x = F.relu(self.layer4(x))
        #x = F.relu(self.layer5(x))
        x = self.layer6(x)
        return x

#检测是否有可用的GPU，有则使用，否则使用CPU
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
#实例化网络
model = Net(11, 300, 800, 300, 800, 300, 1)
try:
    model.load_state_dict(t.load(f'model_.mdl'))
    print(f'---------- Model loaded. ---------')
except:
    pass
model.to(device)

model_avg = Net(11, 300, 800, 300, 800, 300, 1)
try:
    model.load_state_dict(t.load(f'model_avg.mdl'))
    print(f'---------- Model_avg loaded. ---------')
except:
    pass
model_avg.to(device)

# 定义损失函数和优化器
criterion = lossCalc
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# 开始训练
epoch = 0
while True:
    cali = []
    train_loss_1, train_loss_2 = 0, 0
    model.train()
    #动态修改参数学习率
    '''
    if epoch == 5:
        optimizer.param_groups[0]['lr']*=0.1
        epoch = 0
    '''
    '''
    for idx, tpl in enumerate(train_loader):
        eul, facc = tpl
        eul=eul.to(device)
        facc = facc.to(device)
        # 前向传播
        out = model(eul)
        cali.append(out - facc)
        loss = criterion(out, facc)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss_1 += loss.item()
    '''
    xt, yt, recali = [], [], []
    for idx, tpl in enumerate(train_loader):
        x, y = tpl
        if idx <= 10:
            #xt.append(x.tolist())
            yt.append(y.tolist())
            continue
        #xt.pop(0)
        yt.pop(0)
        #xt.append(x.tolist())
        yt.append(y.tolist())
        y =  t.tensor([0, 0, 0], dtype=dtype)
        for i in yt:
            y += t.tensor(i, dtype=dtype)
        # 前向传播
        out = model_avg(x.to(device))
        recali.append(out)
        loss = criterion(out, y.to(device))
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss_2 += loss.item()

    '''
    # 在测试集上检验效果
    eval_loss = 0
    # 将模型改为预测模式
    model.eval()
    for idx, eul, facc in enumerate(test_loader):
        eul=eul.to(device)
        facc = facc.to(device)
        out = model(eul)
        loss = criterion(out, facc, eul)
        # 记录误差
        eval_loss += loss.item()
    '''
    try:
        print('epoch: {}, Final train Loss: {:.4f}'#, Test Loss: {:.4f}'
            .format(epoch, train_loss_2 / len(train_loader)))#, eval_loss / len(test_loader)))
        random.seed(time.time())
        x = random.sample(train_loader, 5)
        #print(f'{x[1].tolist()} \n{model_avg(x[0].to(device)).tolist()}')
        for i in x:
            print(i[1].tolist(),'\n' , model_avg(i[0]).tolist(), '\n')
    except:
        pass

    t.save(model.state_dict(), f'model_.mdl')
    t.save(model_avg.state_dict(), f'model_avg.mdl')
    epoch += 1