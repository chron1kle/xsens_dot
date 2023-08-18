from scipy.stats import normaltest
import numpy as np
import torch as t
import random, time
#导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

# 定义一些超参数
#learning_rate = 0.01
num_epoches = 20
lr = 0.0001
momentum = 0.1

dtype = t.float
with open('rec','r+') as f:
    lines = f.readlines()
train_loader_x = []
train_loader_y = []
train_loader_z = []
for line in lines:
    l = line.strip('\n').split(' ')
    for i in range(6):
        l[i] = [float(l[i])]
    train_loader_x.append((t.tensor(l[:3], dtype=dtype), t.tensor([l[3]], dtype=dtype)))
    train_loader_y.append((t.tensor(l[:3], dtype=dtype), t.tensor([l[4]], dtype=dtype)))
    train_loader_z.append((t.tensor(l[:3], dtype=dtype), t.tensor([l[5]], dtype=dtype)))
with open('test','r+') as f:
    lines = f.readlines()
test_loader_x = []
test_loader_y = []
test_loader_z = []
for line in lines:
    l = line.strip('\n').split(' ')
    for i in range(6):
        l[i] = [float(l[i])]
    test_loader_x.append((t.tensor(l[:3], dtype=dtype), t.tensor([l[3]], dtype=dtype)))
    test_loader_y.append((t.tensor(l[:3], dtype=dtype), t.tensor([l[4]], dtype=dtype)))
    test_loader_z.append((t.tensor(l[:3], dtype=dtype), t.tensor([l[5]], dtype=dtype)))

class Net(nn.Module):
    """
    使用sequential构建网络，Sequential()函数的功能是将网络的层组合到一起
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),nn.BatchNorm1d (n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),nn.BatchNorm1d (n_hidden_3))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim))
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x

#检测是否有可用的GPU，有则使用，否则使用CPU
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
#实例化网络
model_x = Net(3, 500, 1000, 100, 1)
model_y = Net(3, 500, 1000, 100, 1)
model_z = Net(3, 500, 1000, 100, 1)
for i in range(20, 0, -1):
    try:
        model_x.load_state_dict(t.load(f'model_x_{i-1}.mdl'))
        print(f'---------- Model_x num. {i-1} loaded. ---------')
        model_y.load_state_dict(t.load(f'model_y_{i-1}.mdl'))
        print(f'---------- Model_y num. {i-1} loaded. ---------')
        model_z.load_state_dict(t.load(f'model_z_{i-1}.mdl'))
        print(f'---------- Model_z num. {i-1} loaded. ---------')
        break
    except:
        pass
model_x.to(device)
model_y.to(device)
model_z.to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer_x = optim.SGD(model_x.parameters(), lr=lr, momentum=momentum)
optimizer_y = optim.SGD(model_y.parameters(), lr=lr, momentum=momentum)
optimizer_z = optim.SGD(model_z.parameters(), lr=lr, momentum=momentum)

# 开始训练
for epoch in range(num_epoches):
    train_loss_x = 0
    train_loss_y = 0
    train_loss_z = 0
    model_x.train()
    model_y.train()
    model_z.train()
    #动态修改参数学习率
    if epoch%5==0:
        optimizer_x.param_groups[0]['lr']*=0.1

    for eul, facc in train_loader_x:
        eul=eul.to(device)
        facc = facc.to(device)
        #eul = eul.view(eul.size(0), -1)
        # 前向传播
        out = model_x(eul)
        loss = criterion(out, facc)
        # 反向传播
        optimizer_x.zero_grad()
        loss.backward()
        optimizer_x.step()
        # 记录误差
        train_loss_x += loss.item()
        '''
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == facc).sum().item()
        acc = num_correct / eul.shape[0]
        train_acc_x += acc
        '''
    #losses_x.append(train_loss_x / len(train_loader_x))
    for eul, facc in train_loader_y:
        eul=eul.to(device)
        facc = facc.to(device)
        #eul = eul.view(eul.size(0), -1)
        # 前向传播
        out = model_y(eul)
        loss = criterion(out, facc)
        # 反向传播
        optimizer_y.zero_grad()
        loss.backward()
        optimizer_y.step()
        # 记录误差
        train_loss_y += loss.item()
    #losses_y.append(train_loss_y / len(train_loader_y))
    for eul, facc in train_loader_z:
        eul=eul.to(device)
        facc = facc.to(device)
        #eul = eul.view(eul.size(0), -1)
        # 前向传播
        out = model_z(eul)
        loss = criterion(out, facc)
        # 反向传播
        optimizer_z.zero_grad()
        loss.backward()
        optimizer_z.step()
        # 记录误差
        train_loss_z += loss.item()
    #losses_z.append(train_loss_z / len(train_loader_z))

    # 在测试集上检验效果
    eval_loss_x = 0
    eval_loss_y = 0
    eval_loss_z = 0
    # 将模型改为预测模式
    model_x.eval()
    model_y.eval()
    model_z.eval()
    for eul, facc in test_loader_x:
        eul=eul.to(device)
        facc = facc.to(device)
        out = model_x(eul)
        loss = criterion(out, facc)
        # 记录误差
        eval_loss_x += loss.item()
        '''
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == facc).sum().item()
        acc = num_correct / eul.shape[0]
        eval_acc += acc
        '''
    #eval_losses_x.append(eval_loss_x / len(test_loader_x))
    for eul, facc in test_loader_y:
        eul=eul.to(device)
        facc = facc.to(device)
        out = model_y(eul)
        loss = criterion(out, facc)
        # 记录误差
        eval_loss_y += loss.item()
    #eval_losses_y.append(eval_loss_y / len(test_loader_y))
    for eul, facc in test_loader_z:
        eul=eul.to(device)
        facc = facc.to(device)
        out = model_z(eul)
        loss = criterion(out, facc)
        # 记录误差
        eval_loss_z += loss.item()
    #eval_losses_z.append(eval_loss_z / len(test_loader_x))

    print('epoch: {}, Train Loss x: {:.4f}, Train Loss y: {:.4f}, Train Loss z: {:.4f}, Test Loss x: {:.4f}, Test Loss y: {:.4f}, Test Loss z: {:.4f}'
        .format(epoch, train_loss_x / len(train_loader_x), train_loss_y / len(train_loader_y), train_loss_z / len(train_loader_z), 
            eval_loss_x / len(test_loader_x), eval_loss_y / len(test_loader_y), eval_loss_z / len(test_loader_z)))
    random.seed(time.time())
    x, y, z = random.choice(train_loader_x), random.choice(train_loader_y), random.choice(train_loader_z)
    print(f'Example: {x[1]} {y[1]} {z[1]} Out: {model_x(x[0].to(device))} {model_y(y[0].to(device))} {model_z(z[0].to(device))}')

    t.save(model_x.state_dict(), f'model_x_{epoch}.mdl')
    t.save(model_y.state_dict(), f'model_y_{epoch}.mdl')
    t.save(model_z.state_dict(), f'model_z_{epoch}.mdl')
'''
def normaljudge(data):
    stat, p = normaltest(data)
    if p > 0.05:
        return "Positive"
    else: return f"Negative: p = {p}"

def normalData():
    ax, ay, az = [], [], []
    with open('.\\rec.txt', '+r') as f:
        file = f.readlines()
        for line in file:
            line = line.strip('\n')
            try:
                l = line.split(' ')
                if (len(l) < 8): continue
                ax.append(round(float(l[6]), 6))
                ay.append(round(float(l[5]), 6))
                az.append(round(float(l[4]), 6))
            except:
                pass
    ax = np.array(ax)
    ay = np.array(ay)
    az = np.array(az)
    #print(f"{normaljudge(ax)}, {normaljudge(ay)}, {normaljudge(az)}")
    print(f"ax_mean: {np.mean(ax)} ay_mean: {np.mean(ay)} az_mean: {np.mean(az)}")
    print(ax[:1000:10])
    print(ay[:1000:10])
    print(az[:1000:10])
    #{
    axu, axc = np.unique(ax, return_counts=True)
    ayu, ayc = np.unique(ay, return_counts=True)
    azu, azc = np.unique(az, return_counts=True)

    plt.bar(axu, axc)
    plt.show()
    plt.bar(ayu, ayc)
    plt.show()
    plt.bar(azu, azc)
    plt.show()
    }#
    
    return


def linear_regression(x1, x2, x3, y, itr):
    # 将输入张量转换为列向量
    x1 = x1.view(-1, 1)
    x2 = x2.view(-1, 1)
    x3 = x3.view(-1, 1)
    y = y.view(-1, 1)

    # 定义模型参数
    w1 = torch.randn(1, requires_grad=True)
    w2 = torch.randn(1, requires_grad=True)
    w3 = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD([w1, w2, w3, b], lr=0.01)

    # 训练模型
    for epoch in range(itr):
        # 前向传播
        y_pred = torch.matmul(x1, w1) + torch.matmul(x2, w2) + torch.matmul(x3, w3) + b

        # 计算损失函数
        loss = criterion(y_pred, y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 返回模型参数
    return w1, w2, w3, b
    return w1.item(), w2.item(), w3.item(), b.item()

def learn(itr):
    with open(".\data_ori_n_acc.txt", '+r') as f:
        raw = f.readlines()
    data = []
    for line in raw:
        newLine = line.strip('\n').split(' ')
        fline = []
        for i in newLine:
            fline.append(float(i))
        data.append(fline)

    x, y, z, a, b, c = [], [], [], [], [], []
    for i in data:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
        a.append(i[3])
        b.append(i[4])
        c.append(i[5])
    x = torch.tensor(x, requires_grad=True)
    y = torch.tensor(y, requires_grad=True)
    z = torch.tensor(z, requires_grad=True)
    a = torch.tensor(a, requires_grad=True)
    b = torch.tensor(b, requires_grad=True)
    c = torch.tensor(c, requires_grad=True)
    print(linear_regression(x, y, z, a, itr))
    return
'''