
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import mmread as load
import configparser as cfg
from sklearn import preprocessing
from os import makedirs
from os.path import exists
import sys
import progressbar



# 设置参数
k = 10
# epochs = 1000 比较好
epochs = 1000
learning_rate = 0.3
train_data = 'edm&ipsj/mooc_data/Data/mooc.train.rating'
matrix_dir = 'edm&ipsj/stu_nets_EXT'
up_dir = 'edm&ipsj/SP_EXT'

if not exists(up_dir):
    makedirs(up_dir)

# 读取数据集
df = pd.read_csv(train_data, sep='\t', names=['user', 'item', 'rating'], header=None, encoding='utf-8')
users = list(df.user.unique())
num_users = len(users)


# 定义 PyTorch 模型
class MyModel(nn.Module):
    def __init__(self, num_input, num_features):
        super(MyModel, self).__init__()
        self.W1 = nn.Parameter(torch.randn(num_input, num_features) * 0.01)
        self.W2 = nn.Parameter(torch.randn(num_features, num_input) * 0.01)

    def forward(self, X, mask, return_features=False):
        hidden = torch.sigmoid(torch.matmul(X, self.W1 * mask))
        if return_features:
            return hidden
        output = torch.sigmoid(torch.matmul(hidden, self.W2 * mask.t()))
        return output


# 创建一个progressbar实例
bar = progressbar.ProgressBar(max_value=num_users)
bar.start()

processed_user = 0

# 训练模型
for user in users:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载用户特定数据
    # 这里需要根据实际情况加载每个用户的 matrix 和 mask
    matrix = load(matrix_dir + '/{}/matrix.mtx'.format(user)).todense().astype(np.float32)
    mask = load(matrix_dir + '/{}/mask.mtx'.format(user)).todense().astype(np.float32)
    matrix = torch.tensor(matrix)
    mask = torch.tensor(mask)
    matrix, mask = matrix.to(device), mask.to(device)
    

    with open(matrix_dir + '/{}/features'.format(user), encoding='utf-8') as f:
        features = list(f.read().splitlines())
    
    # 用户特定的模型实例
    num_input = matrix.shape[1]
    num_features = mask.shape[1]

    # print(mask.shape, num_input, num_features)

    model = MyModel(num_input, num_features).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)

    # 训练循环
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(matrix, mask)
        loss = criterion(outputs, matrix)
        loss.backward()
        optimizer.step()

    print(f'User {user} - Epoch {epoch}: Loss {loss.item()}')



    # 在训练结束后提取特征
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 关闭梯度计算
        hidden_features = model(matrix, mask, return_features=True).cpu().numpy()
    
    hidden_features = hidden_features.squeeze()
    if hidden_features.ndim == 0:
        hidden_features = np.array([hidden_features])
    

    # 创建特征字典
    up = {}
    for i, v in enumerate(hidden_features):
        # print(features[i],v)
        up[features[i]] = v

    # 排序并保存特征
    s = [(k, up[k]) for k in sorted(up, key=up.get, reverse=True)]
    
    with open(f"{up_dir}/{user}.tsv", "w", encoding='utf-8') as file:
        for k, v in s:
            file.write("{}\t{:.16f}\n".format(k, v))

    # 更新progressbar
    processed_user += 1
    bar.update(processed_user)

    # 重置模型为训练模式
    model.train()

# 完成后关闭progressbar
bar.finish()
