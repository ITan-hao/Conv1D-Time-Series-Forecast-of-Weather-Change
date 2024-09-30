import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg') #Qt5Agg  TkAgg


in_channels = 14
out_channels = 10
kernel_size = 3
stride = 2
padding = 0
timestep = 5  # 时间步长，就是利用多少时间窗口
batch_size = 16  # 批次大小
input_dim = 14  # 每个步长对应的特征数量
epochs = 10
best_loss = 0
save_path = './best_model.pth'
# 1.加载时间序列数据
df = pd.read_csv('./data.csv', index_col=0)
# 2.将数据进行标准化
scaler = StandardScaler()
#StandardScaler 是 sklearn.preprocessing 模块中的一个类，用于执行特征缩放（也称为标准化）。
# 它通过减去每个特征的平均值并除以标准差，使数据具有零均值和单位方差。这样做的目的是为了规范化数据，消除不同特征之间的量纲影响，提高模型训练的效率。
scaler_model = StandardScaler()
data = scaler_model.fit_transform(np.array(df))
#np.array(df)将DataFrame转换成numpy数组，因为fit_transform方法期望输入是numpy数组。
# fit_transform函数首先对整个数据集进行拟合（学习数据的分布），然后返回标准化后的数据。
scaler.fit_transform(np.array(df['T (degC)']).reshape(-1, 1))
#对于单个特征，如列’T (degC)'，我们同样使用fit_transform，但先将其提取出来，转化为形状为(n_samples, 1)的一维数组，
# 这是因为StandardScaler期望二维输入。reshape(-1, 1)用于创建一个具有适当维度的新数组，-1表示自动计算第一个轴的大小以适应数据。

# 形成训练数据，例如12345变成12-3，23-4，34-5
def split_data(data, timestep, input_dim):
    dataX = []  # 保存X
    dataY = []  # 保存Y
    # 将整个窗口的数据保存到X中，将未来一天保存到Y中
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep])#对于每一个索引 index，它会从 data 中提取从该索引开始，长度为 timestep 的子序列（切片），并将这个子序列添加到 dataX 列表中
        dataY.append(data[index + timestep][1])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    # 获取训练集大小
    train_size = int(np.round(0.8 * dataX.shape[0]))
    # 划分训练集、测试集
    x_train = dataX[: train_size, :].reshape(-1, timestep, input_dim)#-1用于自动计算新形状的第一个维度，以保持总元素数量不变
    #创建一个包含timestep行和input_dim列的新矩阵，每一行都是原数据集中的一段时间序列
    y_train = dataY[: train_size]
    x_test = dataX[train_size:, :].reshape(-1, timestep, input_dim)
    y_test = dataY[train_size:]
    return [x_train, y_train, x_test, y_test]


# 3.获取训练数据   x_train: 1700,1,4
x_train, y_train, x_test, y_test = split_data(data, timestep, input_dim)
# 4.将数据转为tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)
# 5.形成训练数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor) #用于组合张量数据集，方便在深度学习模型中一起传入
test_data = TensorDataset(x_test_tensor, y_test_tensor)
# 6.将数据加载成迭代器 迭代器允许逐批读取数据，使得模型可以高效地处理连续流式的数据，而不需要一次性加载全部数据到内存。
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size,
                                           True)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size,
                                          False)


# 7.定义一维卷积网络
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.maxpool1 = nn.AdaptiveMaxPool1d(output_size=20)

        self.conv2 = nn.Conv1d(in_channels=10,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.maxpool2 = nn.AdaptiveAvgPool1d(output_size=15)

        self.conv3 = nn.Conv1d(in_channels=10,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.maxpool3 = nn.AdaptiveAvgPool1d(output_size=10)

        self.fc = nn.Linear(out_channels * 10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)

        x = x.reshape(-1, x.shape[1] * x.shape[2])
        x = self.fc(x)
        return x


model = CNN(in_channels, out_channels, kernel_size, stride, padding)  # 定义卷积网络
loss_function = nn.MSELoss()  # 定义损失函数
#回归损失函数，计算的是预测值与真实值之间差的平方的平均数。在训练时，模型会尝试最小化这个损失，以提高预测精度。
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 定义优化器
# 8.模型训练
for epoch in range(epochs):
    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        # x_train的输入维度为【批次，嵌入维度，序列长度】
        x_train, y_train = data  # 解包迭代器中的X和Y
        optimizer.zero_grad()
        y_train_pred = model(x_train.transpose(1, 2))
        loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        #desc 是 train_bar 的描述属性，用于显示训练过程中的信息。
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)
    # 模型验证
    model.eval()
    test_loss = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x_test, y_test = data
            y_test_pred = model(x_test.transpose(1, 2))#transpose用于交换两个特定的维度
            test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), save_path)
print('Finished Training')

# 9.绘制结果
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform((model(x_train_tensor[:10000].transpose(1, 2)).detach().numpy()).reshape(-1, 1)), "b")
plt.plot(scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)), "r")
plt.legend()# 添加图例
plt.show()
y_test_pred = model(x_test_tensor.transpose(1, 2))
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform(y_test_pred.detach().numpy()), "b")
plt.plot(scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)), "r")
plt.legend()
plt.show()