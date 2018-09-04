'''
@Author 王**
@Team 生男孩48
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.model_selection import RepeatedKFold
import time
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score

N_INPUT = 30  # 特征维度
N_HIDDEN = 20  # 隐层维度
BATCH_SIZE = 8  # 批量大小

# 网络结构
net = torch.nn.Sequential(
    torch.nn.Linear(N_INPUT, N_HIDDEN),
        torch.nn.Dropout(0.5),           # drop 50% neurons
        torch.nn.LeakyReLU(),
        torch.nn.Linear(N_HIDDEN, N_HIDDEN),
        torch.nn.Dropout(0.5),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(N_HIDDEN, 2),
)

# 数据集读取
class InsuranceDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return np.float32(self.x[item]), np.long(self.y[item])

    def __len__(self):
        return len(self.x)


def train():

    # net = Net()
    print(net)

    # 十折交叉验证， 重复十次
    kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=int(time.time()))

    # data
    data = np.genfromtxt('5.csv', delimiter=',')
    X = data[:, :-1]
    Y = data[:, -1]

    optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
    loss_func = nn.CrossEntropyLoss()

    validate_loss_final = 0.0

    for train_index, test_index in kf.split(X):

        X_train = X[train_index]
        X_validate = X[test_index]
        Y_train = Y[train_index]
        Y_validate = Y[test_index]

        train_dataset = InsuranceDataSet(X_train, Y_train)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

        for epoch in range(30):

            # train
            net.train()
            for i, train_data in enumerate(train_loader, 0):
                features, label = train_data
                features = Variable(features)
                label = Variable(label)

                prediction = net(features)

                # print('output size is {}'.format(prediction.size()))

                loss = loss_func(prediction, label)

                # 优化及反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 验证
            net.eval()

            validate_features = Variable(torch.from_numpy(X_validate).type(torch.FloatTensor))
            validate_predictions = net(validate_features).detach().numpy()


            result = validate_predictions[:, 0]-validate_predictions[:, 1]
            result_bool = result<0
            result_bool = result_bool.astype('int')

            validate_loss = f1_score(Y_validate, result_bool)

            print('f1 is {}'.format(validate_loss))
            if epoch == 0 or validate_loss > validate_loss_final:
                torch.save(net.state_dict(), 'Net-round-{}.pth'.format(epoch))
                validate_loss_final = validate_loss

        print('Finish training...')
        print('best is {}'.format(validate_loss_final))
        torch.save(net.state_dict(), 'Net.pth')


def test():
    return


if __name__ == '__main__':
    train()
