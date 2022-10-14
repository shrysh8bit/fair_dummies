
import torch
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
import os
################# Creating training dataset class #################

class AdultData(torch.utils.data.Dataset):

  def __init__(self, X, y, scale_data=True):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i], i



class Dataset(object):

    def __init__(self):

        adult_data = pd.read_csv('adult.csv', na_values='?')
        adult_data = adult_data.dropna()
        encoder = LabelEncoder()
        adult_data.income = adult_data.income.replace('<=50K', 0)
        adult_data.income = adult_data.income.replace('>50K', 1)
        adult_data['workclass']=encoder.fit_transform(adult_data['workclass'])
        adult_data['education']=encoder.fit_transform(adult_data['education'])
        adult_data['marital.status']=encoder.fit_transform(adult_data['marital.status'])
        adult_data['occupation']=encoder.fit_transform(adult_data['occupation'])
        adult_data['relationship']=encoder.fit_transform(adult_data['relationship'])
        adult_data['race']=encoder.fit_transform(adult_data['race'])
        adult_data['sex']=encoder.fit_transform(adult_data['sex'])
        adult_data['native.country']=encoder.fit_transform(adult_data['native.country'])
        #adult_data['income']=encoder.fit_transform(adult_data['income'])


        X = adult_data.iloc[:, :-1].values
        y = adult_data.iloc[:, [-1]].values

        #imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

        #X[:, :-1] = imputer.fit_transform(X[:, :-1])
        #y = imputer.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        #y_train = scaler.fit_transform(y_train)
        #y_test = scaler.transform(y_test)

        self.trainset = AdultData(X_train, y_train)
        self.testset = AdultData(X_test, y_test)


    def load_data(self):

        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size = 10, shuffle=False, num_workers=2)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=10, shuffle=False, num_workers=2)

        return trainloader, testloader, self.trainset

import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(14,64) 
    self.activ1 = nn.ReLU()
    self.layer2 = nn.Linear(64, 32)
    self.activ2 = nn.ReLU()
    self.layer3 = nn.Linear(32,2)


  def forward(self, x):
    out = self.activ1(self.layer1(x))
    out = self.activ2(self.layer2(out))
    out = self.layer3(out)

    return out

import torch.optim as optim
torch.manual_seed(123)

net = Network()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0005, weight_decay=0.01)

ds = Dataset()
trainloader, testloader, trainset = ds.load_data()

'''
dataiter = iter(trainloader)
input, labels, i = dataiter.next()
print(type(input))
print(input)
print(labels)
print(i.shape)
'''
best_acc = 0
for epoch in range(30):
  running_loss = 0
  correct = 0
  for i, data in enumerate(trainloader):
    inputs, targets, _ = data
    
    optimizer.zero_grad()
    outputs = net(inputs.float())
    loss = criterion(outputs, targets.squeeze())
    #print(loss)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    
    _, idx = torch.max(outputs,dim=-1)

    correct += (idx == targets.squeeze()).sum()


  print(f'Epoch: {epoch} => loss: {running_loss / len(trainset):.3f}')
  running_loss = 0
  acc = 100.*correct/len(trainset)
  print("Acc:", acc)
  if acc > best_acc:
    print('Saving..')
    
    if not os.path.isdir('checkpoint_sub'):
      os.mkdir('checkpoint_sub')
    torch.save(net.state_dict(), './checkpoint_sub/adult.pth')
    best_acc = acc


net = Network()
net.load_state_dict(torch.load('./checkpoint_sub/adult.pth'))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        inputs, labels, _ = data
        outputs = net(inputs.float())

        total += labels.size(0)
        _, idx = torch.max(outputs,dim=-1)
        correct += (idx == labels.squeeze()).sum()
print(f'Accuracy of the network : {100 * correct / total} %')
