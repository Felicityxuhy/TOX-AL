import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from config import *
from data.sampler import SubsetSequentialSampler


# read data
data_MACCS_origin = pd.read_csv('data/Genotoxicity_Ames-MACCS.csv',index_col=0)
data_MACCS = data_MACCS_origin.iloc[:,7:].astype('float32')

data_ECFP2_origin = pd.read_csv('data/Genotoxicity_Ames-ECFP2.csv',index_col=0)
data_ECFP2 = data_ECFP2_origin.iloc[:,7:].astype('float32')

data_ECFP4_origin = pd.read_csv('data/Genotoxicity_Ames-ECFP4.csv',index_col=0)
data_ECFP4 = data_ECFP4_origin.iloc[:,7:].astype('float32')

data_ECFP6_origin = pd.read_csv('data/Genotoxicity_Ames-ECFP6.csv',index_col=0)
data_ECFP6 = data_ECFP6_origin.iloc[:,7:].astype('float32')

data_rdkit2d_origin = pd.read_csv('data/Genotoxicity_Ames-rdkit2d.csv',index_col=0)
data_rdkit2d = data_rdkit2d_origin.iloc[:,7:].astype('float32')

data_merge = pd.concat([data_MACCS,data_ECFP2,data_ECFP4,data_ECFP6,data_rdkit2d], axis=1)


label = data_MACCS_origin.iloc[:,6].astype('float32')


# data_merge = pd.concat([data_merge,label], axis=1)
# data_merge.head(5)

# pandas frame to tensor
data_array= np.array(data_merge)
data_tensor = torch.tensor(data_array)

label_array= np.array(label)
label_tensor = torch.tensor(label_array)

# dataset, dataset_1 = train_test_split(data_merge, test_size=0.2,random_state=2022)
# mid_dataset_1, mid_dataset_2 = train_test_split(dataset, test_size=0.5,random_state=2022)

# dataset_2, dataset_3 = train_test_split(mid_dataset_1, test_size=0.5,random_state=2022)
# dataset_4, dataset_5 = train_test_split(mid_dataset_2, test_size=0.5,random_state=2022)


# train_data = 'data/dataset_5.csv'
# dataset_5.to_csv(train_data, index=True, encoding='utf-8')


# # 构建Dataloader
torch_dataset = TensorDataset(data_tensor, label_tensor)

dataset, dataset_1 = train_test_split(torch_dataset, test_size=0.2,random_state=2022)
mid_dataset_1, mid_dataset_2 = train_test_split(dataset, test_size=0.5,random_state=2022)

dataset_2, dataset_3 = train_test_split(mid_dataset_1, test_size=0.5,random_state=2022)
dataset_4, dataset_5 = train_test_split(mid_dataset_2, test_size=0.5,random_state=2022)

train_set =  dataset_1 + dataset_2 + dataset_4 + dataset_5
test_set = dataset_3
unlabeled_data = train_set
FOLD = 3

# for idx, (x_train, y_label) in enumerate(train_set):
#     print(x_train, y_label)


# print(len(dataset_1))
# print(len(dataset_2))
# print(len(dataset_3))
# print(len(dataset_4))
# print(len(dataset_5))

# print(train_set[5007][1])





