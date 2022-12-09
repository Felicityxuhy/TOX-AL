'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

import os
import random
from sched import scheduler
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10
from tqdm import tqdm
from models.DNN import *
import models.resnet_1D as resnet
from models.lossnet import *
from config import *
from data.sampler import SubsetSequentialSampler
from utils import *
from prepare_data import *
from models.VAAL import *


# def same_seeds(seed):
#     random.seed(seed)
#     torch.manual_seed(seed)  # 固定随机种子（CPU）
#     if torch.cuda.is_available():  # 固定随机种子（GPU)
#         torch.cuda.manual_seed(seed)  # 为当前GPU设置
#         torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
#     np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
#     torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
#     torch.backends.cudnn.deterministic = True  # 固定网络结构

# same_seeds(2022)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
##
# Main
if __name__ == '__main__':

    checkpoint_dir = os.path.join('./model_weights', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for trial in range(TRIALS):
        total_best_acc = []
        total_best_f1 = []

        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        print(indices[:5])

        labeled_set = indices[:200]
        np.savetxt('results/random_start200/Trial{}_{}_fold{}_random_start200.txt'.format(trial+1,METHOD,FOLD),labeled_set,fmt='%d')
        unlabeled_set = [x for x in indices if x not in labeled_set]

        # unlabeled_dict = {}
        # unlabeled_dict = {idx: k for idx, k in enumerate(unlabeled_set)}


        train_loader = DataLoader(train_set, batch_size=BATCH, 
                                  sampler=SubsetRandomSampler(labeled_set), 
                                  pin_memory=True,shuffle=False)

        test_loader  = DataLoader(test_set, batch_size=BATCH,shuffle=False)

        dataloaders  = {'train': train_loader, 'test': test_loader}

        labeled_file = []

        # Active learning cycles
        for cycle in range(CYCLES):
            # Model
            # same_seeds(2022)
            CNNmodel = CNNModel_merge().to("cuda:1")

            if METHOD == 'TOX-AL':
                loss_module = LossNet_v1005().to("cuda:1")

            if METHOD == 'TOX-AL':
                models      = {'backbone': CNNmodel, 'module': loss_module}
            else:
                models      = {'backbone': CNNmodel}

            if cycle == 0:
                params = list(models['backbone'].named_parameters())
                print(params[0])

            # print('unlabeled_set:{}',unlabeled_set[:10])

            NUM_SUBSET = len(unlabeled_set)
            subset = unlabeled_set[:NUM_SUBSET]

            criterion = nn.CrossEntropyLoss(reduction='none')

            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                        momentum=MOMENTUM, weight_decay=WDECAY)
            sched_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optim_backbone,T_max =  EPOCH )

            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}
            
            if METHOD == 'TOX-AL':
                # optim_module   = optim.SGD(models['module'].parameters(), lr=LR_module, 
                #                         momentum=MOMENTUM, weight_decay=WDECAY)
                optim_module   = optim.Adam(models['module'].parameters(), lr=8e-3)

                sched_module = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optim_module,T_max =  EPOCH)
                # sched_module = torch.optim.lr_scheduler.StepLR(optimizer = optim_module, step_size=120, gamma=0.5)

                optimizers = {'backbone': optim_backbone, 'module': optim_module}
                schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # if cycle == 0:
            #     epoch = 150
            # else:
            #     epoch = EPOCH

            best_acc, best_F1, best_auc = train(models, criterion, optimizers, schedulers, dataloaders,cycle,trial)

            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {} Test F1 {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), best_acc,best_F1))
            total_best_acc.append(best_acc)
            total_best_f1.append(best_F1)

            """
            Query Method: TOX-AL & Random
            """
            if METHOD == 'TOX-AL':

                print('TOX-AL')
                unlabeled_loader = DataLoader(unlabeled_data, batch_size=BATCH, 
                                            sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
                                            pin_memory=True)

                # # Measure uncertainty of each data points in the subset
                uncertainty = get_uncertainty(models, unlabeled_loader, cycle, checkpoint_dir,trial)

                # argsort
                arg = np.argsort(uncertainty)

                # print(torch.tensor(unlabeled_set)[-ADDENDUM:])

                # map
                # before_map_value = torch.tensor(subset)[arg][-ADDENDUM:].numpy()
                # a_map = [unlabeled_dict[ele] if ele in unlabeled_dict else ele for ele in before_map_value]

                labeled_file.append(list(torch.tensor(subset)[arg][-ADDENDUM:].numpy()))

                # print(torch.tensor(subset)[arg][-ADDENDUM:].numpy())

                labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
                unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[NUM_SUBSET:]


            # hybrid
            # elif METHOD == 'hybrid':
            #     entropy = torch.tensor([]).cuda()
            #     unlabeled_loader = DataLoader(unlabeled_data, batch_size=BATCH, 
            #                                 sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
            #                                 pin_memory=True)

            #     # # Measure uncertainty of each data points in the subset
            #     uncertainty = get_uncertainty(models, unlabeled_loader, cycle, checkpoint_dir)

            #     # loss采样取的参数值

            #     with torch.no_grad():
            #         for (inputs, labels) in unlabeled_loader:
            #             inputs = inputs.cuda()
            #             labels = labels.cuda()
            #             labels = labels.long()

            #             input1 = inputs[:,:167]
            #             input2 = inputs[:,167:2215]
            #             input3 = inputs[:,2215:4263]
            #             input4 = inputs[:,4263:6311]

            #             input2= input2.unsqueeze(1)
            #             input3= input3.unsqueeze(1)
            #             input4= input4.unsqueeze(1)

            #             input2 = torch.cat([input2, input3, input4],dim=1)
            #             input5 = inputs[:,6311:]
            #             input1 = input1.to("cuda:0")
            #             input2 = input2.to("cuda:0")
            #             input5 = input5.to("cuda:0")

            #             scores, _,_ = models['backbone'](input1, input2, input5)
            #             probs = F.softmax(scores, dim=1)
            #             log_probs = torch.log(probs)
            #             uncertainties = (probs*log_probs).sum(1)
            #             entropy = torch.cat((entropy, uncertainties), 0)

            #     # 标准化
            #     uncertainty = np.array(uncertainty, dtype=float)    
            #     uncertainty = (uncertainty - np.mean(uncertainty))/ np.std(uncertainty) 
            #     entropy = np.array(entropy.cpu(), dtype=float)  
            #     entropy = (entropy - np.mean(entropy))/ np.std(entropy) 
                 
            #     arg1 = np.argsort(uncertainty-entropy)
                
            #     # print(torch.tensor(unlabeled_set)[-ADDENDUM:])

            #     # loss采样
            #     labeled_set += list(torch.tensor(subset)[arg1][-ADDENDUM:].numpy())
            #     unlabeled_set = list(torch.tensor(subset)[arg1][:-ADDENDUM].numpy()) + unlabeled_set[NUM_SUBSET:]
            

            elif METHOD == 'random':
                random.shuffle(unlabeled_set)
                if cycle == 0:
                    print('random')
                    print(unlabeled_set[0:5])
                unlabeled_loader = DataLoader(unlabeled_data, batch_size=BATCH, 
                                            sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
                                            pin_memory=True)


                labeled_file.append(list(torch.tensor(unlabeled_set)[-ADDENDUM:].numpy()))
                # random sampling
                labeled_set += list(torch.tensor(unlabeled_set)[-ADDENDUM:].numpy())
                unlabeled_set = list(torch.tensor(unlabeled_set)[:-ADDENDUM].numpy()) + unlabeled_set[NUM_SUBSET:]


            # Query Method: VAAL
            elif False:
                # Create unlabeled dataloader for the unlabeled subset
                unlabeled_loader = DataLoader(unlabeled_data, batch_size=BATCH, 
                                            sampler=SubsetSequentialSampler(subset), 
                                            pin_memory=True)
                labeled_loader = DataLoader(unlabeled_data, batch_size=BATCH, 
                                            sampler=SubsetSequentialSampler(labeled_set), 
                                            pin_memory=True)

                vae = VAE()
                discriminator = Discriminator(32)
                models_VAAL      = {'vae': vae, 'discriminator': discriminator}
                
                optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
                optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
                optimizers = {'vae': optim_vae, 'discriminator':optim_discriminator}

                train_vaal(models_VAAL,optimizers, labeled_loader, unlabeled_loader, cycle+1, NUM_SUBSET)
                
                # all_preds, all_indices = [], []
                all_preds = []

                for images, _ in unlabeled_loader:                       
                    images = images.cuda()
                    with torch.no_grad():
                        _, _, mu, _ = vae(images)
                        preds = discriminator(mu)

                    preds = preds.cpu().data
                    all_preds.extend(preds)
                    # all_indices.extend(indices)

                all_preds = torch.stack(all_preds)
                all_preds = all_preds.view(-1)
                # need to multiply by -1 to be able to use torch.topk 
                all_preds *= -1
                # select the points which the discriminator things are the most likely to be unlabeled
                _, arg = torch.sort(all_preds) 

                # Update the labeled dataset and the unlabeled dataset, respectively
                labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
                listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) 
                unlabeled_set = listd + unlabeled_set[NUM_SUBSET:]

        three_results = pd.DataFrame({'acc':total_best_acc,'F1':total_best_f1})
        three_results.to_csv('results/acc_f1_auc/Trial{}_{}_fold{}_results.csv'.format(trial+1,METHOD,FOLD),index = None,encoding = 'utf8')
        np.savetxt('results/labeled_file_aftermapping/Trial{}_{}_fold{}_labeled_file.txt'.format(trial+1,METHOD,FOLD),labeled_file,fmt='%d')

