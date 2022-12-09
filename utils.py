from dataclasses import dataclass
import torch
from tqdm import tqdm
from config import *
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import gc
import torch.nn as nn
from kcenterGreedy import *
from kcenterGreedy import kCenterGreedy
from prepare_data import FOLD

##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


##
# Train Utils
iters = 0

#
def train_epoch(models, criterion, optimizers, dataloaders, epoch,  vis=None, plot_data=None):
    models['backbone'].train()
    models['module'].train()
    train_steps = len(dataloaders['train'])
    global iters
    backbone_running_loss = 0.0
    module_running_loss = 0.0
    total_running_loss = 0.0
    iters +=1
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].to("cuda:1")
        labels = data[1].to("cuda:1")
        labels = labels.long()

        input1 = inputs[:,:167]
        input2 = inputs[:,167:2215]
        input3 = inputs[:,2215:4263]
        input4 = inputs[:,4263:6311]

        input2= input2.unsqueeze(1)
        input3= input3.unsqueeze(1)
        input4= input4.unsqueeze(1)

        input2 = torch.cat([input2, input3, input4],dim=1)
        input5 = inputs[:,6311:]
        input1 = input1.to("cuda:1")
        input2 = input2.to("cuda:1")
        input5 = input5.to("cuda:1")

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores,_,features = models['backbone'](input1,input2,input5)
        target_loss = criterion(scores, labels)

        # if epoch > 150:
        #     # After 200 epochs, stop the gradient from the loss prediction module propagated to the target model.
        #     features[0] = features[0].detach()
        #     features[1] = features[1].detach()

        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
        loss            = m_backbone_loss + WEIGHT * m_module_loss

        # m_backbone_loss.backward(retain_graph=True)
        # m_module_loss.backward()

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()

        backbone_running_loss += m_backbone_loss
        module_running_loss += m_module_loss
        total_running_loss += loss

    backbone_running_loss = backbone_running_loss / train_steps
    module_running_loss = module_running_loss / train_steps
    total_running_loss = total_running_loss / train_steps
    # print(backbone_running_loss)

    
        

def train_epoch_Baselines(models, criterion, optimizers, dataloaders, epoch,  vis=None, plot_data=None):
    models['backbone'].train()
    train_steps = len(dataloaders['train'])
    global iters
    backbone_running_loss = 0.0
    module_running_loss = 0.0
    total_running_loss = 0.0
    iters +=1

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].to("cuda:1") # torch.Size([64, 1048])
        # print(inputs.shape)

        input1 = inputs[:,:167]
        input2 = inputs[:,167:2215]
        input3 = inputs[:,2215:4263]
        input4 = inputs[:,4263:6311]

        input2= input2.unsqueeze(1)
        input3= input3.unsqueeze(1)
        input4= input4.unsqueeze(1)

        input2 = torch.cat([input2, input3, input4],dim=1)
        input5 = inputs[:,6311:]
        input1 = input1.to("cuda:1")
        input2 = input2.to("cuda:1")
        input5 = input5.to("cuda:1")

        labels = data[1].to("cuda:1")
        labels = labels.long()

        optimizers['backbone'].zero_grad()

        scores, _, _  = models['backbone'](input1, input2, input5)
        target_loss = criterion(scores, labels)

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        loss            = m_backbone_loss

        loss.backward()
        optimizers['backbone'].step()

        backbone_running_loss += loss

    backbone_running_loss = backbone_running_loss / train_steps
    # print(backbone_running_loss)

#
def test(models, dataloaders, mode='val'):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    if METHOD == 'lloss':
        models['module'].eval()

    prob_all = []
    label_all = []

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.to("cuda:1")
            labels = labels.to("cuda:1")
            labels = labels.long()

            input1 = inputs[:,:167]
            input2 = inputs[:,167:2215]
            input3 = inputs[:,2215:4263]
            input4 = inputs[:,4263:6311]

            input2= input2.unsqueeze(1)
            input3= input3.unsqueeze(1)
            input4= input4.unsqueeze(1)

            input2 = torch.cat([input2, input3, input4],dim=1)
            input5 = inputs[:,6311:]
            input1 = input1.to("cuda:1")
            input2 = input2.to("cuda:1")
            input5 = input5.to("cuda:1")

            scores, _, _ = models['backbone'](input1, input2, input5)

            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            # prob_all.extend(scores[:,1].cpu().numpy())
            prob_all.extend(torch.nn.functional.softmax(scores.data,dim=1)[:,1].cpu().numpy())
            label_all.extend(labels.cpu().numpy())
            

            TP += ((preds == 1) & (labels == 1)).cpu().sum()
            # TN predict  label 0
            TN += ((preds == 0) & (labels == 0)).cpu().sum()
            # FN predict 0 label 1
            FN += ((preds == 0) & (labels == 1)).cpu().sum()
            # FP predict 1 label 0
            FP += ((preds == 1) & (labels == 0)).cpu().sum()
            
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            f1 = 2 * r * p / (r + p)
            f1 = f1.numpy()


    return 100 * correct / total , 100 * f1

#
def train(models, criterion, optimizers, schedulers, dataloaders,cycle,trial):
    print('>> Train a Model.')
    best_acc = 0
    best_F1 = 0
    best_auc = 0
    checkpoint_dir = os.path.join('./model_weights', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(EPOCH):
        if METHOD == 'TOX-AL':
            train_epoch(models, criterion, optimizers, dataloaders, epoch)
        else:
            train_epoch_Baselines(models, criterion, optimizers, dataloaders, epoch)

        # Save a checkpoint
        if True:
            acc, f1 = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
                best_F1 = f1
                if METHOD == 'TOX-AL':
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict_backbone': models['backbone'].state_dict(),
                        'state_dict_module': models['module'].state_dict()
                    },
                    '%s/Trial{}_cycle_{}_fold{}_DNNModel_TOX-AL.pth'.format(trial+1, cycle+1, FOLD) % (checkpoint_dir))
                else:
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict_backbone': models['backbone'].state_dict(),
                    },
                    '%s/Trial{}_cycle_{}_fold{}_DNNModel_{}.pth'.format(trial+1, cycle+1, FOLD, METHOD) % (checkpoint_dir))
            # if epoch % 50 == 0:
            #     print('学习率：{}'.format(optimizers['backbone'].state_dict()['param_groups'][0]['lr']))
            # print('Cycle:', cycle+1, 'Epoch:', epoch, '---', 'Val Acc: {:.2f} \t Best Acc: {:.2f}'.format(acc, best_acc), flush=True)

        # schedulers['backbone'].step()
        # if METHOD == 'lloss':
        #     schedulers['module'].step()

    # params = list(models['backbone'].named_parameters())
    # print(params[0])

    print('>> Finished.')

    return best_acc,best_F1,best_auc



def get_uncertainty(models, unlabeled_loader,cycle, checkpoint_dir,trial):
    # if cycle == 0:
    #     print('TOX-AL')
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).to("cuda:1")

    model_parameters = torch.load('%s/Trial{}_cycle_{}_fold{}_DNNModel_TOX-AL.pth'.format(trial+1, cycle+1, FOLD) % (checkpoint_dir))
    models['backbone'].load_state_dict(model_parameters['state_dict_backbone'])
    models['module'].load_state_dict(model_parameters['state_dict_module'])

    # params = list(models['backbone'].named_parameters())
    # print(params[0])
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(unlabeled_loader):
            inputs = inputs.to("cuda:1")
            # labels = labels.cuda()
            # if idx == 0:
            #     print(idx, inputs)

            input1 = inputs[:,:167]
            input2 = inputs[:,167:2215]
            input3 = inputs[:,2215:4263]
            input4 = inputs[:,4263:6311]

            input2= input2.unsqueeze(1)
            input3= input3.unsqueeze(1)
            input4= input4.unsqueeze(1)

            input2 = torch.cat([input2, input3, input4],dim=1)
            input5 = inputs[:,6311:]
            input1 = input1.to("cuda:1")
            input2 = input2.to("cuda:1")
            input5 = input5.to("cuda:1")

            scores, _, features = models['backbone'](input1, input2, input5)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()


def get_kcg(models, labeled_data_size, unlabeled_loader,NUM_SUBSET,cycle, checkpoint_dir,trial):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()

    model_parameters = torch.load('%s/Trial{}_cycle_{}_fold{}_DNNModel_{}.pth'.format(trial+1, cycle+1, FOLD, METHOD) % (checkpoint_dir))
    models['backbone'].load_state_dict(model_parameters['state_dict_backbone'])

    with torch.no_grad():
        for inputs, _, in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()

                input1 = inputs[:,:167]
                input2 = inputs[:,167:2215]
                input3 = inputs[:,2215:4263]
                input4 = inputs[:,4263:6311]

                input2= input2.unsqueeze(1)
                input3= input3.unsqueeze(1)
                input4= input4.unsqueeze(1)

                input2 = torch.cat([input2, input3, input4],dim=1)
                input5 = inputs[:,6311:]
                input1 = input1.to("cuda:0")
                input2 = input2.to("cuda:0")
                input5 = input5.to("cuda:0")
            _, features_batch, _ = models['backbone'](input1, input2, input5)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        new_av_idx = np.arange(NUM_SUBSET,(NUM_SUBSET + labeled_data_size))
        sampling = kCenterGreedy(feat)  
        batch = sampling.select_batch_(new_av_idx, ADDENDUM)
        other_idx = [x for x in range(NUM_SUBSET) if x not in batch]
    return  other_idx + batch


def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD


def train_vaal(models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle, NUM_SUBSET):
    
    vae = models['vae']
    discriminator = models['discriminator']
    vae.train()
    discriminator.train()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        vae = vae.cuda()
        discriminator = discriminator.cuda()
    
    adversary_param = 1
    beta          = 1
    num_adv_steps = 1
    num_vae_steps = 2

    bce_loss = nn.BCELoss()
    
    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)

    # train_iterations = int( (ADDENDUM*cycle+ NUM_SUBSET) * EPOCHV / BATCH )
    train_iterations = 300

    for iter_count in range(train_iterations):
        labeled_imgs, labels = next(labeled_data) # torch.Size([64, 1048])
        unlabeled_imgs = next(unlabeled_data)[0]

        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            labeled_imgs = labeled_imgs.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()
            labels = labels.cuda()

        # VAE step
        for count in range(num_vae_steps): # num_vae_steps
            recon, _, mu, logvar = vae(labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, beta)
        
            labeled_preds = discriminator(mu) # torch.Size([64, 1])
            unlabeled_preds = discriminator(unlab_mu) # unlabeled_preds
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_real_preds = unlab_real_preds.cuda()

            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            
            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_vae_steps - 1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(labeled_imgs)
                _, _, unlab_mu, _ = vae(unlabeled_imgs)
            
            labeled_preds = discriminator(mu)
            print(labeled_preds)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_fake_preds = unlab_fake_preds.cuda()
            
            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

            optimizers['discriminator'].zero_grad()
            dsc_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_adv_steps-1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()
            # if iter_count % 100 == 0:
                # print("Iteration: " + str(iter_count) + "  vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " +str(dsc_loss.item()))

