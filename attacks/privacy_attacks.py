import os
from posixpath import join
import sys
import inspect

import math
from random import randrange
import pickle 
import copy
import numpy as np
import pandas as pd
import argparse
from collections import OrderedDict

from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

import torch
import torchvision
import torchvision.transforms as transforms 
import torchvision.models as models 
from torch.utils.data import Subset

from training.image_classification import train
from utils.masks import to_mask, evaluate_masks
from torch.nn import functional as F
from models import build_model
from utils.misc import bool_flag
from utils.masks import to_mask
from attacks.privacy_attacks import get_parser 


from opacus.grad_sample import GradSampleModule
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description='Privacy attack parameters')

    # config parameters
    parser.add_argument("--dump_path", type=str, default=None) # model saving location
    parser.add_argument('--print_freq', type=int, default=50) # training printing frequency
    parser.add_argument("--save_periodic", type=int, default=0) # training saving frequency

    # attack parameters 
    parser.add_argument("--model_path", type=str, default="model") # path to the private model
    parser.add_argument("--attack_type", type=str, default="loss") # type of auxiliary attack
    parser.add_argument("--aux_epochs", type=int, default=20) # number of auxiliary training epochs
    parser.add_argument("--num_aux", type=int, default=1) # number of auxiliary models 
    parser.add_argument("--aug_style", type=str, default="mean") # combination method for augmented data values 
    parser.add_argument("--aux_style", type=str, default="sum") # combination method for multiple aux. model values 
    parser.add_argument("--public_data", type=str, default="train") # specify which part of the public data to use for aux model training (e.g. train is the training mask, rand50 is a random selection of the public data) 
    parser.add_argument("--norm_type", type=str, default=None) # norm for gradient norm  
    parser.add_argument("--num_points", type=int, default=10) # number of points to use for the label-only attack
    parser.add_argument("--clip_min", type=float, default=0) # minimum value for adversarial feature in label-only attack
    parser.add_argument("--clip_max", type=float, default=1) # maximum value for adversarial feature in label-only attack

    # Data parameters
    parser.add_argument("--data_root", type=str, default="data") # path to the data
    parser.add_argument("--dataset", type=str, choices=["cifar10", "imagenet", "cifar100", "gaussian","credit", "hep", "adult", "mnist", "lfw"], default="cifar10")
    parser.add_argument("--mask_path", type=str, required=True) # path to the data mask
    parser.add_argument('--n_data', type=int, default=500) # specify number of data points for gaussian data
    parser.add_argument('--data_num_dimensions', type=int, default=75) # number of features for non-image data
    parser.add_argument('--random_seed', type=int, default=10) # seed for gaussian data 
    parser.add_argument("--num_classes", type=int, default=10) # number of classes for classification task 
    parser.add_argument("--in_channels", type=int, default=3) # number of input channels for image data

    # Model parameters
    parser.add_argument("--architecture", choices=["lenet", "smallnet", "resnet18", "kllenet","linear", "mlp"], default="lenet")
    
    # training parameters
    parser.add_argument("--aug", type=bool_flag, default=False) # data augmentation flag 
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument("--epochs", type=int, default=50) 
    parser.add_argument("--optimizer", default="sgd,lr=0.1,momentum=0.9")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_gradients", type=bool_flag, default=False) 
    parser.add_argument("--log_batch_models", type=bool_flag, default=False) # save model for each batch of data
    parser.add_argument("--log_epoch_models", type=bool_flag, default=False) # save model for each training epoch

    # privacy parameters
    parser.add_argument("--private", type=bool_flag, default=False) # privacy flag 
    parser.add_argument("--noise_multiplier", type=float, default=None)
    parser.add_argument("--privacy_epsilon", type=float, default=None)
    parser.add_argument("--privacy_delta", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    #multi gpu paramaeters
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--master_port", type=int, default=-1)
    parser.add_argument("--debug_slurm", type=bool_flag, default=False)

    return parser

def adult_data_transform(df):
    """
    transform adult data.
    """

    binary_data = pd.get_dummies(df)
    feature_cols = binary_data[binary_data.columns[:-2]]
    scaler = preprocessing.StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(feature_cols), columns=feature_cols.columns)
    return data

def get_dataset(params):
    """
    load data for privacy attacks
    """
    if params.dataset=='cifar10':
        if params.aug==True:
            print('Using data augmentation')
            augmentations = [transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()]
            normalize = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            model_transform = transforms.Compose(augmentations + normalize)
        else:
            print('Not using data augmentation')
            normalize = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            model_transform = transforms.Compose(normalize)
        return torchvision.datasets.CIFAR10(root=params.data_root, train=True, download=True, transform=model_transform)
    
    if params.dataset=='mnist':
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        return torchvision.datasets.MNIST(root=params.data_root, train=True, download=True, transform=transform)

    elif params.dataset=='imagenet':
        if params.aug==True:
            print('Using data augmentation to train model')
            augmentations = [transforms.Resize(256),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip()]
            normalize = [transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            transform = transforms.Compose(augmentations + normalize)
        else:
            print('Not using data augmentation to train model')
            transform = transforms.Compose( [transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
   
        dataset = torchvision.datasets.ImageFolder(root=params.data_root+'/train',transform=transform)
        
        return dataset
    elif params.dataset=='cifar100':
        if params.aug:
            augmentations = [transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()]
            normalize = [transforms.ToTensor(),transforms.Normalize(mean=[n/255 for n in [129.3, 124.1, 112.4]], std=[n/255 for n in [68.2,  65.4,  70.4]])]
            transform = transforms.Compose(augmentations + normalize)

        else:
            transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize(mean=[n/255 for n in [129.3, 124.1, 112.4]], std=[n/255 for n in [68.2,  65.4,  70.4]])])
        
        dataset = torchvision.datasets.CIFAR100(root=params.data_root, train=True, download=True, transform=transform)
        return dataset
    
    elif params.dataset=='credit':
        cred=fetch_openml('credit-g')
    
        data = SimpleImputer(missing_values=np.nan, strategy='mean', copy=True).fit(cred.data).transform(cred.data)
        target = preprocessing.LabelEncoder().fit(cred.target).transform(cred.target)   
        X=data
        norm = np.max(np.concatenate((-1*X.min(axis=0)[np.newaxis], X.max(axis=0)[np.newaxis]),axis=0).T, axis=1).astype('float32')
        data=np.divide(data,norm)

        data=torch.tensor(data).float()
        target=torch.tensor(target).long()
        
        ids=np.arange(1000)[:800]
        
        
        final_data = []
        for i in ids:
            final_data.append([data[i], target[i]])
        
        # norm=np.max
        params.num_classes = 2
        
        # dataloader = torch.utils.data.DataLoader(final_data, shuffle=True, batch_size=params.batch_size)
        # n_data=len(final_data)
        return final_data
    elif params.dataset=='hep':
        
        hep=fetch_openml('hepatitis')
    
        data = SimpleImputer(missing_values=np.nan, strategy='mean', copy=True).fit(hep.data).transform(hep.data)
        target = preprocessing.LabelEncoder().fit(hep.target).transform(hep.target)   
        
        X=data
        norm = np.max(np.concatenate((-1*X.min(axis=0)[np.newaxis], X.max(axis=0)[np.newaxis]),axis=0).T, axis=1).astype('float32')
        data=np.divide(data,norm)

        data=torch.tensor(data).float()
        target=torch.tensor(target).long()
    
        ids=np.arange(155)[:124]
        
        
        final_data = []
        for i in ids:
            final_data.append([data[i], target[i]])
        
        params.num_classes = 2

        return final_data
    elif params.dataset == 'adult':
        columns = ["age", "workClass", "fnlwgt", "education", "education-num","marital-status", "occupation", "relationship","race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
        train_data = pd.read_csv(params.data_root+'/adult.data', names=columns, sep=' *, *', na_values='?')
        test_data  = pd.read_csv(params.data_root+'/adult.test', names=columns, sep=' *, *', skiprows=1, na_values='?')

        original_train=train_data
        original_test=test_data
        num_train = len(original_train)
        original = pd.concat([original_train, original_test])
        labels = original['income']
        labels = labels.replace('<=50K', 0).replace('>50K', 1)
        labels = labels.replace('<=50K.', 0).replace('>50K.', 1)

        # Remove target 
        del original["income"]

        data = adult_data_transform(original)
        train_data = data[:num_train]
        train_labels = labels[:num_train]
        test_data = data[num_train:]
        test_labels = labels[num_train:]

        test_data=torch.tensor(test_data.to_numpy()).float()
        train_data=torch.tensor(train_data.to_numpy()).float()
        test_labels=torch.tensor(test_labels.to_numpy(dtype='int64')).long()
        train_labels=torch.tensor(train_labels.to_numpy(dtype='int64')).long()
        
        final_data = []
        for i in np.arange(len(train_data)):
            final_data.append([train_data[i], train_labels[i]])
        
        return final_data       

def get_uncalibrated_gradnorm(params, mask):
    """
    return uncalibrated gradient norm values for data indicated by the mask. 
    """
    #load the dataset
    dataset = get_dataset(params)
    #initialize to 0
    grad_norms=np.zeros(len(mask))

    #get the final model 
    final_model=build_model(params)
    final_model_path = os.path.join(params.model_path, "checkpoint.pth")
    state_dict_final = torch.load(final_model_path, map_location='cuda:0')
    if params.dataset=='imagenet':
        new_state_dict = OrderedDict()
        for k, v in state_dict_final["model"].items():
            if k[:7]=='module.': # remove `module.`
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k]=v
        final_model.load_state_dict(new_state_dict)
    else:
        final_model.load_state_dict(state_dict_final['model'])
    final_model=final_model.cuda()
    
    original_model=[]
    for p in final_model.parameters():
        original_model.append(p.view(-1))
    original_model=torch.cat(original_model)

    #get the appropriate ids to dot product
    ids=(mask==True).nonzero().flatten().numpy()

    #load 1-by-1. See get_calibrated_gradnorm for batched method using Opacus gradsamplemodule. 
    for id in ids:
        #load each image and target
        image = dataset[id][0].unsqueeze(0)
        image = image.cuda(non_blocking=True)
        target = torch.tensor(dataset[id][1]).unsqueeze(0)
        target = target.cuda(non_blocking=True)

        #reload the original batch model, if imagenet may need to rename keys. 
        if params.dataset=='imagenet':
            new_state_dict = OrderedDict()
            for k, v in state_dict_final["model"].items():
                if k[:7]=='module.': # remove "module.""
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k]=v
            final_model.load_state_dict(new_state_dict)
        else:
            final_model.load_state_dict(state_dict_final['model'])
        # check the model gradient is zeros
        final_model.zero_grad()

        #get the gradient
        output=final_model(image)
        loss=F.cross_entropy(output, target)
        loss.backward()

        grads=[]
        for param in final_model.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
          
        g=grads.cpu().numpy()
        grad_norms[id]=np.linalg.norm(g)
        
    return grad_norms


def get_calibrated_gradnorm(params, private_model, private_params, attack_model,attack_params, ids, mask, aug_style='mean',norm_type=None):
    """
    return calibrated gradient norm values. 
    """
    #load the dataset
    dataset = get_dataset(params)

    #initialize to 0
    grad_norms=np.zeros(len(mask))

    if params.aug:
        batch_vals=[[0] for i in np.arange(len(mask))]
        for t in np.arange(10):
            batched_ids=np.array_split(ids, 1000)
            for b_ids in batched_ids:
                image_data=torch.stack([dataset[i][0] for i in b_ids])
                image_data=image_data.cuda()
                target_data=torch.stack([torch.tensor(dataset[i][1]) for i in b_ids])
                target_data=target_data.cuda()
            
                private_model.zero_grad()
                out_private=private_model(image_data)
                loss_private=F.cross_entropy(out_private, target_data)
                loss_private.backward()

                attack_model.zero_grad()          
                out_attack=attack_model(image_data)
                loss_attack=F.cross_entropy(out_attack, target_data)
                loss_attack.backward()
                

                for i,id in enumerate(b_ids):

                    private_grads=[]
                    for param in private_model.parameters():
                        private_grads.append(param.grad_sample[i].view(-1))
                    private_grads = torch.cat(private_grads)

                    attack_grads=[]
                    for param in attack_model.parameters():
                        attack_grads.append(param.grad_sample[i].view(-1))
                    attack_grads = torch.cat(attack_grads)

                    g_private=private_grads.cpu().numpy()
                    g_attack=attack_grads.cpu().numpy() 
                    
                    if norm_type=='inf':
                        batch_vals[id].append(max(g_private-g_attack))
                    else:
                        if norm_type=='1':
                            norm_type=1
                        elif norm_type=='2':
                            norm_type=2
                        elif norm_type=='3':
                            norm_type=3
                        batch_vals[id].append(np.linalg.norm(g_private, ord=norm_type)-np.linalg.norm(g_attack,ord=norm_type))
    
        for id in ids:
            if aug_style=='mean':    
                grad_norms[id]=np.mean(batch_vals[id][1:])
            elif aug_style=='max':
                grad_norms[id]=np.max(batch_vals[id][1:])
            elif aug_style=='median':
                grad_norms[id]=np.median(batch_vals[id][1:])
            elif aug_style=='std':
                grad_norms[id]=np.std(batch_vals[id][1:])
    else:
        batched_ids=np.array_split(ids, 1000)
        for b_ids in batched_ids:
            image_data=torch.stack([dataset[i][0] for i in b_ids])
            image_data=image_data.cuda()
            target_data=torch.stack([torch.tensor(dataset[i][1]) for i in b_ids])
            target_data=target_data.cuda()
            
            private_model.zero_grad()
            out_private=private_model(image_data)
            loss_private=F.cross_entropy(out_private, target_data)
            loss_private.backward()

            attack_model.zero_grad()          
            out_attack=attack_model(image_data)
            loss_attack=F.cross_entropy(out_attack, target_data)
            loss_attack.backward()

            for i,id in enumerate(b_ids):
            
                private_grads=[]
                for param in private_model.parameters():
                    private_grads.append(param.grad_sample[i].view(-1))
                private_grads = torch.cat(private_grads)

                attack_grads=[]
                for param in attack_model.parameters():
                    attack_grads.append(param.grad_sample[i].view(-1))
                attack_grads = torch.cat(attack_grads)

                g_private=private_grads.cpu().numpy()
                g_attack=attack_grads.cpu().numpy()         

                
                if norm_type=='inf':
                    grad_norms[id]=max(g_private-g_attack)
                else:
                    if norm_type=='1':
                        norm_type=1
                    elif norm_type=='2':
                        norm_type=2
                    elif norm_type=='3':
                        norm_type=3
                    grad_norms[id]=np.linalg.norm(g_private, ord=norm_type)-np.linalg.norm(g_attack,ord=norm_type)
                
        
    return grad_norms

def calibrated_gradient_attack(params):
    """
    run a calibrated gradient attack. 
    """
    #load the masks
    known_masks, hidden_masks = {}, {}
    hidden_masks['public'], hidden_masks['private']={},{}
    known_masks['public'] = torch.load(params.mask_path + "public.pth")
    known_masks['private'] = torch.load( params.mask_path + "private.pth")
    hidden_masks['private']['train']=torch.load( params.mask_path + "hidden/train.pth")
    hidden_masks['private']['heldout'] = torch.load( params.mask_path + "hidden/heldout.pth")
    hidden_masks['public']['train']=torch.load( params.mask_path + "hidden/public_train.pth")
    hidden_masks['public']['heldout'] = torch.load( params.mask_path + "hidden/public_heldout.pth")

    if params.public_data=='train':
        print('Using public training data for auxiliary model')
        attack_model=train(params, hidden_masks['public']['train'])
    elif params.public_data[:4]=='rand':
        print('Using random subset for auxiliary model')
        public_ids=(known_masks['public']==True).nonzero().flatten().numpy()
        prop_selected=float(params.public_data[4:])/100
        num_selected=math.ceil(prop_selected*len(public_ids))
        permuted_ids=np.random.permutation(public_ids)
        aux_data_mask=to_mask(len(known_masks['public']),permuted_ids[:num_selected])
        print('Number of public model training points', len((aux_data_mask==True).nonzero().flatten().numpy()))
        attack_model=train(params, aux_data_mask)
    else:
        print('Using all public data for auxiliary model')
        attack_model=train(params, known_masks['public'])
    attack_model=attack_model.cuda()

    #get the attack model parameters
    original_attack_model=[]
    for p in attack_model.parameters():
        original_attack_model.append(p.view(-1))
    original_attack_model=torch.cat(original_attack_model)

    #get the final model parameters
    private_model=build_model(params)
    private_model_path = os.path.join(params.model_path, "checkpoint.pth")
    state_dict_private = torch.load(private_model_path)
    private_model.load_state_dict(state_dict_private['model'])
    private_model=private_model.cuda()
 
    original_private_model=[]
    for p in private_model.parameters():
        original_private_model.append(p.view(-1))
    original_private_model=torch.cat(original_private_model)

    #get the appropriate ids to dot product
    private_train_ids=(hidden_masks['private']['train']==True).nonzero().flatten().numpy()
    private_heldout_ids=(hidden_masks['private']['heldout']==True).nonzero().flatten().numpy()

    # reload model to allow use of gradsamplemodule
    new_model=build_model(params)
    new_model_path = os.path.join(params.dump_path, "checkpoint.pth")
    state_dict_new = torch.load(new_model_path)
    new_model.load_state_dict(state_dict_new['model'])
    new_model=new_model.cuda()

    private_model=GradSampleModule(private_model)
    attack_model=GradSampleModule(new_model)

    train_dots=get_calibrated_gradnorm(params, private_model, original_private_model, attack_model,original_attack_model,private_train_ids,hidden_masks['private']['train'])
    heldout_dots=get_calibrated_gradnorm(params, private_model, original_private_model, attack_model,original_attack_model,private_heldout_ids,hidden_masks['private']['heldout'])

    return train_dots, heldout_dots

def get_calibrated_losses(params, private_model, attack_model, ids, mask, aug_style='mean'):
    """
    return calibrated losses 
    """
    #load the dataset
    dataset = get_dataset(params)
    #initialize dot products to 0
    losses=np.zeros(len(mask))
    
    if params.aug:
        summed_loss=[[0] for i in np.arange(len(mask))]
        for j in np.arange(10):
            print('aug',j)
            batched_ids=np.array_split(ids, 1000)
            for b_ids in batched_ids:
                image_data=torch.stack([dataset[i][0] for i in b_ids])
                image_data=image_data.cuda()
                target_data=torch.stack([torch.tensor(dataset[i][1]) for i in b_ids])
                target_data=target_data.cuda()
                out_private=private_model(image_data)
                out_attack=attack_model(image_data)
                for i,id in enumerate(b_ids):
                    output=out_private[i].unsqueeze(0)
                    loss=F.cross_entropy(output, target_data[i].unsqueeze(0))
                    attack_output=out_attack[i].unsqueeze(0)
                    attack_loss=F.cross_entropy(attack_output, target_data[i].unsqueeze(0))
                    loss_diff=loss-attack_loss
                    summed_loss[id].append(loss_diff.cpu().detach().numpy())
        for id in ids:
            if aug_style=='mean':
                losses[id]=np.mean(summed_loss[id][1:])
            elif aug_style=='max':
                losses[id]=np.max(summed_loss[id][1:])
            elif aug_style=='median':
                losses[id]=np.median(summed_loss[id][1:])
            elif aug_style=='std':
                losses[id]=np.std(summed_loss[id][1:])
    else:
        for id in ids:
            #load each image and target
            image = dataset[id][0].unsqueeze(0)
            image = image.cuda(non_blocking=True)
            target = torch.tensor(dataset[id][1]).unsqueeze(0)
            target = target.cuda(non_blocking=True)

            #get the loss
            output=private_model(image)
            loss=F.cross_entropy(output, target)

            attack_output=attack_model(image)
            attack_loss=F.cross_entropy(attack_output, target)

            losses[id]=loss-attack_loss
       
    return losses

def get_calibrated_confidences(params, private_model, attack_model, ids, mask, aug_style='mean'):
    """
    return calibrated confidences. 
    """
    #load the dataset
    dataset = get_dataset(params)
    #initialize dot products to 0
    confidences=np.zeros(len(mask))
    
    if params.aug:
        summed_confs=[[0] for i in np.arange(len(mask))]
        for j in np.arange(10):
            print('Aug', j)
            images=torch.stack([dataset[i][0] for i in ids])
            images=images.cuda()

            log_softmax = torch.nn.LogSoftmax(dim=1)
        
            output=private_model(images)
            attack_output=attack_model(images)
        
            log_output=log_softmax(output)
            log_attack_output=log_softmax(attack_output)
        
            private_confidences,_=torch.max(log_output,dim=1)
            attack_confidences,_=torch.max(log_attack_output,dim=1)
            confs=private_confidences-attack_confidences
            confs=confs.cpu().detach().numpy()
            for i,id in enumerate(ids):
                summed_confs[id].append(confs[i])
        for id in ids:
            if aug_style=='mean':
                confidences[id]=np.mean(summed_confs[id][1:])
            elif aug_style=='max':
                confidences[id]=np.max(summed_confs[id][1:])
            elif aug_style=='median':
                confidences[id]=np.median(summed_confs[id][1:])
            elif aug_style=='std':
                confidences[id]=np.std(summed_confs[id][1:])
    else:
        images=torch.stack([dataset[i][0] for i in ids])
        images=images.cuda()

        log_softmax = torch.nn.LogSoftmax(dim=1)
    
        output=private_model(images)
        attack_output=attack_model(images)
    
        log_output=log_softmax(output)
        log_attack_output=log_softmax(attack_output)
    
        private_confidences,_=torch.max(log_output,dim=1)
        attack_confidences,_=torch.max(log_attack_output,dim=1)
        confidences=private_confidences-attack_confidences
       
    return confidences

def calibrated_loss_attack(params):
    """
    run a calibrated loss attack. 
    """
       #load the masks
    known_masks, hidden_masks = {}, {}
    hidden_masks['public'], hidden_masks['private']={},{}
    known_masks['public'] = torch.load(params.mask_path + "public.pth")
    known_masks['private'] = torch.load( params.mask_path + "private.pth")
    hidden_masks['private']['train']=torch.load( params.mask_path + "hidden/train.pth")
    hidden_masks['private']['heldout'] = torch.load( params.mask_path + "hidden/heldout.pth")
    hidden_masks['public']['train']=torch.load( params.mask_path + "hidden/public_train.pth")
    hidden_masks['public']['heldout'] = torch.load( params.mask_path + "hidden/public_heldout.pth")

    if params.public_data=='train':
        print('Using public training data for auxiliary model')
        attack_model=train(params, hidden_masks['public']['train'])
    elif params.public_data[:4]=='rand':
        print('Using random subset for auxiliary model')
        public_ids=(known_masks['public']==True).nonzero().flatten().numpy()
        prop_selected=float(params.public_data[4:])/100
        num_selected=math.ceil(prop_selected*len(public_ids))
        permuted_ids=np.random.permutation(public_ids)
        aux_data_mask=to_mask(len(known_masks['public']),permuted_ids[:num_selected])
        print('Number of public model training points', len((aux_data_mask==True).nonzero().flatten().numpy()))
        attack_model=train(params, aux_data_mask)
    else:
        print('Using all public data for auxiliary model')
        attack_model=train(params, known_masks['public'])
    attack_model=attack_model.cuda()

    #get the final model parameters
    private_model=build_model(params)
    private_model_path = os.path.join(params.model_path, "checkpoint.pth")
    state_dict_private = torch.load(private_model_path)
    private_model.load_state_dict(state_dict_private['model'])
    private_model=private_model.cuda()

    #get the appropriate ids to dot product
    private_train_ids=(hidden_masks['private']['train']==True).nonzero().flatten().numpy()
    private_heldout_ids=(hidden_masks['private']['heldout']==True).nonzero().flatten().numpy()

    train_losses=get_calibrated_losses(params, private_model, attack_model,private_train_ids,hidden_masks['private']['train'])
    heldout_losses=get_calibrated_losses(params, private_model, attack_model,private_heldout_ids,hidden_masks['private']['heldout'])

    return train_losses, heldout_losses

def calibrated_confidence_attack(params):
    """
    run a calibrated confidence attack. 
    """

    #load the masks
    known_masks, hidden_masks = {}, {}
    hidden_masks['public'], hidden_masks['private']={},{}
    known_masks['public'] = torch.load(params.mask_path + "public.pth")
    known_masks['private'] = torch.load( params.mask_path + "private.pth")
    hidden_masks['private']['train']=torch.load( params.mask_path + "hidden/train.pth")
    hidden_masks['private']['heldout'] = torch.load( params.mask_path + "hidden/heldout.pth")
    hidden_masks['public']['train']=torch.load( params.mask_path + "hidden/public_train.pth")
    hidden_masks['public']['heldout'] = torch.load( params.mask_path + "hidden/public_heldout.pth")

    if params.public_data=='train':
        print('Using public training data for auxiliary model')
        attack_model=train(params, hidden_masks['public']['train'])
    elif params.public_data[:4]=='rand':
        print('Using random subset for auxiliary model')
        public_ids=(known_masks['public']==True).nonzero().flatten().numpy()
        prop_selected=float(params.public_data[4:])/100
        num_selected=math.ceil(prop_selected*len(public_ids))
        permuted_ids=np.random.permutation(public_ids)
        aux_data_mask=to_mask(len(known_masks['public']),permuted_ids[:num_selected])
        print('Number of public model training points', len((aux_data_mask==True).nonzero().flatten().numpy()))
        attack_model=train(params, aux_data_mask)
    else:
        print('Using all public data for auxiliary model')
        attack_model=train(params, known_masks['public'])
    attack_model=attack_model.cuda()

    #get the final model parameters
    private_model=build_model(params)
    private_model_path = os.path.join(params.model_path, "checkpoint.pth")
    state_dict_private = torch.load(private_model_path)
    private_model.load_state_dict(state_dict_private['model'])
    private_model=private_model.cuda()

    #get the appropriate ids to dot product
    private_train_ids=(hidden_masks['private']['train']==True).nonzero().flatten().numpy()
    private_heldout_ids=(hidden_masks['private']['heldout']==True).nonzero().flatten().numpy()

    train_losses=get_calibrated_confidences(params, private_model, attack_model,private_train_ids,hidden_masks['private']['train'])
    heldout_losses=get_calibrated_confidences(params, private_model, attack_model,private_heldout_ids,hidden_masks['private']['heldout'])

    return train_losses, heldout_losses

def auxiliary_attack(params, aux_epochs, attack_type='loss', aug_style='mean', norm_type=None, public_data='train', num_aux=1,aux_style='sum'):
    """
    run an auxiliary attack, type (loss, grad_norm, conf, dist) specified by attack_type. 
    """
    #load the masks
    known_masks, hidden_masks = {}, {}
    hidden_masks['public'], hidden_masks['private']={},{}
    known_masks['public'] = torch.load(params.mask_path + "public.pth")
    known_masks['private'] = torch.load( params.mask_path + "private.pth")
    hidden_masks['private']['train']=torch.load( params.mask_path + "hidden/train.pth")
    hidden_masks['private']['heldout'] = torch.load( params.mask_path + "hidden/heldout.pth")
    hidden_masks['public']['train']=torch.load( params.mask_path + "hidden/public_train.pth")
    hidden_masks['public']['heldout'] = torch.load( params.mask_path + "hidden/public_heldout.pth")

    #get the final model parameters
    private_model=build_model(params)
    private_model_path = os.path.join(params.model_path, "checkpoint.pth")
    state_dict_private = torch.load(private_model_path,map_location='cuda:0')
    if params.dataset=='imagenet':
        new_state_dict = OrderedDict()
        for k, v in state_dict_private["model"].items():
            if k[:7]=='module.': # remove `module.`
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k]=v
        private_model.load_state_dict(new_state_dict)
    else:
        private_model.load_state_dict(state_dict_private['model'])
    private_model=private_model.cuda()

    # updated_params=copy.deepcopy(params)
    updated_params=params
    updated_params.epochs=updated_params.epochs+aux_epochs
   
    private_train_ids=(hidden_masks['private']['train']==True).nonzero().flatten().numpy()
    private_heldout_ids=(hidden_masks['private']['heldout']==True).nonzero().flatten().numpy()

    train_losses=np.zeros(len(known_masks['public']))
    heldout_losses=np.zeros(len(known_masks['public']))

    for i in np.arange(num_aux):
        if params.dataset=='cifar10' or params.dataset=='credit' or params.dataset=='hep' or params.dataset=='adult' or params.dataset=='mnist':
            model_num=params.model_path[-6:-5]
        elif params.dataset=='cifar100':
            model_num=params.model_path[-15:-14]
        else:
            model_num='0'
        new_model_path='updated_model_'+str(aux_epochs) +'_'+str(params.batch_size)+'_'+params.optimizer+'_aux_model_'+str(i)+'_num_aux_'+str(num_aux)+'_public_data_'+params.public_data+'_model_'+model_num
        if not os.path.isdir(new_model_path):
            os.mkdir(new_model_path)
        updated_params.dump_path=new_model_path
        if updated_params.local_rank!=-1:
            updated_params.local_rank=-1
        path = os.path.join(updated_params.dump_path, 'checkpoint.pth')
        torch.save(state_dict_private, path)

        if public_data=='train':
            print('Using public training data for auxiliary model')
            updated_model=train(updated_params, hidden_masks['public']['train'])
        elif public_data[:4]=='rand':
            print('Using random subset for auxiliary model')
            public_ids=(known_masks['public']==True).nonzero().flatten().numpy()
            prop_selected=float(public_data[4:])/100
            num_selected=math.ceil(prop_selected*len(public_ids))
            permuted_ids=np.random.permutation(public_ids)
            aux_data_mask=to_mask(len(known_masks['public']),permuted_ids[:num_selected])
            print('Number of public model training points', len((aux_data_mask==True).nonzero().flatten().numpy()))
            updated_model=train(updated_params, aux_data_mask)
        else:
            print('Using all public data for auxiliary model')
            updated_model=train(updated_params, known_masks['public'])
        updated_model=updated_model.cuda()

        new_model=build_model(params)
        new_model_path=os.path.join(updated_params.dump_path, "checkpoint.pth")
        state_dict_new=torch.load(new_model_path,map_location='cuda:0')
        if params.dataset=='imagenet':
            new_state_dict = OrderedDict()
            for k, v in state_dict_new["model"].items():
                if k[:7]=='module.': # remove `module.`
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k]=v
            new_model.load_state_dict(new_state_dict)
        else:
            new_model.load_state_dict(state_dict_new['model'])
        new_model=new_model.cuda()

        #get losses
        if attack_type=='loss':
            train_vals=get_calibrated_losses(params, private_model, updated_model,private_train_ids,hidden_masks['private']['train'], aug_style)
            heldout_vals=get_calibrated_losses(params, private_model, updated_model,private_heldout_ids,hidden_masks['private']['heldout'], aug_style)
        elif attack_type=='conf':
            train_vals=get_calibrated_confidences(params, private_model, updated_model,private_train_ids,hidden_masks['private']['train'], aug_style)
            heldout_vals=get_calibrated_confidences(params, private_model, updated_model,private_heldout_ids,hidden_masks['private']['heldout'], aug_style)
        elif attack_type=='dist':
            private_train_ids=private_train_ids[np.random.choice(len(private_train_ids), size=params.num_points, replace=False)]
            private_heldout_ids=private_heldout_ids[np.random.choice(len(private_heldout_ids), size=params.num_points, replace=False)]
            train_vals=get_calibrated_distances(params, private_model, updated_model,private_train_ids)
            heldout_vals=get_calibrated_distances(params, private_model, updated_model,private_heldout_ids)
        else:
            original_private_model=[]
            for p in private_model.parameters():
                original_private_model.append(p.view(-1))
            original_private_model=torch.cat(original_private_model)

            original_updated_model=[]
            for p in new_model.parameters():
                original_updated_model.append(p.view(-1))
            original_updated_model=torch.cat(original_updated_model)

            if i==0:
                private_model=GradSampleModule(private_model)
            attack_model=GradSampleModule(new_model)

            train_vals=get_calibrated_gradnorm(params, private_model,original_private_model, attack_model,original_updated_model,private_train_ids,hidden_masks['private']['train'],  aug_style=aug_style, norm_type=norm_type)
            heldout_vals=get_calibrated_gradnorm(params, private_model, original_private_model,attack_model,original_updated_model,private_heldout_ids,hidden_masks['private']['heldout'], aug_style=aug_style,norm_type=norm_type)
        if aux_style=='max':
            train_losses=np.maximum(train_losses, train_vals)
            heldout_losses=np.maximum(heldout_losses, heldout_vals)
        else: 
            if params.attack_type=='conf' or params.attack_type=='dist':
                train_losses=train_vals
                heldout_losses=heldout_vals
            else:
                train_losses+=train_vals
                heldout_losses+=heldout_vals
    if aux_style=='mean':
        train_losses=train_losses/num_aux
        heldout_losses=heldout_losses/num_aux
    return train_losses, heldout_losses

def get_losses(params):
    """
    return uncalibrated losses. 
    """
    known_masks, hidden_masks = {}, {}
    hidden_masks['public'], hidden_masks['private']={},{}
    known_masks['public'] = torch.load(params.mask_path + "public.pth")
    known_masks['private'] = torch.load( params.mask_path + "private.pth")
    hidden_masks['private']['train']=torch.load( params.mask_path + "hidden/train.pth")
    hidden_masks['private']['heldout'] = torch.load( params.mask_path + "hidden/heldout.pth")
    hidden_masks['public']['train']=torch.load( params.mask_path + "hidden/public_train.pth")
    hidden_masks['public']['heldout'] = torch.load( params.mask_path + "hidden/public_heldout.pth")

    #get the final model parameters
    private_model=build_model(params)
    private_model_path = os.path.join(params.model_path, "checkpoint.pth")
    state_dict_private = torch.load(private_model_path,map_location='cuda:0')
    if params.dataset=='imagenet':
        new_state_dict = OrderedDict()
        for k, v in state_dict_private["model"].items():
            if k[:7]=='module.': # remove `module.`
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k]=v
        private_model.load_state_dict(new_state_dict)
    else:
        private_model.load_state_dict(state_dict_private['model'])
    private_model=private_model.cuda()

    #get the appropriate ids to dot product
    private_train_ids=(hidden_masks['private']['train']==True).nonzero().flatten().numpy()
    private_heldout_ids=(hidden_masks['private']['heldout']==True).nonzero().flatten().numpy()
    #load the dataset
    dataset = get_dataset(params)
    #initialize dot products to 0
    train_losses=[]
    heldout_losses=[]

    for id in private_train_ids:
        #load each image and target
        image = dataset[id][0].unsqueeze(0)
        image = image.cuda(non_blocking=True)
        target = torch.tensor(dataset[id][1]).unsqueeze(0)
        target = target.cuda(non_blocking=True)

        #get the loss
        output=private_model(image)
        loss=F.cross_entropy(output, target).item()
        train_losses.append(loss)
    
    for id in private_heldout_ids:
        #load each image and target
        image = dataset[id][0].unsqueeze(0)
        image = image.cuda(non_blocking=True)
        target = torch.tensor(dataset[id][1]).unsqueeze(0)
        target = target.cuda(non_blocking=True)

        #get the loss
        output=private_model(image)
        loss=F.cross_entropy(output, target).item()
        heldout_losses.append(loss)
    return train_losses,heldout_losses

def get_confidences(params):
    """
    return uncalibrated confidences. 
    """
    known_masks, hidden_masks = {}, {}
    hidden_masks['public'], hidden_masks['private']={},{}
    known_masks['public'] = torch.load(params.mask_path + "public.pth")
    known_masks['private'] = torch.load( params.mask_path + "private.pth")
    hidden_masks['private']['train']=torch.load( params.mask_path + "hidden/train.pth")
    hidden_masks['private']['heldout'] = torch.load( params.mask_path + "hidden/heldout.pth")
    hidden_masks['public']['train']=torch.load( params.mask_path + "hidden/public_train.pth")
    hidden_masks['public']['heldout'] = torch.load( params.mask_path + "hidden/public_heldout.pth")

    device = torch.device('cpu')

    #get the final model parameters
    private_model=build_model(params)
    private_model_path = os.path.join(params.model_path, "checkpoint.pth")
    state_dict_private = torch.load(private_model_path,map_location=device)
    private_model.load_state_dict(state_dict_private['model'])
    private_model=private_model.cpu()

    #get the appropriate ids to dot product
    private_train_ids=(hidden_masks['private']['train']==True).nonzero().flatten().numpy()
    private_heldout_ids=(hidden_masks['private']['heldout']==True).nonzero().flatten().numpy()
    
    #load the dataset
    dataset = get_dataset(params)
    
    if params.aug:
        train_confidences=np.zeros(len(hidden_masks['private']['train']))
        heldout_confidences=np.zeros(len(hidden_masks['private']['train']))
        train_summed_confs=[[0] for i in np.arange(len(hidden_masks['private']['train']))]
        heldout_summed_confs=[[0] for i in np.arange(len(hidden_masks['private']['train']))]
        for j in np.arange(10):
            print('Aug', j)
            train_images=torch.stack([dataset[i][0] for i in private_train_ids])
            train_images=train_images.cpu()
            
            heldout_images=torch.stack([dataset[i][0] for i in private_heldout_ids])
            heldout_images=heldout_images.cpu()

            log_softmax = torch.nn.LogSoftmax(dim=1)
        
            train_output=private_model(train_images)
            heldout_output=private_model(heldout_images)
            
            log_train_output=log_softmax(train_output)
            log_heldout_output=log_softmax(heldout_output)
            
            train_confs,_=torch.max(log_train_output,dim=1)
            heldout_confs,_=torch.max(log_heldout_output,dim=1)

            train_confs=train_confs.cpu().detach().numpy()
            heldout_confs=heldout_confs.cpu().detach().numpy()

            for i,id in enumerate(private_train_ids):
                train_summed_confs[id].append(train_confs[i])
            for i,id in enumerate(private_heldout_ids):
                heldout_summed_confs[id].append(heldout_confs[i])
        for id in private_train_ids:
            if params.aug_style=='mean':
                train_confidences[id]=np.mean(train_summed_confs[id][1:])
            elif params.aug_style=='max':
                train_confidences[id]=np.max(train_summed_confs[id][1:])
            elif params.aug_style=='median':
                train_confidences[id]=np.median(train_summed_confs[id][1:])
            elif params.aug_style=='std':
                train_confidences[id]=np.std(train_summed_confs[id][1:])
        for id in private_heldout_ids:
            if params.aug_style=='mean':
                heldout_confidences[id]=np.mean(heldout_summed_confs[id][1:])
            elif params.aug_style=='max':
                heldout_confidences[id]=np.max(heldout_summed_confs[id][1:])
            elif params.aug_style=='median':
                heldout_confidences[id]=np.median(heldout_summed_confs[id][1:])
            elif params.aug_style=='std':
                heldout_confidences[id]=np.std(heldout_summed_confs[id][1:])
        
        train_confidences=train_confidences[private_train_ids]
        heldout_confidences=heldout_confidences[private_heldout_ids]

    else:
        train_confidences=[]
        heldout_confidences=[]

        train_images=torch.stack([dataset[i][0] for i in private_train_ids])
        train_images=train_images.cpu()
        
        heldout_images=torch.stack([dataset[i][0] for i in private_heldout_ids])
        heldout_images=heldout_images.cpu()
        
        log_softmax = torch.nn.LogSoftmax(dim=1)
        
        train_output=private_model(train_images)
        heldout_output=private_model(heldout_images)
        
        log_train_output=log_softmax(train_output)
        log_heldout_output=log_softmax(heldout_output)
        
        train_confidences,_=torch.max(log_train_output,dim=1)
        heldout_confidences,_=torch.max(log_heldout_output,dim=1)
        train_confidences=train_confidences.cpu().detach().numpy()
        heldout_confidences=heldout_confidences.cpu().detach().numpy()
    return train_confidences,heldout_confidences

def get_calibrated_distances(params, model1, model2, ids):
    """
    return calibrated boundary distances. 
    """
    dataset = get_dataset(params)
    images=torch.stack([dataset[i][0] for i in ids])
    images=images.cuda()
    targets=torch.stack([torch.tensor(dataset[i][1]) for i in ids])
    targets=targets.cuda()

    outputs1=model1(images)
    outputs2=model2(images)
    images_pert1= hop_skip_jump_attack(model1,images,2, verbose=False,clip_min=params.clip_min, clip_max=params.clip_max)
    images_pert2= hop_skip_jump_attack(model2,images,2, verbose=False,clip_min=params.clip_min, clip_max=params.clip_max)
    # images_pert1=carlini_wagner_l2(model1,images,params.num_classes ,targets)
    # images_pert2=carlini_wagner_l2(model2,images,params.num_classes ,targets)
    dists1=[]
    for i, id in enumerate(ids):
        _, pred = torch.topk(outputs1[i], 1)
        if pred==targets[i].item():
            dists1.append(torch.norm(images_pert1[i]- images[i], p=2).item())
        else:
            dists1.append(-torch.norm(images_pert1[i]- images[i], p=2).item())
    dists2=[]
    for i, id in enumerate(ids):
        _, pred = torch.topk(outputs2[i], 1)
        if pred==targets[i].item():
            dists2.append(torch.norm(images_pert2[i]- images[i], p=2).item())
        else:
            dists2.append(-torch.norm(images_pert1[i]- images[i], p=2).item())
    calibrated_dists=np.subtract(np.array(dists1),np.array(dists2))
    return calibrated_dists

def calibrated_distance_attack(params, num=10):
    """
    run calibrated boundary distance attack. 
    """
       #load the masks
    known_masks, hidden_masks = {}, {}
    hidden_masks['public'], hidden_masks['private']={},{}
    known_masks['public'] = torch.load(params.mask_path + "public.pth")
    known_masks['private'] = torch.load( params.mask_path + "private.pth")
    hidden_masks['private']['train']=torch.load( params.mask_path + "hidden/train.pth")
    hidden_masks['private']['heldout'] = torch.load( params.mask_path + "hidden/heldout.pth")
    hidden_masks['public']['train']=torch.load( params.mask_path + "hidden/public_train.pth")
    hidden_masks['public']['heldout'] = torch.load( params.mask_path + "hidden/public_heldout.pth")

    if params.public_data=='train':
        print('Using public training data for auxiliary model')
        attack_model=train(params, hidden_masks['public']['train'])
    elif params.public_data[:4]=='rand':
        print('Using random subset for auxiliary model')
        public_ids=(known_masks['public']==True).nonzero().flatten().numpy()
        prop_selected=float(params.public_data[4:])/100
        num_selected=math.ceil(prop_selected*len(public_ids))
        permuted_ids=np.random.permutation(public_ids)
        aux_data_mask=to_mask(len(known_masks['public']),permuted_ids[:num_selected])
        print('Number of public model training points', len((aux_data_mask==True).nonzero().flatten().numpy()))
        attack_model=train(params, aux_data_mask)
    else:
        print('Using all public data for auxiliary model')
        attack_model=train(params, known_masks['public'])
    attack_model=attack_model.cuda()

    #get the final model parameters
    private_model=build_model(params)
    private_model_path = os.path.join(params.model_path, "checkpoint.pth")
    state_dict_private = torch.load(private_model_path)
    private_model.load_state_dict(state_dict_private['model'])
    private_model=private_model.cuda()

    #get the appropriate ids 
    private_train_ids=(hidden_masks['private']['train']==True).nonzero().flatten().numpy()
    private_train_ids=private_train_ids[np.random.choice(len(private_train_ids), size=num, replace=False)]
    private_heldout_ids=(hidden_masks['private']['heldout']==True).nonzero().flatten().numpy()
    private_heldout_ids=private_heldout_ids[np.random.choice(len(private_heldout_ids), size=num, replace=False)]

    train_losses=get_calibrated_distances(params, private_model, attack_model,private_train_ids)
    heldout_losses=get_calibrated_distances(params, private_model, attack_model,private_heldout_ids)

    return train_losses, heldout_losses

def get_boundary_distances(params, model, ids):
    """
    return uncalibrated boundary distances. 
    """
    dataset = get_dataset(params)
    images=torch.stack([dataset[i][0] for i in ids])
    images=images.cuda()
    targets=[]
    for i in ids:
        temp=np.zeros(params.num_classes)
        temp[dataset[i][1]]=1
        temp=torch.tensor(temp)
        targets.append(temp)
    original_targets=torch.stack([torch.tensor(dataset[i][1]) for i in ids])
    original_targets=original_targets.cuda()
    targets=torch.stack(targets)
    targets=targets.cuda()

    outputs=model(images)

    images_pert= hop_skip_jump_attack(model,images,2 ,verbose=False, clip_min=params.clip_min, clip_max=params.clip_max)
    # images_pert=carlini_wagner_l2(model,images,params.num_classes ,original_targets)
    dists=[]
    for i, id in enumerate(ids):
        _, pred = torch.topk(outputs[i], 1)
        if pred==original_targets[i].item():
            dists.append(torch.norm(images_pert[i]- images[i], p=2).item())
        else:
            dists.append(0)
    return dists

def boundary_distance_attack(params, num=10):
    """
    run uncalibrated boundary distance attack. 
    """
    
    train_mask=torch.load(params.mask_path+'/hidden/train.pth')
    heldout_mask=torch.load(params.mask_path+'/hidden/heldout.pth')

    train_ids=(train_mask==True).nonzero().flatten().numpy()
    heldout_ids=(heldout_mask==True).nonzero().flatten().numpy()
    train_ids=train_ids[np.random.choice(len(train_ids), size=num, replace=False)]
    heldout_ids=heldout_ids[np.random.choice(len(heldout_ids), size=num, replace=False)]

    private_model=build_model(params)
    private_model_path = os.path.join(params.model_path, "checkpoint.pth")
    state_dict_private = torch.load(private_model_path)
    private_model.load_state_dict(state_dict_private['model'])
    private_model=private_model.cuda()

    train_dists=get_boundary_distances(params, private_model, train_ids )
    heldout_dists=get_boundary_distances(params, private_model, heldout_ids )
    
    return train_dists, heldout_dists

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    train_vals, heldout_vals=calibrated_loss_attack(params)