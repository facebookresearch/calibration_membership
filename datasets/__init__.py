# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset
import numpy as np
from numpy.random import multivariate_normal
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from .text_data import TextIterator

class IdxDataset(Dataset):
    """
    Wraps a dataset so that with each element is also returned its index
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, i: int):
        sample = self.dataset[i]
        if type(sample) is tuple:
            sample = list(sample)
            sample.insert(0, i)
            return tuple(sample)
        else:
            return i, sample

    def __len__(self):
        return len(self.dataset)


class MaskDataset(Dataset):
    def __init__(self, dataset: Dataset, mask: torch.Tensor):
        """
        example:
        mask: [0, 1, 1]
        cumul: [-1, 0, 1]
        remap: {0: 1, 1: 2}
        """
        assert mask.dim() == 1
        assert mask.size(0) == len(dataset)
        assert mask.dtype == torch.bool

        mask = mask.long()
        cumul = torch.cumsum(mask, dim=0) - 1
        self.remap = {}
        for i in range(mask.size(0)):
            if mask[i] == 1:
                self.remap[cumul[i].item()] = i
            assert mask[i] in [0, 1]

        self.dataset = dataset
        self.mask = mask
        self.length = cumul[-1].item() + 1

    def __getitem__(self, i: int):
        return self.dataset[self.remap[i]]

    def __len__(self):
        return self.length

def adult_data_transform(df):
    binary_data = pd.get_dummies(df)
    feature_cols = binary_data[binary_data.columns[:-2]]
    scaler = preprocessing.StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(feature_cols), columns=feature_cols.columns)
    return data

def get_transform(dataset, aug, is_train):
    if dataset == "cifar10":
        if aug and is_train:
            print('Using data augmentation to train model')
            augmentations = [transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()]
            normalize = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            transform = transforms.Compose(augmentations + normalize)
        else:
            print('Not using data augmentation to train model')
            transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif dataset=='mnist':
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    elif dataset=='imagenet':
        if aug and is_train:
            print('Using data augmentation to train model')
            augmentations = [transforms.Resize(256),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip()]
            normalize = [transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            transform = transforms.Compose(augmentations + normalize)
        else:
            print('Not using data augmentation to train model')
            transform = transforms.Compose( [transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    elif dataset=='cifar100':
        if aug and is_train:
            print('Using data augmentation to train model')
            augmentations = [transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()]
            normalize = [transforms.ToTensor(),transforms.Normalize(mean=[n/255 for n in [129.3, 124.1, 112.4]], std=[n/255 for n in [68.2,  65.4,  70.4]])]
            transform = transforms.Compose(augmentations + normalize)
        else:
            print('Not using data augmentation to train model')
            transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize(mean=[n/255 for n in [129.3, 124.1, 112.4]], std=[n/255 for n in [68.2,  65.4,  70.4]])])

    return transform


def get_dataset(*, params, is_train, mask=None):
    if is_train:
        assert mask is not None

    if params.dataset == "cifar10":
        if is_train:
            transform = get_transform(params.dataset, params.aug, True)
        else:
            transform = get_transform(params.dataset, params.aug, False)

        dataset = torchvision.datasets.CIFAR10(root=params.data_root, train=is_train, download=True, transform=transform)
        dataset = IdxDataset(dataset)
        if mask is not None:
            dataset = MaskDataset(dataset, mask)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
        n_data = len(dataset)
        params.num_classes = 10
        return dataloader, n_data

    elif params.dataset=="imagenet":
        if is_train:
            transform = get_transform(params.dataset, params.aug, True)
        else:
            transform = get_transform(params.dataset, params.aug, False)
        if is_train:
            dataset = torchvision.datasets.ImageFolder(root=params.data_root+'/train',transform=transform)
        else:
            dataset = torchvision.datasets.ImageFolder(root=params.data_root+'/val',transform=transform)
        dataset = IdxDataset(dataset)
        if mask is not None:
            dataset = MaskDataset(dataset, mask)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
        n_data = len(dataset)

        params.num_classes = 1000

        return dataloader, n_data

    elif params.dataset=='cifar100':

        if is_train:
            transform = get_transform(params.dataset, params.aug, True)
        else:
            transform = get_transform(params.dataset, params.aug, False)

        dataset = torchvision.datasets.CIFAR100(root=params.data_root, train=is_train, download=True, transform=transform)
        dataset = IdxDataset(dataset)
        
        if mask is not None:
            dataset = MaskDataset(dataset, mask)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
        
        n_data = len(dataset)

        params.num_classes = 100

        return dataloader, n_data
    
    elif params.dataset=='mnist':

        transform = get_transform(params.dataset, params.aug, True)

        dataset = torchvision.datasets.MNIST(root=params.data_root, train=is_train, download=True, transform=transform)
        dataset = IdxDataset(dataset)
        
        if mask is not None:
            dataset = MaskDataset(dataset, mask)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
        
        n_data = len(dataset)

        params.num_classes = 10

        return dataloader, n_data

    elif params.dataset=='gaussian':
        
        x,y=get_gaussian_dataset(params.n_data,params.num_classes,params.data_num_dimensions,params.random_seed,scale=params.scale)
        
        data = []
        for i in range(len(x)):
            data.append([i,x[i], y[i]])

        dataloader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=params.batch_size)

        return dataloader, params.n_data

    elif params.dataset=='credit':
        
        cred=fetch_openml('credit-g')
    
        data = SimpleImputer(missing_values=np.nan, strategy='mean', copy=True).fit(cred.data).transform(cred.data)
        target = preprocessing.LabelEncoder().fit(cred.target).transform(cred.target)   
        X=data
        norm = np.max(np.concatenate((-1*X.min(axis=0)[np.newaxis], X.max(axis=0)[np.newaxis]),axis=0).T, axis=1).astype('float32')
        data=np.divide(data,norm)

        data=torch.tensor(data).float()
        target=torch.tensor(target).long()
        if is_train:
            ids=np.arange(1000)[:800]
        else:
            ids=np.arange(1000)[800:]
        
        final_data = []
        for i in ids:
            final_data.append([i,data[i], target[i]])
        
        norm=np.max
        params.num_classes = 2
        
        if mask is not None:
            final_data = MaskDataset(final_data, mask)
        dataloader = torch.utils.data.DataLoader(final_data, shuffle=True, batch_size=params.batch_size)
        
        n_data=len(final_data)
        print('Datasize', n_data)

        return dataloader, n_data

    elif params.dataset=='hep':
        
        hep=fetch_openml('hepatitis')
    
        data = SimpleImputer(missing_values=np.nan, strategy='mean', copy=True).fit(hep.data).transform(hep.data)
        target = preprocessing.LabelEncoder().fit(hep.target).transform(hep.target)   
        
        X=data
        norm = np.max(np.concatenate((-1*X.min(axis=0)[np.newaxis], X.max(axis=0)[np.newaxis]),axis=0).T, axis=1).astype('float32')
        data=np.divide(data,norm)

        data=torch.tensor(data).float()
        target=torch.tensor(target).long()
        if is_train:
            ids=np.arange(155)[:124]
        else:
            ids=np.arange(155)[124:]
        
        final_data = []
        for i in ids:
            final_data.append([i,data[i], target[i]])
        
        params.num_classes = 2
        
        if mask is not None:
            final_data = MaskDataset(final_data, mask)
        
        dataloader = torch.utils.data.DataLoader(final_data, shuffle=True, batch_size=params.batch_size)
        
        n_data=len(final_data)
        print('Datasize', n_data)

        return dataloader, n_data
    
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
        
        if is_train:
            final_data = []
            for i in np.arange(len(train_data)):
                final_data.append([i,train_data[i], train_labels[i]])
                        
            if mask is not None:
                final_data = MaskDataset(final_data, mask)
            
            dataloader = torch.utils.data.DataLoader(final_data, shuffle=True, batch_size=params.batch_size)
            
            n_data=len(final_data)
        else:
            final_data = []
            for i in np.arange(len(test_data)):
                final_data.append([i,test_data[i], test_labels[i]])
            
            dataloader = torch.utils.data.DataLoader(final_data, batch_size=params.batch_size)
            
            n_data=len(final_data)

            print('Datasize', n_data)

        return dataloader,n_data

    elif params.dataset == 'lfw':
        
        lfw_people = fetch_lfw_people(data_home=params.data_root,min_faces_per_person=100, resize=0.4)
        n_samples, h, w = lfw_people.images.shape
        lfw_images=torch.tensor(lfw_people.images).float()
        lfw_targets=torch.tensor(lfw_people.target).long()

        if is_train:
            ids=np.arange(1140)[:912]
        else:
            ids=np.arange(1140)[912:]
        
        final_data = []
        for i in ids:
            image=lfw_images[i].reshape((h, w)).unsqueeze(0)
            target=lfw_targets[i]
            final_data.append([i,image, target])
        
        params.num_classes = 5
        
        if mask is not None:
            final_data = MaskDataset(final_data, mask)
        
        dataloader = torch.utils.data.DataLoader(final_data, shuffle=True, batch_size=params.batch_size)
        
        n_data=len(final_data)

        return dataloader, n_data

    elif params.dataset == "dummy":
        # Creates a dummy dataset for NLP
        n_data, delta = 10000, 3
        data = torch.randint(-delta, delta, size=(n_data, params.seq_len))
        data = torch.cumsum(data, dim=1)
        data = torch.remainder(data, params.n_vocab)

        iterator = TextIterator(data.view(-1), params.batch_size, params.seq_len)

        return iterator, n_data



def get_gaussian_dataset(n,num_classes,num_dimensions,random_seed,scale=1):
    
    np.random.seed(random_seed)
    
    mu = [(2*np.random.rand(num_dimensions) - 1) * scale for c in range(num_classes)]
    S = np.diag(np.random.rand(num_dimensions)) + 0.5

    np.random.seed(np.random.randint(1000))
    x = np.concatenate([multivariate_normal(mu[c], S, n//num_classes) for c in range(num_classes)])
    y = np.concatenate([np.ones(n//num_classes) * c for c in range(num_classes)])
    x=torch.tensor(x).float()
    y=torch.tensor(y).long()
    
    return x, y