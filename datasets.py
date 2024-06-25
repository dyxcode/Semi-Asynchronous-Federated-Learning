#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

from knowledge_distillation.datasets import DatasetWrapper
    
def get_dataset(env):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if env.dataset == 'mnist':
        train_dataset, test_dataset = get_mnist()
    elif env.dataset == 'cifar10':
        train_dataset, test_dataset = get_cifar10()
    elif env.dataset == 'cifar100':
        train_dataset, test_dataset = get_cifar100()
    else:
        raise NotImplementedError

    dataset_size = len(train_dataset)
    indices = torch.randperm(dataset_size).tolist()
    indices *= (env.n_users * env.n_sample // dataset_size + 1)

    user_datasets = [Subset(train_dataset, indices[i * env.n_sample : (i + 1) * env.n_sample])
                        for i in range(env.n_users)]
        
    return user_datasets, test_dataset

def get_mnist():
    data_dir = './data/mnist/'

    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                    transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
    
    return train_dataset, test_dataset

def get_cifar10():
    """
    cifar 10
    """
    data_folder = './data/cifar10/'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = DatasetWrapper(datasets.CIFAR10(root=data_folder, download=True, train=True, transform=transform_train),
                            logits_path='./knowledge_distillation/teacher_logits_cifar10/',
                            topk=5,
                            write=False)

    test_set = datasets.CIFAR10(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=transform_test)

    return train_set, test_set

def get_cifar100():
    """
    cifar 100
    """
    data_folder = './data/cifar100/'

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = datasets.CIFAR100(root=data_folder,
                                  download=True,
                                  train=True,
                                  transform=train_transform)
    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)

    return train_set, test_set