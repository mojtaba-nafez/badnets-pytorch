from .poisoned_dataset import CIFAR10Poison, MNISTPoison, CIFAR100Poison, ImageNetExposure
from torchvision import datasets, transforms
import torch 
import os 

def build_init_data(dataname, download, dataset_path):
    if dataname == 'MNIST':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data  = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif dataname == 'CIFAR10':
        train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=download)
        test_data  = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    return train_data, test_data

def build_poisoned_training_set(is_train, args):
    transform, detransform = build_transform(args.dataset, args.model)
    print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        if args.exposure_training :
            # clean_trainset = datasets.CIFAR10(root=args.data_path, train=True,  download=True, transform=transform)
            clean_trainset = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform, count=2500)
            exposure_dataset = ImageNetExposure(args=args, root='./tiny-imagenet-200', count=2500, transform=transform)
            trainset = torch.utils.data.ConcatDataset([clean_trainset, exposure_dataset])
        else:
            trainset = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        trainset = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(trainset)

    return trainset, nb_classes


def build_testset(is_train, args):
    transform, detransform = build_transform(args.dataset, args.model)
    print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        testset_clean = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        testset_clean = datasets.MNIST(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(testset_clean, testset_poisoned)

    return testset_clean, testset_poisoned

def build_transform(dataset, model):
    if dataset == "CIFAR10":
        if model=='resnet18':
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        elif model=='vit':
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        elif model=='simple_conv':
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif dataset == "MNIST":
        if model=='resnet18':
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        elif model=='vit':
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        elif model=='simple_conv':
            mean, std = (0.5,), (0.5,)
    else:
        raise NotImplementedError()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean, std)
        ])
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()) # you can use detransform to recover the image
    
    return transform, detransform

def build_ood_testset(is_train, args):
    transform, detransform = build_transform(args.dataset, args.model)
    print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        testset_clean_10 = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        for i in range(len(testset_clean_10.targets)):
            testset_clean_10.targets[i] = 1
        testset_clean_100 = datasets.CIFAR100(args.data_path, train=is_train, download=True, transform=transform)
        for i in range(len(testset_clean_100.targets)):
            testset_clean_100.targets[i] = 0
            
        testset_clean = torch.utils.data.ConcatDataset([testset_clean_10, testset_clean_100])
        testset_poisoned_100 = CIFAR100Poison(args, args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = torch.utils.data.ConcatDataset([testset_clean_10, testset_poisoned_100])

        nb_classes = 10
    elif args.dataset == 'MNIST':
        testset_clean = datasets.MNIST(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    return testset_clean, testset_poisoned
