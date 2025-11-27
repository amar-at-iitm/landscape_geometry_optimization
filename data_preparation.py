import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')


def prepare_dataloaders(args):
    if args.dataset == 'cifar10':
        os.makedirs(args.data_root, exist_ok=True)
        cifar_folder = os.path.join(args.data_root, 'cifar-10-batches-py')
        expected_files = ['data_batch_1', 'test_batch']
        missing_cache = (
            not os.path.isdir(cifar_folder)
            or any(not os.path.exists(os.path.join(cifar_folder, f)) for f in expected_files)
        )
        download_flag = args.download_data or missing_cache
        if missing_cache and not args.download_data:
            print('Local CIFAR-10 cache missing. Downloading...')
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
        try:
            trainset = torchvision.datasets.CIFAR10(
                root=args.data_root,
                train=True,
                download=download_flag,
                transform=transform_train,
            )
            testset = torchvision.datasets.CIFAR10(
                root=args.data_root,
                train=False,
                download=download_flag,
                transform=transform_test,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                'CIFAR-10 not found. Supply --download_data or place files under data_root/cifar-10-batches-py.'
            ) from exc
    else:
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = torchvision.datasets.FakeData(
            size=5000,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=base_transform,
            random_offset=0,
        )
        testset = torchvision.datasets.FakeData(
            size=1000,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=base_transform,
            random_offset=1,
        )

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    testloader = DataLoader(
        testset,
        batch_size=min(256, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return trainloader, testloader, trainset


def build_analysis_loader(trainset, args):
    subset_size = min(len(trainset), args.analysis_samples)
    subset = Subset(trainset, list(range(subset_size)))
    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset only')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['synthetic', 'cifar10'])
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--analysis_samples', type=int, default=512)
    parser.add_argument('--download_data', action='store_true')
    args = parser.parse_args()

    prepare_dataloaders(args)
