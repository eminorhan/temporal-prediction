import argparse
import os
import random
import shutil
import time
import warnings
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


parser = argparse.ArgumentParser(description='Cache IntPhys embeddings')
parser.add_argument('--workers', default=32, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--batch-size', default=125, type=int, metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--model', default='say', type=str, choices=['say', 'in', 'spatial', 'wsl', 'rand', 'robust'], help='which model to use for caching')
parser.add_argument('--data-dir', default='', type=str, metavar='PATH', help='path to data (default: none)')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', help='launch N processes per node, which has N GPUs.')


def main():
    args = parser.parse_args()
    print(args)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.model == 'say':
        model = models.resnext50_32x4d(pretrained=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=6269, bias=True)
        model = torch.nn.DataParallel(model).cuda()

        model_path = '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/self_supervised_models/TC-SAY-resnext.tar'

        if os.path.isfile(model_path):
            print("=> loading model: '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

        model.module.fc = torch.nn.Identity()  # dummy layer
    elif args.model == 'robust':
        from robustness import model_utils
        from robustness.datasets import ImageNet
        imagenet_ds = ImageNet('/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/')
        arch = models.resnext50_32x4d()
        model_path = 'resnext50_32x4d_l2_eps3.ckpt'  # to be changed later
        model, _ = model_utils.make_and_restore_model(arch=arch, dataset=imagenet_ds, resume_path=model_path, parallel=True, add_custom_forward=True)
        model = model.module.model.model
        model = torch.nn.DataParallel(model).cuda() 
        model.module.fc = torch.nn.Identity()  # dummy layer
    elif args.model == 'in':
        print('=> loading the ImageNet pre-trained model')
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = torch.nn.Identity()  # dummy layer
        model = torch.nn.DataParallel(model).cuda()
    elif args.model == 'spatial':
        print('=> spatial kmeans features')
        model = models.resnext50_32x4d(pretrained=True)
        layer_list = list(model.children())[:-2]  # remove avgpool and fc
        model = torch.nn.Sequential(*layer_list)
    elif args.model == 'wsl':
        torch.hub.set_dir('/misc/vlgscratch4/LakeGroup/emin/robust_vision/pretrained_models')
        print('=> loading the WSL pre-trained model')
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')        
        model.fc = torch.nn.Identity()  # dummy layer
        model = torch.nn.DataParallel(model).cuda()
    elif args.model == 'rand':
        print('=> loading the untrained model')
        model = models.resnext50_32x4d(pretrained=False)
        model.fc = torch.nn.Identity()  # dummy layer
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        args.data_dir,
        transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    print('Dataset size:', len(train_dataset))
    print('Loader size:', len(train_loader))

    embeddings, components = evaluate(train_loader, model, args)
    
    np.savez('adept_train_' + args.model, x=embeddings, W=components)

    return

def evaluate(data_loader, model, args):

    embeddings = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.model == 'spatial':
                embedding = model(images).permute(0, 2, 3, 1) #.reshape(-1, 2048)
            else:
                embedding = model(images) 
            embedding = embedding.cpu().numpy()
            
            print(i, embedding.shape)

            embeddings.append(np.expand_dims(embedding, 0))


    embeddings = np.concatenate(embeddings)
    print('Embeddings shape:', embeddings.shape)

    components = np.eye(2048)  # dummy

    return embeddings, components


if __name__ == '__main__':
    main()
