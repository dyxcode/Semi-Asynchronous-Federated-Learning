# --------------------------------------------------------
# TinyViT Save Teacher Logits
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Save teacher logits
# --------------------------------------------------------
from torchvision import datasets

import random
import argparse
from collections import defaultdict
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from datasets import DinoV2Dataset, DatasetWrapper
from models import CustomDinoV2

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_logits_path', default='./teacher_logits_cifar10')
    parser.add_argument('--topk', default=5)
    args = parser.parse_args()
    return args


def main(args):
    model = CustomDinoV2(n_cls=10)
    model_path = 'dinov2_cifar10_ft.pt'
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"number of params: {n_parameters}")

    # if not args.skip_eval and not args.check_saved_logits:
    #     acc1, acc5, loss = validate(config, data_loader_val, model)
    #     logger.info(
    #         f"Accuracy of the network on the {len(dataset_val)} test images: top-1 acc: {acc1:.1f}%, top-5 acc: {acc5:.1f}%")

    # if args.check_saved_logits:
    #     logger.info("Start checking logits")
    # else:
    #     logger.info("Start saving logits")

    #     if args.check_saved_logits:
    #         check_logits_one_epoch(
    #             args, model, trainloader)
    #     else:
    #         save_logits_one_epoch(
    #             args, model, trainloader)
    # save_logits(args, model)
    check_logits(args, model)

@torch.no_grad()
def save_logits(config, model):
    trainset = DinoV2Dataset(datasets.CIFAR10(root='./data/cifar10', train=True, download=True))
    dataset_train = DatasetWrapper(trainset,
                            logits_path=args.teacher_logits_path,
                            topk=args.topk,
                            write=True)
    data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)

    model.eval()
    topk = config.topk

    logits_manager = data_loader.dataset.get_manager()

    for (samples, targets), keys in data_loader:
        samples = samples.cuda()
        targets = targets.cuda()

        outputs = model(samples)

        # save teacher logits
        softmax_prob = torch.softmax(outputs, -1)

        values, indices = softmax_prob.topk(
            k=topk, dim=-1, largest=True, sorted=True)

        cpu_device = torch.device('cpu')
        values = values.detach().to(device=cpu_device, dtype=torch.float32)
        indices = indices.detach().to(device=cpu_device, dtype=torch.int32)

        values = values.numpy()
        indices = indices.numpy()

        # check data type
        assert indices.dtype == np.int32, indices.dtype
        assert values.dtype == np.float32, values.dtype

        for key, indice, value in zip(keys, indices, values):
            print(f'key: {key}, indice: {indice}, value: {value}')
            bstr = indice.tobytes() + value.tobytes()
            logits_manager.write(key, bstr)

@torch.no_grad()
def check_logits(config, model):
    trainset = DinoV2Dataset(datasets.CIFAR10(root='./data/cifar10', train=True, download=True))
    dataset_train = DatasetWrapper(trainset,
                            logits_path=args.teacher_logits_path,
                            topk=args.topk,
                            write=False)
    data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)

    model.eval()
    topk = config.topk

    for (samples, targets), (saved_logits_index, saved_logits_value) in data_loader:
        samples = samples.cuda()
        targets = targets.cuda()

        outputs = model(samples)

        softmax_prob = torch.softmax(outputs, -1)

        values, indices = softmax_prob.topk(
            k=topk, dim=-1, largest=True, sorted=True)

        saved_logits_value = saved_logits_value.float().cuda()
        saved_logits_index = saved_logits_index.int32().cuda()

        assert torch.allclose(values, saved_logits_value),\
            f"values and saved_value are not equal, values: {values}, saved_value: {saved_logits_value}"

        assert torch.equal(indices, saved_logits_index),\
            f"indices and saved_index are not equal, indices: {indices}, saved_index: {saved_logits_index}"


if __name__ == '__main__':
    args = parse_option()
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    main(args)
