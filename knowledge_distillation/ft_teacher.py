from torchvision import datasets
import torch
from torch import nn
from timm.utils import accuracy
from datasets import DinoV2Dataset
from models import CustomDinoV2

device = 'cuda'

trainset = DinoV2Dataset(datasets.CIFAR10(root='./data/cifar10', train=True, download=True))
testset = DinoV2Dataset(datasets.CIFAR10(root='./data/cifar10', train=False, download=True))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 加载模型和处理器
model = CustomDinoV2(n_cls=10)
model = model.to(device)

model_path = 'dinov2_cifar10_ft.pt'
total_epoch = 100
best_acc = 0.0

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(total_epoch):
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()


    model.eval()
    top1, top5 = 0., 0.
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1 += acc1.item() * inputs.size(0)
            top5 += acc5.item() * inputs.size(0)
            total += labels.size(0)

    top1, top5 = top1 / total, top5 / total
    if top1 > best_acc:
        best_acc = top1
        torch.save(model.state_dict(), model_path)
        print(f'Epoch: {epoch}/{total_epoch}, Top 1: {top1:.2f}%, Top 5: {top5:.2f}%')
