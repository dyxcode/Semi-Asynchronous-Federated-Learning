#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from timm.loss import SoftTargetCrossEntropy
from collections import defaultdict

class Client(object):
    def __init__(self, model: nn.Module, dataset: Dataset, env):
        self.model = model
        self.trainloader = DataLoader(dataset, batch_size=env.local_bs)
        self.lr = env.lr
        self.topk = env.topk
        self.n_cls = env.n_cls
        self.use_sl = env.use_softlabel

    def train(self, local_ep):
        # Set mode to train model
        self.model.train()
        device = next(self.model.parameters()).device

        # Set optimizer for the local updates
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        # criterion = nn.CrossEntropyLoss().to(device)
        criterion = (SoftTargetCrossEntropy() if self.use_sl else nn.CrossEntropyLoss()).to(device)

        for i in range(local_ep):
            batch_loss = []
            # start_time = time.perf_counter()
            for (images, labels), (logits_index, logits_value) in self.trainloader:
                images = images.to(device)

                if self.use_sl:
                    logits_index = logits_index.to(device, dtype=torch.int64)
                    logits_value = logits_value.to(device, dtype=torch.float32)

                    minor_value = (1.0 - logits_value.sum(-1, keepdim=True)
                                ) / (self.n_cls - self.topk)
                    minor_value = minor_value.repeat_interleave(self.n_cls, dim=-1)
                    targets = minor_value.scatter_(-1, logits_index, logits_value)
                else:
                    targets = labels.to(device)
                    
                self.model.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            # end_time = time.perf_counter()
            # print(f"Epoch {iter}运行时间: {end_time - start_time}秒")
            if i == 0: print(f"Init Loss: {np.mean(batch_loss)},", end=' ')
            if i == local_ep - 1: print(f"Final Loss: {np.mean(batch_loss)}")

        return {k: v.cpu() for k, v in self.model.state_dict().items()}


def helper_mp_train(client: Client, ep: int):
    return client.train(ep)

class SyncGroup(object):
    pool = None

    def __init__(self, env, global_model: nn.Module, user_datasets: list[Dataset], grouped_users: list[int], test_dataset, log_dict):
        self.clients = [Client(copy.deepcopy(global_model), user_datasets[i], env) for i in grouped_users]
        self.use_mp = env.use_mp
        self.test_dataset = test_dataset
        self.log_dict = log_dict

        log_dict['local models']['indices'].extend(grouped_users)

    @staticmethod
    def get_pool(processes):
        if SyncGroup.pool is None:
            SyncGroup.pool = torch.multiprocessing.Pool(processes=processes)
        return SyncGroup.pool
    
    def load_weights(self, w):
        for client in self.clients:
            client.model.load_state_dict(w)

    def update_weights(self, user_eps, is_beyond_range):
        # print(f"start training...")
        if self.use_mp:
            pool = self.get_pool(4)
            w = pool.starmap(helper_mp_train, zip(self.clients, user_eps))
        else:
            w = [client.train(user_eps[i]) for i, client in enumerate(self.clients)]

        if np.all(is_beyond_range):
            return 0
        
        device = next(self.clients[0].model.parameters()).device
        total_ep = np.sum([user_eps[i] for i in range(len(user_eps)) if not is_beyond_range[i]])
        w_avg = {key: sum(user_eps[i] / total_ep * w[i][key] for i in range(len(w)) if not is_beyond_range[i]).to(device)
                for key in w[0].keys()}
        return w_avg
    
    def log(self):
        for client in self.clients:
            test_acc, test_loss = test_inference(client.model, self.test_dataset)
            self.log_dict['local models']['accuracies'].append(test_acc)
            self.log_dict['local models']['losses'].append(test_loss)
    
class AsyncFed(object):
    def __init__(self, global_model: nn.Module, test_dataset: Dataset, log_dict: dict):
        self.global_model = global_model
        self.test_dataset = test_dataset
        self.map = defaultdict(list)
        self.update_t = 0
        self.log_dict = log_dict

    def put_weight(self, w, aggregation_t, avg_ep):
        self.map[aggregation_t].append((self.update_t, w, avg_ep))

    def update_weight(self, current_t, alpha):
        if current_t in self.map:
            self.map[current_t] = sorted(self.map[current_t], key=lambda x: x[2], reverse=True)
            for tao, w, _ in self.map[current_t]:
                alpha_t = alpha * staleness(self.update_t, tao)
                if self.update_t == tao:
                    self.update_t = self.update_t + 1
                # self.update_t = self.update_t + 1

                global_w = self.global_model.state_dict()
                for k in global_w.keys():
                    global_w[k] = global_w[k] * (1-alpha_t) + w[k] * alpha_t
                self.global_model.load_state_dict(global_w)

                test_acc, test_loss = test_inference(self.global_model, self.test_dataset)
                print(f'Update at {current_t}, Accuracy: {100*test_acc:.2f}%, Alpha: {alpha}, Alpha_t: {alpha_t}')
                self.log(test_acc, test_loss, current_t)

            del self.map[current_t]
        return self.update_t
    
    def log(self, acc, loss, time):
        self.log_dict['global model']['accuracies'].append(acc)
        self.log_dict['global model']['losses'].append(loss)
        self.log_dict['global model']['time slots'].append(time)

def staleness(t, tao):
    return 1 / (t - tao + 1)
    # return 1 / np.exp(t - tao)
    # return 1 / np.power(t - tao + 1, 2)
    # return 1 / np.power(t - tao + 1, 3)
    
def test_inference(model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0
    total, correct = 0.0, 0.0

    testloader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = nn.functional.cross_entropy(outputs, labels)
        total_loss += loss.item() * labels.size(0)

        _, pred_labels = torch.max(outputs, 1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += labels.size(0)

    accuracy = correct/total
    avg_loss = total_loss/total
    return accuracy, avg_loss