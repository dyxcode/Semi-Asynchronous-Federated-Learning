# class DistillKL(nn.Module):
#     """Distilling the Knowledge in a Neural Network"""
#     def __init__(self, T):
#         super(DistillKL, self).__init__()
#         self.T = T

#     def forward(self, y_s, y_t):
#         p_s = F.log_softmax(y_s/self.T, dim=1)
#         p_t = F.softmax(y_t/self.T, dim=1)
#         loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
#         return loss
import os
import yaml
from yaml.representer import SafeRepresenter
from datetime import datetime
from sklearn.cluster import KMeans
import numpy as np

class CosineAnnealingDecay:
    def __init__(self, max_v, min_v, total_epochs):
        self.max_v = max_v
        self.min_v = min_v
        self.total_epochs = total_epochs

    def get(self, current_epoch):
        cos_inner = np.pi * current_epoch / self.total_epochs
        return self.min_v + (self.max_v - self.min_v) / 2 * (1 + np.cos(cos_inner))
    
class StepDecay:
    def __init__(self, max_v, step_size, gamma=0.1):
        self.max_v = max_v
        self.step_size = step_size
        self.gamma = gamma
        self.last_t = 0

    def get(self, current_epoch):
        if current_epoch - self.last_t == self.step_size:
            self.max_v *= self.gamma
            self.last_t = current_epoch
        return self.max_v
    
class GroupClusterer():
    def __init__(self) -> None:
        self.kmeans = KMeans(n_init=10)

    def auto_cluster(self, samples, indices):
        samples = [[i] for i in samples]
        
        sorted_samples = sorted(samples)
        diff = [sorted_samples[i+1][0] - sorted_samples[i][0] for i in range(len(sorted_samples)-1)]
        optimal_k = sum(i > 50 for i in diff) + 1

        labels = self.kmeans.set_params(n_clusters=optimal_k).fit(samples).labels_

        # (samples, indices)
        clusters = [([], []) for _ in range(optimal_k)]

        for sample, index, label in zip(samples, indices, labels):
            clusters[label][0].append(sample[0])
            clusters[label][1].append(index)

        return clusters
    
def find_approximate_lcm(arr, end):
    start = float(np.ceil(np.max(arr)*10)) / 10
    best_lcm = start
    min_remainder = float('inf')
    for i in np.arange(start, end, 0.1):
        remainder = np.sum(i % arr)
        if remainder < min_remainder:
            min_remainder = remainder
            best_lcm = i
    return best_lcm

class CustomRepresenter(SafeRepresenter):
    def represent_list(self, data):
        return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

class Logger():
    def __init__(self, opts):
        yaml.add_representer(list, CustomRepresenter.represent_list)
        yaml.add_multi_representer(object, SafeRepresenter.represent_undefined)

        self.log_dict = {
            'options': str(opts),
            'global model': {
                'accuracies': [],
                'losses': [],
                'time slots': []
            },
            'local models': {
                'accuracies': [],
                'losses': [],
                'indices': []
            },
            'saved communication time': 0,
            'delay by sync': 0
        }

    def store(self):
        os.makedirs('./result', exist_ok=True)
        with open(f'./result/{datetime.now().strftime("%m-%d-%H-%M")}.yaml', 'w') as f:
            yaml.dump(self.log_dict, f)