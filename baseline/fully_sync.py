import copy
import random
import numpy as np
import torch

from environment import Environment
from options import args_parser
from client import AsyncFed, Client, SyncGroup, test_inference
from resnet import resnet18
from trajectory import Trajectory
from datasets import get_dataset
from utils import CosineAnnealingDecay, GroupClusterer, Logger, StepDecay

if __name__ == '__main__':
    opts = args_parser()

    if opts.fixed_seed:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if opts.use_mp:
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

        torch.multiprocessing.set_start_method('spawn')
        torch.multiprocessing.set_sharing_strategy('file_system')

    logger = Logger(opts)
    global_model = resnet18(opts.n_cls).to(opts.device)
    env = Environment(global_model, opts)
    traj = Trajectory(env)
        
    user_datasets, test_dataset = get_dataset(env)

    grouped_users = set()
    ws = []
    eps = []
    leaving_times = []
    max_leaving_time = 0
    final_last_leaving_idx = 0
    for current_time in range(traj.max_time()):
        in_range_users, _ = traj.get_indices_in_range(current_time)
        ungrouped_users = set(in_range_users) - grouped_users

        indices = list(ungrouped_users)
        exist_times = traj.get_exist_times(indices, current_time)
        times, indices = np.array(exist_times), np.array(indices)

        computing_times = times / 10 - env.T_com[indices] # 0.1second to second
        user_eps = np.floor_divide(computing_times, env.T_itr[indices]).astype(int)
        assert np.all(user_eps >= 0), f"error user_eps: {user_eps}"

        is_beyond_range = traj.get_beyond_range(indices)
        for i, id in enumerate(indices):
            worker = Client(copy.deepcopy(global_model), user_datasets[id], env)
            w = worker.train(user_eps[i])

            test_acc, test_loss = test_inference(worker.model, test_dataset)
            logger.log_dict['local models']['accuracies'].append(test_acc)
            logger.log_dict['local models']['losses'].append(test_loss)

            if not is_beyond_range[i]:
                ws.append(w)
                eps.append(user_eps[i])

                leaving_times.append(times[i] + current_time)
                if times[i] + current_time > max_leaving_time:
                    final_last_leaving_idx = indices[i]
                    max_leaving_time = times[i] + current_time

        grouped_users = grouped_users | ungrouped_users

    total_ep = np.sum([eps[i] for i in range(len(user_eps))])
    w_avg = {key: sum(eps[i] / total_ep * ws[i][key] for i in range(len(w))) for key in w[0].keys()}
    
    global_model.load_state_dict(w_avg)

    saved_time = np.sum(env.T_com) - env.T_com[final_last_leaving_idx]
    delay = np.sum(max_leaving_time - np.array(leaving_times))
    logger.log_dict['saved communication time'] += saved_time.item()
    logger.log_dict['delay by async'] += delay.item()

    # SyncGroup.pool.close()
    logger.store()
    
