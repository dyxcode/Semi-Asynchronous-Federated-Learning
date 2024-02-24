#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
import torch

from environment import Environment
from options import args_parser
from client import AsyncFed, SyncGroup
from resnet import resnet18
from trajectory import Trajectory
from datasets import get_dataset
from utils import CosineAnnealingDecay, GroupClusterer, Logger, StepDecay, find_approximate_lcm

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
    clusterer = GroupClusterer()
    
    user_datasets, test_dataset = get_dataset(env)
    lr_scheduler = StepDecay(max_v=env.lr, step_size=5, gamma=0.5)
    alpha_scheduler = CosineAnnealingDecay(max_v=env.alpha_max, min_v=env.alpha_min, total_epochs=traj.max_time())

    async_fed = AsyncFed(global_model, test_dataset, logger.log_dict)

    last_ungrouped_users = set()
    grouped_users = set()
    for current_time in range(traj.max_time()):
        # if (current_time > 1000): break
        update_t = async_fed.update_weight(current_time, alpha_scheduler.get(current_time))
        env.lr = lr_scheduler.get(update_t)

        in_range_users, ungrouped_users = traj.get_indices_in_range(current_time)
        ungrouped_users = set(ungrouped_users) - grouped_users

        # if there are users out of range
        if last_ungrouped_users - ungrouped_users:
            print(f"Time {current_time}/{traj.max_time()}:")
            assert not (last_ungrouped_users & grouped_users),\
                f"error users: {last_ungrouped_users & grouped_users}"
            # all last ungrouped users will be grouped
            grouped_users = grouped_users | last_ungrouped_users
            ungrouped_users = ungrouped_users - last_ungrouped_users

            group_indices = list(last_ungrouped_users)
            exist_times = traj.get_exist_times(group_indices, current_time)

            groups = clusterer.auto_cluster(exist_times, group_indices)

            for times, indices in groups:
                print(f'new_grouped_users: {indices}', end=' ')
                s_group = SyncGroup(env, global_model, user_datasets, indices, test_dataset, logger.log_dict)

                times, indices = np.array(times), np.array(indices)
                computing_times = times / 10 - env.T_com[indices] # 0.1second to second

                last_leaving_idx = np.argmax(times)
                is_beyond_range = traj.get_beyond_range(indices)
                print(f'beyond range users: {indices[is_beyond_range]}')

                if not is_beyond_range[last_leaving_idx]:
                    computing_times[last_leaving_idx] -= env.T_com[indices][last_leaving_idx]

                    not_beyond_indices = np.where(is_beyond_range == False)[0]
                    saved_time = np.sum(env.T_com[indices][not_beyond_indices]) - env.T_com[indices][last_leaving_idx]
                    delay = np.sum(times[last_leaving_idx] - times[not_beyond_indices])
                    logger.log_dict['saved communication time'] += saved_time.item()
                    logger.log_dict['delay by sync'] += delay.item()

                iters = env.T_itr[indices]
                lcm = find_approximate_lcm(iters, np.min(computing_times))
                eps = np.floor_divide(lcm, iters).astype(int)

                now_is_beyond_range = np.zeros_like(is_beyond_range, dtype=bool)
                total_ep = 0
                while True:
                    print(f'user_eps: {eps}', end=' ')
                    total_ep += np.sum(eps)
                    if np.all(now_is_beyond_range) or np.all(is_beyond_range[np.where(eps != 0)]):
                        s_group.update_weights(eps, now_is_beyond_range)
                    else:
                        sync_w = s_group.update_weights(eps, now_is_beyond_range)

                    computing_times = np.maximum(computing_times - lcm, 0)
                    about_to_leave = np.where(computing_times < lcm)
                    eps[about_to_leave] = np.floor_divide(computing_times[about_to_leave], iters[about_to_leave]).astype(int)
                    if np.all(eps == 0): break

                    s_group.load_weights(sync_w)
                    now_is_beyond_range[about_to_leave] = is_beyond_range[about_to_leave]
                    
                s_group.log()
                if not is_beyond_range[last_leaving_idx]:
                    avg_ep = total_ep / indices.size
                    async_fed.put_weight(sync_w, current_time + times[last_leaving_idx], avg_ep)

        last_ungrouped_users = ungrouped_users
        assert (set(in_range_users) - ungrouped_users).issubset(grouped_users),\
            f"error users: {set(in_range_users) - ungrouped_users - grouped_users}"

    SyncGroup.pool.close()
    logger.store()
    
