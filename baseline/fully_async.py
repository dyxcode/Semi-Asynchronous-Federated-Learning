import copy
import random
import numpy as np
import torch

from environment import Environment
from options import args_parser
from client import AsyncFed, Client, test_inference
from resnet import resnet18
from trajectory import Trajectory
from datasets import get_dataset
from utils import CosineAnnealingDecay, Logger, StepDecay

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
    lr_scheduler = StepDecay(max_v=env.lr, step_size=5, gamma=0.5)
    alpha_scheduler = CosineAnnealingDecay(max_v=env.alpha_max, min_v=env.alpha_min, total_epochs=traj.max_time())

    async_fed = AsyncFed(global_model, test_dataset, logger.log_dict)

    started_users = set()
    for current_time in range(traj.max_time()):
        print(f"Time {current_time}/{traj.max_time()}:")

        update_t = async_fed.update_weight(current_time, alpha_scheduler.get(current_time))
        env.lr = lr_scheduler.get(update_t)

        in_range_users, _ = traj.get_indices_in_range(current_time)
        unstarted_users = set(in_range_users) - started_users

        indices = list(unstarted_users)
        logger.log_dict['local models']['indices'].extend(indices)
        print(f'new started users: {indices}')

        exist_times = traj.get_exist_times(indices, current_time)
        times, indices = np.array(exist_times), np.array(indices)
        is_beyond_range = traj.get_beyond_range(indices)

        computing_times = times / 10 - 2*env.T_com[indices]
        user_eps = np.floor_divide(computing_times, env.T_itr[indices]).astype(int)
        assert np.all(user_eps >= 0), f"error user_eps: {user_eps}"
        print(f'user_eps: {user_eps}')

        for i, id in enumerate(indices):
            worker = Client(copy.deepcopy(global_model), user_datasets[id], env)
            w = worker.train(user_eps[i])

            test_acc, test_loss = test_inference(worker.model, test_dataset)
            logger.log_dict['local models']['accuracies'].append(test_acc)
            logger.log_dict['local models']['losses'].append(test_loss)

            if not is_beyond_range[i]:
                async_fed.put_weight(w, current_time + times[i], user_eps[i])

        started_users = started_users | unstarted_users

    # SyncGroup.pool.close()
    logger.store()
    
