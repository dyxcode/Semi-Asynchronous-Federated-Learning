import pandas as pd
import numpy as np

from environment import Environment
from resnet import resnet18

# def select_trajectories(env):
#     df = pd.read_csv('./trajectory_prediction/data/adjust_i-80.csv', usecols=['New_ID', 'Local_X', 'Local_Y'])
#     grouped = df.groupby('New_ID')
#     all_indices = df['New_ID'].unique()

#     existing_indices = []
#     trajectories = []
        
#     while len(existing_indices) < env.n_users:
#         random_indices = np.random.choice(all_indices, (env.n_users - len(existing_indices)) * 2, replace=False)
#         for i in random_indices:
#             trajectory = grouped.get_group(i)[['Local_X', 'Local_Y']].values
#             distances = np.linalg.norm(trajectory - np.array([env.Ex, env.Ey]), axis=1)
#             if distances[0] > env.Er and distances[-1] > env.Er and np.any(distances <= env.Er):
#                 existing_indices.append(i)
#                 trajectories.append(trajectory)
#                 if len(existing_indices) == env.n_users:
#                     break
#         all_indices = np.setdiff1d(all_indices, existing_indices)
        
#     return trajectories

# class Trajectory(object):
#     """
#         Trajectory class
#     """

#     def __init__(self, env: Environment):
#         self.env = env
#         self.trajectories = select_trajectories(env)
#         start_time = np.random.randint(0, 2000, size=len(self.trajectories))
#         self.trajectories = [np.vstack((np.zeros((n, 2)), t))
#                                 for n, t in zip(start_time, self.trajectories)]
    
#     def get_indices_in_range(self, current_time):
#         current_pos = [trajectory[current_time] if current_time < len(trajectory) else [np.inf, np.inf]
#                                 for trajectory in self.trajectories]
        
#         edge_pos = np.array([self.env.Ex, self.env.Ey])
#         indices_in_range = [i for i, pos in enumerate(current_pos)
#                     if np.linalg.norm(pos - edge_pos) <= self.env.Er]
        
#         indices_in_tol_range = [i for i in indices_in_range
#                     if np.linalg.norm(current_pos[i] - edge_pos) >= self.env.Er - self.env.l_tol]
        
#         return indices_in_range, indices_in_tol_range
    
#     def get_exist_times(self, indices, current_time):
#         exist_times = []
#         for i in indices:
#             for t, pos in enumerate(self.trajectories[i][current_time:]):
#                 if np.linalg.norm(pos - np.array([self.env.Ex, self.env.Ey])) > self.env.Er:
#                     exist_times.append(t)
#                     break
#         return exist_times
    
#     def max_time(self):
#         return max([len(x) for x in self.trajectories])

def load_trajectories(env):
    traj_df = pd.read_csv('./trajectory_prediction/data/filter_i-80.csv', usecols=['New_ID', 'Local_X', 'Local_Y'])
    traj_dict = {id: group[['Local_X', 'Local_Y']].values for id, group in traj_df.groupby('New_ID')}

    df = pd.read_csv(f'./trajectory_prediction/data/{env.predict_model}_predictions.csv')
    result_dict = df.set_index('New_ID').T.to_dict()
    for id, traj in traj_dict.items():
        distances = np.linalg.norm(traj - np.array([env.Ex, env.Ey]), axis=1)
        enter_index, leave_index = result_dict[id]['enter_index'], result_dict[id]['leave_index']
        traj_distances = distances[enter_index:leave_index]
        indices = np.where(traj_distances < env.Er - env.l_tol)[0]
        assert indices.size > 0
        result_dict[id].update({'tol_index': indices[0] + enter_index - 1})

    start_times = np.random.randint(0, 2000, len(result_dict))
    result_list = [{prop: value + start_time for prop, value in properties.items()} 
                    for properties, start_time in zip(result_dict.values(), start_times)]

    min_enter_index = min(item['enter_index'] for item in result_list)
    result_list = [{prop: value - min_enter_index for prop, value in properties.items()} 
                    for properties in result_list]
    return result_list

class Trajectory(object):
    """
        Trajectory class
    """

    def __init__(self, env: Environment):
        self.result_list = load_trajectories(env)
        self.index_key = 'predict_index' if env.use_predict else 'leave_index'

        if env.use_predict:
            self.beyond_range = np.array([traj['predict_index'] > traj['leave_index'] for traj in self.result_list])
        else:
            self.beyond_range = np.zeros(env.n_users, dtype=bool)
        
    def get_indices_in_range(self, current_time):
        indices_in_range = [i for i, props in enumerate(self.result_list)
            if props['enter_index'] <= current_time and props[self.index_key] >= current_time]

        indices_in_tol_range = [i for i in indices_in_range
            if self.result_list[i]['tol_index'] >= current_time]
        
        return indices_in_range, indices_in_tol_range
    
    def get_exist_times(self, indices, current_time):
        return [self.result_list[i][self.index_key] - current_time + 1 for i in indices]
    
    def get_beyond_range(self, indices):
        return self.beyond_range[indices]
    
    def max_time(self):
        return max([props[self.index_key] for props in self.result_list])


if __name__ == '__main__':
    # env = Environment(100, 'mnist', resnet18(10))
    # traj = Trajectory(env)
    load_trajectories(None)