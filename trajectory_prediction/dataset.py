import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TrajectoryDataset(Dataset):
    def __init__(self, data, lag, horizon):
        self.data = data
        self.lag = lag
        self.horizon = horizon

    def __getitem__(self, index):
        # 获取训练数据和标签
        assert len(self.data[index]) == self.lag + self.horizon
        x = torch.tensor(self.data[index][:self.lag], dtype=torch.float32)  # shape=[self.lag, 2]
        y = torch.tensor(self.data[index][self.lag:], dtype=torch.float32)  # shape=[self.horizon, 2]   
        return x, y
    
    def __len__(self):
        return len(self.data)


# def train_valid_test_split(data, SplitPercentage: list[float]):
#     train_bound = int(len(data) * SplitPercentage[0])
#     valid_bound = int(len(data) * (SplitPercentage[0] + SplitPercentage[1]))
#     return data[:train_bound], data[train_bound:valid_bound], data[valid_bound:]

# def get_dataset(Lag: int, Horizon: int, Batch_size: int):
#     df = pd.read_csv('./data/filter_i-80.csv', usecols=['New_ID', 'Local_X', 'Local_Y'])

#     # 0.1s to 0.4s, every 4 items average 1
#     df = df.groupby('New_ID').apply(lambda x: x.rolling(4).mean().iloc[3::4]).reset_index(drop=True)
    
#     x_scaler = StandardScaler()
#     y_scaler = StandardScaler()

#     df['x'] = x_scaler.fit_transform(df[['Local_X']])
#     df['y'] = y_scaler.fit_transform(df[['Local_Y']])

#     piece_length = Lag + Horizon
    
#     data_path = Path(f'./data/extract_{Lag}_{Horizon}_data.npy')
#     if data_path.exists():
#         final_data = np.load(data_path)
#         print(f'Load from exist file, there are {final_data.shape[0]} items in total\n')
#     else:
#         print('Start extracting data...')
#         final_data = []
#         for _, group in tqdm(df.groupby('New_ID')):
#             group_values = group[['x','y']].values
#             for i in range(len(group_values) - piece_length + 1):
#                 final_data.append(group_values[i: i + piece_length])
#         final_data = np.stack(final_data)   # shape=[n, piece_length, 2]
#         np.random.shuffle(final_data)
#         np.save(f'./data/extract_{Lag}_{Horizon}_data.npy', final_data)
#         print(f'Extraction completed, there are {final_data.shape[0]} items in total\n')

#     train_data, valid_data, test_data = train_valid_test_split(final_data, [0.7, 0.1, 0.2])
#     train_dataset = TrajectoryDataset(train_data, lag=Lag, horizon=Horizon)
#     valid_dataset = TrajectoryDataset(valid_data, lag=Lag, horizon=Horizon)
#     test_dataset = TrajectoryDataset(test_data, lag=Lag, horizon=Horizon)

#     train_loader = DataLoader(train_dataset, batch_size=Batch_size)
#     valid_loader = DataLoader(valid_dataset, batch_size=Batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=Batch_size)
    
#     return train_loader, valid_loader, test_loader, x_scaler, y_scaler

def get_dataset(trajectory, Ex, Ey, Er, stride, Lag, Horizon, Batch_size: int):
    distances = np.linalg.norm(trajectory - np.array([Ex, Ey]), axis=1)
    inside_circle_indices = np.where(distances <= Er)[0]
    enter_index = inside_circle_indices[0]
    leave_index = inside_circle_indices[-1]

    piece_length = Lag + Horizon
    final_data = []
    for i in range(stride):
        stride_traj = trajectory[i::stride]
        diff_traj = np.diff(stride_traj, axis=0)

        if (enter_index - i) % stride == 0:
            n = (enter_index - i) // stride
            inputs = diff_traj[n - 1 - Lag: n - 1]
            targets = diff_traj[n - 1 : n - 1 + Horizon]
            init_value = stride_traj[n - 1]

        for i in range(len(diff_traj) - piece_length + 1):
            final_data.append(diff_traj[i: i + piece_length])

    final_data = np.stack(final_data)   # shape=[n, piece_length, 2]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    final_data[:,:,0] = x_scaler.fit_transform(final_data[:,:,0].reshape(-1, 1)).reshape(final_data[:,:,0].shape)
    final_data[:,:,1] = y_scaler.fit_transform(final_data[:,:,1].reshape(-1, 1)).reshape(final_data[:,:,1].shape)
    
    # print(final_data.shape[0])

    train_dataset = TrajectoryDataset(final_data, lag=Lag, horizon=Horizon)
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    
    return train_loader, inputs, targets, init_value, x_scaler, y_scaler, enter_index, leave_index

def get_trajectories():
    df = pd.read_csv('./data/filter_i-80.csv', usecols=['New_ID', 'Local_X', 'Local_Y'])
    
    # def process_group(group):
    #     return group[['Local_X','Local_Y']].rolling(4).mean().iloc[3::4].values
    # trajectories = [(id, process_group(group)) for id, group in df.groupby('New_ID')]

    trajectories = [(id, group[['Local_X','Local_Y']].values) for id, group in df.groupby('New_ID')]

    return trajectories

# def get_trajectories(Ex, Ey, Er, Lag, Horizon):
#     df = pd.read_csv('./data/adjust_i-80.csv', usecols=['New_ID', 'Local_X', 'Local_Y'])
#     df = df.groupby('New_ID').apply(lambda x: x.rolling(10).mean().iloc[9::10]).reset_index(drop=True)

#     x_scaler = StandardScaler()
#     y_scaler = StandardScaler()

#     df['x'] = x_scaler.fit_transform(df[['Local_X']])
#     df['y'] = y_scaler.fit_transform(df[['Local_Y']])

#     grouped = df.groupby('New_ID')
#     trajectories = dict()

#     timesheet_path = Path(f'./data/timesheet_{Ex}_{Ey}_{Er}.csv')
#     if timesheet_path.exists():
#         timesheet = pd.read_csv(timesheet_path, index_col='New_ID').to_dict(orient='index')
#         for New_ID in timesheet.keys():
#             group = grouped.get_group(New_ID)
#             trajectories[New_ID] = {
#                 'trajectory': group[['Local_X', 'Local_Y']].values,
#                 'scaler_traj': group[['x', 'y']].values,
#             }
#     else:
#         timesheet = dict()
#         for group in grouped:
#             trajectory = group[1][['Local_X', 'Local_Y']].values
#             distances = np.linalg.norm(trajectory - np.array([Ex, Ey]), axis=1)
#             if distances[0] > Er and distances[-1] > Er and np.any(distances <= Er):
#                 inside_circle_indices = np.where(distances <= Er)[0]
#                 assert np.all(np.diff(inside_circle_indices) <= 3)
#                 enter_index = inside_circle_indices[0]
#                 leave_index = inside_circle_indices[-1]
#                 if enter_index < Lag or enter_index + Horizon < leave_index - 1:
#                     continue

#                 timesheet[group[0]] = {
#                     'enter_index': enter_index,
#                     'leave_index': leave_index,
#                 }
#                 trajectories[group[0]] = {
#                     'trajectory': trajectory,
#                     'scaler_traj': group[1][['x', 'y']].values,
#                 }

#         df = pd.DataFrame(timesheet).T
#         df.index.name = 'New_ID'
#         df.reset_index(inplace=True)
#         df['New_ID'] = df['New_ID'].astype(int)
#         df.to_csv(timesheet_path, index=False)
        
#     return trajectories, timesheet, x_scaler, y_scaler

if __name__ == '__main__':
    get_trajectories(40, 250, 50)