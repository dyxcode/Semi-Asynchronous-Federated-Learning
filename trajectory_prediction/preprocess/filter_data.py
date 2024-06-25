import pandas as pd
import numpy as np

df = pd.read_csv('./data/adjust_i-80.csv', usecols=['New_ID', 'Local_X', 'Local_Y'])

Ex, Ey, Er = 40, 400, 50
Lag, Horizon = 400, 200

def check_trajectory(group):
    trajectory = group[['Local_X', 'Local_Y']].values
    distances = np.linalg.norm(trajectory - np.array([Ex, Ey]), axis=1)
    inside_circle_indices = np.where(distances <= Er)[0]
    if distances[0] > Er and distances[-1] > Er and inside_circle_indices.size:
        assert np.all(np.diff(inside_circle_indices) <= 3)
        enter_index = inside_circle_indices[0]
        leave_index = inside_circle_indices[-1]
        return enter_index >= Lag + Horizon + 250 and enter_index + Horizon >= leave_index - 9
    return False

df = df.groupby('New_ID').filter(check_trajectory)
print(f"There are {df['New_ID'].nunique()} trajectories")
df.to_csv('./data/filter_i-80.csv', index=False)