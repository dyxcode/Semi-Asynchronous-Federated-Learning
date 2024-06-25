import pandas as pd
import numpy as np
from pathlib import Path 

dataset_path = Path('./data/i-80.csv')
cols = ['Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time', 'Local_X', 'Local_Y']
df = pd.read_csv(dataset_path, usecols=cols)
df['New_ID'] = df.groupby(['Vehicle_ID', 'Total_Frames']).ngroup()

# format data to seconds
df['Global_Time'] = df['Global_Time'] / 100       # 0.1s

# format feet to meter
conversion_factor = 0.3048
for label in ['Local_X', 'Local_Y']:
  df[label] = df[label] * conversion_factor

df = df.sort_values(by=['New_ID', 'Global_Time'])

# filter discontinuity New_ID
is_time_continuity = df.groupby('New_ID')['Global_Time'].apply(lambda x: np.all(np.diff(x) == 1))
df = df[df['New_ID'].isin(is_time_continuity[is_time_continuity == True].index)]

# check again  
assert all(df.groupby('New_ID')['Global_Time'].apply(lambda x: np.all(np.diff(x) == 1)) == 1)
assert df.notnull().all().all()

df.to_csv(f'./data/adjust_{dataset_path.stem}.csv', index=False)

