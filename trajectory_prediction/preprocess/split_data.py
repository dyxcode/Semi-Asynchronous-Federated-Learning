from pathlib import Path
import pandas as pd

data_dir = Path('./data')
origin_dataset = 'Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_20240108.csv'
data = pd.read_csv(data_dir / origin_dataset)

groups = data.groupby('Location')
for name, group in groups:
    group.to_csv(data_dir / f'{name}.csv', index=False)