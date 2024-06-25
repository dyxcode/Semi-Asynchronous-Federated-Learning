import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from dataset import get_trajectories

Ex, Ey, Er = 40, 400, 50
trajectories = get_trajectories()

results = []
for id, trajectory in trajectories:
    distances = np.linalg.norm(trajectory - np.array([Ex, Ey]), axis=1)
    inside_circle_indices = np.where(distances <= Er)[0]
    enter_index = inside_circle_indices[0]
    leave_index = inside_circle_indices[-1]

    last_point = trajectory[enter_index - 1]
    x = (trajectory[0:enter_index, 0] - last_point[0]).reshape((-1, 1))
    y = trajectory[0:enter_index, 1] - last_point[1]
    Ex_relative = Ex - last_point[0]
    Ey_relative = Ey - last_point[1]

    model = LinearRegression(fit_intercept=False).fit(x, y)

    # 得到直线的斜率
    slope = model.coef_

    discriminant = np.sqrt((slope**2 + 1)*Er**2 - Ey_relative**2)
    x1 = (Ex_relative + Ey_relative*slope + discriminant) / (slope**2 + 1)
    x2 = (Ex_relative + Ey_relative*slope - discriminant) / (slope**2 + 1)
    y1 = slope*x1
    y2 = slope*x2

    pos = (np.array([x1, y1]) if y1 > y2 else np.array([x2, y2])).squeeze(1)
    pos += last_point
    
    # v = np.linalg.norm(trajectory[enter_index] - trajectory[0]) / enter_index
    v = np.linalg.norm(np.diff(trajectory[enter_index-10:enter_index], axis=0), axis=1).sum() / 10
    t = np.linalg.norm(pos - trajectory[enter_index]) / v

    results.append({
        'New_ID': int(id),
        'enter_index': enter_index,
        'leave_index': leave_index,
        'predict_index': int(t) + enter_index
    })

df = pd.DataFrame(results)
df.to_csv('./data/Regression_predictions.csv', index=False)