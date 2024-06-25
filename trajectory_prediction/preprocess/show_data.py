import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./data/adjust_i-80.csv', usecols=['Local_X', 'Local_Y', 'New_ID'])
df = df.groupby('New_ID').apply(lambda x: x.rolling(10).mean().iloc[9::10]).reset_index(drop=True)

print(df.size)


plt.figure(figsize=(6, 6))
# 假设你的DataFrame名为df，且包含列x和y
plt.scatter(df['Local_X'], df['Local_Y'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of x and y')

x = 40
y = 400
r = 50
circle = plt.Circle((x, y), r, fill=False)
plt.gca().add_artist(circle)

plt.xlim(-100, 400)
plt.ylim(0, 500)

plt.show()