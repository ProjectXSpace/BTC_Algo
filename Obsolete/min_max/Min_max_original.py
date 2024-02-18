import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


folder_path = os.path.dirname(os.path.realpath(__file__))

csv_path = os.path.join(folder_path, 'BTC-USD.csv')





# Example DataFrame for Bitcoin data (replace this with your data source)
df = pd.read_csv(csv_path)
df.dropna(inplace=True)

# Calculate the derivative
df['derivative'] = df['Close'].diff().fillna(0)

# Find local minima and maxima
conditions = [
    (df['derivative'] < 0) & (df['derivative'].shift(-1) > 0), # Minima condition
    (df['derivative'] > 0) & (df['derivative'].shift(-1) < 0)  # Maxima condition
]

values = [-1, 1]

# Applying the conditions and assigning values, the default is 0 for other cases
df['min_max'] = np.select(conditions, values, default=0)

#print(df.head)
#df.to_csv(os.path.join(folder_path, 'min_max.csv'))

# Plot the data
# Select the last 200 rows
df_last_200 = df.tail(3000)

plt.plot(df_last_200['Date'], df_last_200['Close'], label='Price')
plt.scatter(df_last_200['Date'][df_last_200['min_max'] == -1], df_last_200['Close'][df_last_200['min_max'] == -1], color='r', label='Minima')
plt.scatter(df_last_200['Date'][df_last_200['min_max'] == 1], df_last_200['Close'][df_last_200['min_max'] == 1], color='g', label='Maxima')
plt.legend()
plt.show()

counts = df['min_max'].value_counts()
num_minima = counts.get(-1, 0) # Number of -1 (minima)
num_maxima = counts.get(1, 0)  # Number of 1 (maxima)
total_rows = df.shape[0]
num_zeros = total_rows - (num_maxima + num_minima)

print("Number of Minima:", num_minima)
print("Number of Maxima:", num_maxima)
print("Number of 0-s:", num_zeros)



