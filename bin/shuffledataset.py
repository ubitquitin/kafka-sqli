import pandas as pd

datafile = 'data/sql_data.csv'
file_suffix = datafile.split('/')[-1]
df = pd.read_csv(datafile, header=None)
df = df.sample(frac=1)

df.to_csv(f'data/shuffled_{file_suffix}', index=False, header=False)
