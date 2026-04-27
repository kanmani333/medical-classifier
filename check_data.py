
import pandas as pd

for filename in ['train.dat', 'test.dat']:
    try:
        df = pd.read_csv(filename, sep='\t', header=None, names=['label', 'text'])
        print(f'\n{filename} Shape:', df.shape)
        print('Label counts:')
        print(df['label'].value_counts())
    except Exception as e:
        print(f'Error reading {filename}:', e)