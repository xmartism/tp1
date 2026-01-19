import pandas as pd

# Načítaj dataset bez hlavičky
df = pd.read_csv('dataset/SinTwentyWaves.csv', header=None, names=['value'])

# Pridaj date stĺpec
df.insert(0, 'date', pd.date_range('2020-01-01 00:00:00', periods=len(df), freq='15min'))

# Ulož späť
df.to_csv('dataset/SinTwentyWaves.csv', index=False)

