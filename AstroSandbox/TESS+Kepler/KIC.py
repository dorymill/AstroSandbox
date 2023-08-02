# Throw KIC in front of every kepid
import pandas as pd

inlist = input('\nEnter inlist | ')
df =  pd.read_csv(inlist)
df['kic'] = ['KIC '+ str(s) for s in df['kic']]
df.to_csv(inlist, index=False)