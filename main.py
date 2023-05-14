import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv("DataSet/flo_data_20k.csv")
df_copy = df.copy()

df_copy.head(20)
df_copy.shape
df_copy.index
df_copy.columns
df_copy.info()
df_copy.count()

df_copy.describe().T

df_copy.nunique()
df_copy.dropna(inplace=True)
