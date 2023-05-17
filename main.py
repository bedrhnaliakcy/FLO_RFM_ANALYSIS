import pandas as pd
from matplotlib import pyplot as plt

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


df_copy["total_shopping"] = df_copy["order_num_total_ever_online"] + df_copy["order_num_total_ever_offline"]
df_copy["total_price"] = df_copy["customer_value_total_ever_online"] + df_copy["customer_value_total_ever_offline"]

df_copy["first_order_date"] = pd.to_datetime(df_copy["first_order_date"], format='%Y-%m-%d')
df_copy["last_order_date"] = pd.to_datetime(df_copy["last_order_date"], format='%Y-%m-%d')
df_copy["last_order_date_online"] = pd.to_datetime(df_copy["last_order_date_online"], format='%Y-%m-%d')
df_copy["last_order_date_offline"] = pd.to_datetime(df_copy["last_order_date_offline"], format='%Y-%m-%d')


plt.hist(df_copy["last_order_channel"])

plt.xlabel("toplam alışver")
plt.ylabel("toplam tutar")
plt.scatter(df_copy["total_shopping"], df_copy["total_price"])

df_copy.groupby(by="order_channel").agg(ToplamMusteri=("master_id", "count"),
                                        ToplamAlınanUrun=(["order_num_total_ever_online","order_num_total_ever_offline"],"sum"),
                                        ToplamHarcama=(["customer_value_total_ever_offline","customer_value_total_ever_offline"], "sum"))