#----------------------------------------------------- todo Kütüphaneler
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# ekran ayarları
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#----------------------------------------------------- todo Veri Okuma

# veri setini çekelim
df_origin = pd.read_csv("D:/programlama/python/MIUUL-PROJECTS/FLO_RFM_ANALYSIS/DataSet/flo_data_20k.csv")
#----------------------------------------------------- todo Veri Önişleme
def data_preprocessing(data):

    # veri setinin kopyasını oluşturuyoruz
    df_copy = data.copy()

    df_copy.nunique()  # boş değerler
    df_copy.dropna(inplace=True) # boş değerleri kalıcı sil


    df_copy["total_shopping"] = df_copy["order_num_total_ever_online"] + df_copy["order_num_total_ever_offline"] # toplam alışveriş
    df_copy["total_price"] = df_copy["customer_value_total_ever_online"] + df_copy["customer_value_total_ever_offline"] # toplam tutar

    # toplam harcama ve sipariş
    df_copy["total_shopping"].sort_values().head(10)
    df_copy["total_price"].sort_values().head(10)

    # tarih değişkenlerini fromat olark tarihe çevir
    df_copy["first_order_date"] = pd.to_datetime(df_copy["first_order_date"], format='%Y-%m-%d')
    df_copy["last_order_date"] = pd.to_datetime(df_copy["last_order_date"], format='%Y-%m-%d')
    df_copy["last_order_date_online"] = pd.to_datetime(df_copy["last_order_date_online"], format='%Y-%m-%d')
    df_copy["last_order_date_offline"] = pd.to_datetime(df_copy["last_order_date_offline"], format='%Y-%m-%d')

    return df_copy

df_copy = data_preprocessing(df_origin)

#----------------------------------------------------- todo Temel analiz

print("\n ilk 10 veri--------\n",df_copy.head(20),"\n") # ilk 10 veri
print("\n veri boyutu--------\n",df_copy.shape,"\n")  # veri seti boyutu
print("\n veri index sistemi--------\n",df_copy.index,"\n")  #
print("\n veri kolon isimleri--------\n",df_copy.columns,"\n")  # değişkenlerin isimleri
print("\n veri bilgilendirmesi--------\n",df_copy.info(),"\n")  # veri seti hakınnda bilgilendirme
print("\n veri seti ortalaması--------\n",df_copy.count(),"\n")  # veri seti ortalaması

print("\n veri seri istatistiksel değerler--------\n",df_copy.describe().T,"\n")  # sayısal değişkenlerin temel istatistik değerleri

#----------------------------------------------------- todo Görsel analiz

# histogram gösterimi ile "satın alma kanalının" dağılımı
def visual(df,*columns):
    for i in columns:
        plt.boxplot(df[i])
        plt.title(columns)
        plt.show()
        plt.close(fig=True)

#-----------------------------------------------------

# "satın alma kanalı" temel istatistiksel olarak incele
df_copy.groupby(by="order_channel").describe()

#----------------------------------------------------- todo RFM analizi
def rfm_analysis(data):
    # RFM değerlerini hesaplama
    snapshot_date = max(data['last_order_date']) + pd.DateOffset(1)  # Bugünden bir gün sonrası

    rfm = pd.DataFrame()  # rfm adında dataframe oluştur
    # müşteri numarası, güncellik, sıklık ve parasal değerlerini tanımlıyoruz
    rfm["customer_id"] = data["master_id"]
    rfm['Recency'] = (snapshot_date - data['last_order_date']).dt.days
    rfm['Frequency'] = data['total_shopping']
    rfm['Monetary'] = data['total_price']

    # rfm skorlarını her müşteri için kesme işlemi ile oluşturuyoruz
    rfm['recency_score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    rfm['frequency_score'] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm['monetary_score'] = pd.qcut(rfm['Monetary'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')

    # RFM skoru oluşturma
    rfm['RFM_Score'] = rfm['Recency'].astype(str) + rfm['Frequency'].astype(str) + rfm['Monetary'].astype(str)

    #RF skoru oluşturma
    rfm['RF_Score'] = rfm['Recency'].astype(str) + rfm['Frequency'].astype(str)

    # RFM segmentlerini tanımlama
    segment_mapping = {
        '1': 'Champions',
        '2': 'Loyal Customers',
        '3': 'Promising',
        '4': 'New Customers',
        '5': 'Abandoned Checkouts',
        '6': 'Warm Leads',
        '7': 'Cold Leads',
        '8': 'Need Attention',
        '9': 'Shouldn^t Lose',
        '10': 'Sleepers',
        '11': 'Lose',
    }

    rfm['Segment'] = rfm['RF_Score'].map(segment_mapping)

    return rfm
#----------------------------------------------------- todo aykırı değerleri tespiti
def remove_outliers(df, *columns, threshold=3):
    df_no_outliers = df.copy()
    for column in columns:
        z_scores = np.abs((df_no_outliers[column] - df_no_outliers[column].mean()) / df_no_outliers[column].std())
        outliers = df_no_outliers[z_scores >= threshold]
        df_no_outliers = df_no_outliers.drop(outliers.index)
    return df_no_outliers

#----------------------------------------------------- todo Aksiyon Alınması
rfm = pd.DataFrame()

rfm = rfm_analysis(df_copy)
rfm_non_outliers = remove_outliers(rfm, "Recency", "Frequency", "Monetary")

print("\n\nRFM İstatistikleri:\n", rfm.describe())
print("\n\nRFM İstatistikleri: \t\t(aykırı değerler silindi)\n", rfm_non_outliers.describe())