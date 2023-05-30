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
    rfm['RFM_Score'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str)

    #RF skoru oluşturma
    rfm['RF_Score'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

    # RFM segmentlerini tanımlama
    segment_mapping = {
        r'[5][5]': 'Champions',
        r'[3-4-5][4-5]': 'Loyal Customers',
        r'[4-5][2-3]': 'Promising',
        r'[5][1]': 'New Customers',
        r'[5][0-1]': 'Abandoned Checkouts',
        r'[4][1]': 'Warm Leads',
        r'[3][1]': 'Cold Leads',
        r'[2-3][2-3]': 'Need Attention',
        r'[1-2][5]': 'Shouldn^t Lose',
        r'[1-2][3-4]': 'Sleepers',
        r'[1-2][1-2]': 'Lose',
    }
    # RF_Score değişkinene göre regex yapısına uygun segmetlerin oluluşturulması
    rfm['Segment'] = rfm['RF_Score'].replace(segment_mapping,regex=True)

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

# hedef segmentleri seçmekte tip hatası sonu verirse skorların kategorik olmasıdır.
"""
    ----bunu çalıştırarak sorun ortadan kalkar
    rfm["recency_score"] = rfm["recency_score"].astype("float64")
    rfm["frequency_score"] = rfm["frequency_score"].astype("float64")
    rfm["monetary_score"] = rfm["monetary_score"].astype("float64")
"""

# hedef müşteri          -segment  -içinde    [           fm_skoru 4'ten büyük olanlar         ]      ve     [ r_skoru 3'ten büyük ] olan [ müşteri numaraları ]
target_segment = rfm[rfm["Segment"].isin(["Champions","Loyal Customers"])]["customer_id"]

#    -müşteriler    -içinden (target_segment) değişkenine   ve        kadın müşterilerine uygun [müşteri numaralarını] al
target_segment = df_copy[df_copy["master_id"].isin(target_segment) & df_copy["interested_in_categories_12"].str.contains("KADIN")]["master_id"]

target_segment.to_csv("yeni_ürün_hedef_kitlesi.csv")




target_segment = rfm[rfm["Segment"].isin(["New Customers","Shouldn^t Lose", "Sleepers"])]["customer_id"]
target_segment = df_copy[df_copy["master_id"].isin(target_segment) & ((df_copy["interested_in_categories_12"].str.contains("ERKEK")) | (df_copy["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
target_segment.to_csv("yeni_ürün_hedef_kitlesi_ekek-cocuk.csv")