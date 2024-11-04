import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

# Veri kümesini yükle
şarkılar = pd.read_csv('dataset/tracks_features.csv/tracks_features.csv')  # Gerekirse yolu ayarlayın
print(şarkılar.head())
print(şarkılar.shape)
print(şarkılar.info())
print(şarkılar.isnull().sum())

# Null değerleri içeren satırları kaldır
şarkılar.dropna(inplace=True)

# Kalan Boş değerlerini görselleştir
şarkılar.isnull().sum().plot.bar()
plt.title('Her Sütundaki Boş Değerlerinin Sayısı')
plt.show()

# Gerekli olmayan 'id' ve 'artist_ids' sütunlarını kaldır
şarkılar = şarkılar.drop(['id', 'artist_ids'], axis=1)

# t-SNE görselleştirmesi için verileri hazırla
özellikler_tsne = şarkılar[['danceability', 'energy', 'loudness', 'tempo', 'valence']]
model = TSNE(n_components=2, random_state=0)
tsne_veri = model.fit_transform(özellikler_tsne.head(500))

# t-SNE sonuçlarını görselleştir
plt.figure(figsize=(7, 7))
plt.scatter(tsne_veri[:, 0], tsne_veri[:, 1])
plt.title('Şarkı Özelliklerinin t-SNE Görselleştirmesi')
plt.xlabel('Bileşen 1')
plt.ylabel('Bileşen 2')
plt.show()

# Eşsiz şarkı isimlerini ve veri kümesinin şekli
print(f'Eşsiz şarkı isimleri: {şarkılar["name"].nunique()}, Toplam şekil: {şarkılar.shape}')
şarkılar.drop_duplicates(subset=['name'], keep='first', inplace=True)  # Tekrar eden şarkı isimlerini kaldır

# Çıkış yılına göre şarkı sayısını gösteren grafiği çiz
plt.figure(figsize=(10, 5))
sb.countplot(data=şarkılar, x='year')  # 'year' sütununu kullanmak için değiştirildi
plt.title('Çıkış Yılına Göre Şarkı Sayısı')
plt.xticks(rotation=45)  # X ekseni etiketlerini daha iyi görünür hale getirmek için döndür
plt.show()

# Dağılım grafikleri için float türündeki sütunları al
float_sütunlar = [col for col in şarkılar.columns if şarkılar[col].dtype == 'float']

# Float sütunları için dağılım grafikleri çiz
plt.subplots(figsize=(15, 5))
for i, col in enumerate(float_sütunlar):
    plt.subplot(2, 5, i + 1)
    sb.histplot(şarkılar[col], kde=True)
    plt.title(f'{col} Dağılımı')
plt.tight_layout()
plt.show()

# Eğer 'genres' sütunu varsa CountVectorizer'ı başlat
şarkı_vectorizer = CountVectorizer()
if 'genres' in şarkılar.columns:
    şarkı_vectorizer.fit(şarkılar['genres'])  # 'genres' sütunu varsa fit et

# Mevcut verilerle ilk 10,000 şarkıyı sakla
şarkılar = şarkılar.head(10000)

def benzerlikleri_hesapla(şarkı_ismi, veri):
    """Verilen şarkıya göre benzerlikleri hesapla."""
    # Giriş şarkısı için vektörü al
    if 'genres' in veri.columns:
        metin_array1 = şarkı_vectorizer.transform(veri[veri['name'] == şarkı_ismi]['genres']).toarray()
    else:
        metin_array1 = np.zeros((1, 1))  # Tür yoksa boş vektör

    sayısal_array1 = veri[veri['name'] == şarkı_ismi].select_dtypes(include=np.number).to_numpy()

    # Veri kümesinin her satırı için benzerliği sakla
    benzerlikler = []
    for idx, row in veri.iterrows():
        isim = row['name']

        # Mevcut şarkı için vektörü al
        if 'genres' in veri.columns:
            metin_array2 = şarkı_vectorizer.transform(veri[veri['name'] == isim]['genres']).toarray()
        else:
            metin_array2 = np.zeros((1, 1))  # Tür yoksa boş vektör

        sayısal_array2 = veri[veri['name'] == isim].select_dtypes(include=np.number).to_numpy()

        # Metin ve sayısal özellikler için benzerlikleri hesapla
        metin_benzerlik = cosine_similarity(metin_array1, metin_array2)[0][0]
        sayısal_benzerlik = cosine_similarity(sayısal_array1, sayısal_array2)[0][0]
        benzerlikler.append(metin_benzerlik + sayısal_benzerlik)

    return benzerlikler

def şarkı_tavsiye_et(şarkı_ismi, veri=şarkılar):
    """Giriş şarkısına göre benzer şarkıları öner."""
    # Şarkının veri kümesinde olup olmadığını kontrol et
    if veri[veri['name'] == şarkı_ismi].shape[0] == 0:
        print('Bu şarkı ya veri kümesinde yok ya da adı geçersiz.\nTavsiye edilebilecek bazı şarkılar:\n')
        for şarkı in veri.sample(n=5)['name'].values:
            print(şarkı)
        return

    # Belirtilen şarkı için benzerlikleri hesapla
    veri['benzerlik_faktörü'] = benzerlikleri_hesapla(şarkı_ismi, veri)

    # Benzerlik faktörüne göre sırala
    veri.sort_values(by='benzerlik_faktörü', ascending=False, inplace=True)

    # Tavsiye edilen şarkıları göster (giriş şarkısını hariç tut)
    print('Seçiminize göre tavsiye edilen şarkılar:')
    tavsiye_edilen_şarkılar = veri[['name', 'artists']][veri['name'] != şarkı_ismi].head(5)  # Giriş şarkısını hariç tut
    print(tavsiye_edilen_şarkılar)

# Şarkı önerme fonksiyonunu çağır
şarkı_tavsiye_et('Shape of You')  # Bu örnek şarkıyı değiştirebilir ve veri kümesinde bulunan başka bir şarkıyı kullanabilirsiniz
