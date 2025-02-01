# Zatürre (Pnömoni) Tespiti Projesi

Bu proje, göğüs röntgeni görüntülerini kullanarak bireylerde **zatürre (pnömoni) veya normal** durumu tespit etmek için derin öğrenme yöntemlerini uygulamaktadır.
Proje, veri ön işleme, veri artırma (augmentation), model eğitimi, performans değerlendirme ve sonuç görselleştirme aşamalarını kapsamaktadır.



## Veri Seti

Bu projede kullanılan veri seti, **Guangzhou Kadın ve Çocuk Sağlığı Merkezi**'nden elde edilen **göğüs röntgeni (X-ray) görüntülerini** içermektedir. Veri seti üç ana klasöre ayrılmıştır:

- **train/**: Modelin eğitildiği görüntüler
- **test/**: Modelin test edildiği görüntüler
- **val/**: Model doğrulama verileri

### Veri Seti Özellikleri:
- **Toplam Görüntü Sayısı:** 5.863
- **Görüntü Boyutu:** 150x150 piksel (gri tonlamalı)
- **Sınıflar:**
  - **PNEUMONIA:** Pnömoni teşhisi konmuş görüntüler
  - **NORMAL:** Sağlıklı bireylerin göğüs röntgenleri

Tüm röntgen görüntüleri, kalite kontrolünden geçirilmiş ve tanılar iki uzman doktor tarafından doğrulanmıştır.



## Proje Aşamaları

### 1. Veri Yükleme ve Ön İşleme
- Görüntüler gri tonlamaya dönüştürülerek boyutları **150x150** olarak yeniden boyutlandırılmıştır.
- Görüntüler **0-1 aralığında normalize edilmiştir**.
- Eksik veya okunamayan görüntüler hata yönetimi ile filtrelenmiştir.
- Veri artırma teknikleri uygulanmıştır:
  - **Dönme (rotation)**
  - **Yakınlaştırma (zoom)**
  - **Yatay/Dikey Çevirme (flip)**
  - **Kaydırma (shift)**

### 2. Derin Öğrenme Modeli
Model, **Convolutional Neural Network (CNN)** mimarisi kullanılarak oluşturulmuştur.

#### Model Mimarisi:
- **Özellik Çıkarım Bloğu:**
  - 3 adet **Conv2D** katmanı (farklı filtre boyutlarıyla)
  - **Batch Normalization**, **MaxPooling2D**, ve **Dropout** katmanları
- **Sınıflandırma Bloğu:**
  - **Flatten** katmanı
  - **Dense (Tam Bağlantılı)** katmanlar (128 nöron ve sigmoid aktivasyon)

#### Model Derleyici:
- **Optimizasyon:** RMSprop
- **Kayıp Fonksiyonu:** Binary Crossentropy
- **Değerlendirme Metrikleri:** Doğruluk (Accuracy)

### 3. Model Eğitimi
- Model **15 epoch** boyunca eğitilmiştir.
- **Batch size:** 8
- **Öğrenme Oranı Azaltma:** Validation doğruluğuna göre dinamik olarak güncellenmiştir.



## Model Performansı ve Değerlendirme

### 1. Eğitim ve Doğrulama Sonuçları
- **Model Doğruluğu:** %83.97
- **Model Kayıp (Loss):** 0.72

### 2. Eğitim Süreci (Epoch Bazlı Sonuçlar)
| Epoch | Eğitim Doğruluğu (%) | Doğrulama Doğruluğu (%) | Eğitim Kaybı | Doğrulama Kaybı |
|-------|----------------------|-------------------------|--------------|-----------------|
| 1     | 73.48                | 62.66                   | 1.23         | 11.31           |
| 5     | 91.12                | 60.42                   | 0.24         | 0.63            |
| 10    | 92.64                | 73.56                   | 0.22         | 0.52            |
| 15    | 94.18                | 71.79                   | 0.18         | 1.44            |

### 3. Performans Grafikleri
- **Doğruluk (Accuracy) Grafiği:** Eğitim doğruluğu artarken, doğrulama doğruluğunda dalgalanmalar gözlemlenmiştir.
- **Kayıp (Loss) Grafiği:** Eğitim kaybı azalırken, doğrulama kaybında düzensiz artışlar gözlenmiştir.



## Sonuç

Bu proje, göğüs röntgeni görüntülerini kullanarak pnömoni tespiti için başarılı bir derin öğrenme modeli geliştirmiştir. Model, yüksek eğitim doğruluğu gösterse de doğrulama sonuçlarında bazı dalgalanmalar gözlemlenmiştir. Bu durum, modelin **aşırı öğrenme (overfitting)** yapabileceğini göstermektedir. Gelecekte:

- **Veri seti genişletilebilir.**
- **Farklı derin öğrenme mimarileri (ResNet, VGG gibi)** denenebilir.
- **Düzenlileştirme (regularization) teknikleri** uygulanarak modelin genelleme yeteneği artırılabilir.
