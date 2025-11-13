# Bug Detection System  
*Repository:* **bugDetectionsystem1**

## 1. Proje Hakkında  
Bu proje, kalite güvence sürecini desteklemek üzere geliştirilmiş bir “bug/defect detection” sistemidir. Özellikle yazılım, donanım ya da görsel veri üzerinde hataları (bugları) otomatik olarak tespit etmeyi hedeflemektedir.

### Amaç  
- Hataların erken aşamada keşfedilmesi ve raporlanması  
- Manuel test sürecini hızlandırmak / desteklemek  
- Veri odaklı hata tespiti için yapay zeka ve makine öğrenmesi yaklaşımlarının kullanılması  
- Geliştirici ve test ekipleri için hataları yönetilebilir hâle getirmek

### Kapsam  
- Görsel/imalat/veri seti üzerinden hataların tespiti (örneğin: görsel hata, imalat kusuru, yazılım bugı)  
- Python tabanlı bir prototip (örneğin Jupyter Notebook kullanımı)  
- Web arayüzü ya da basit API üzerinden çalışabilme imkânı

## 2. Özellikler  
- Jupyter Notebook ile veri ön işleme, model eğitimi ve değerlendirme adımları  
- `app.py` dosyası ile basit bir web uygulaması ya da API başlangıcı  
- Kolayca genişletilebilir: farklı veri setleri ya da yeni algor­itmalar eklenebilir  
- Modüler yapı: veri yükleme, ön işleme, model, sonuç analizi gibi bileşenler ayrılmış durumda

## 3. Teknolojiler  
- Python (ana dil)  
- Jupyter Notebook (analiz, prototip için)  
- Web uygulaması için (örneğin) Flask veya benzeri bir mikro framework (`app.py` içinde)  
- Makine öğrenmesi/makine görmesi kütüphaneleri: (örneğin) pandas, numpy, scikit-learn, tensorflow/keras, OpenCV  
- Versiyon kontrolü için Git + GitHub

## 4. Kurulum & Çalıştırma  
Aşağıdaki adımlarla projeyi yerel makinenizde çalıştırabilirsiniz:

```bash
# Reposu klonlayın
git clone https://github.com/erdemyildiz35/bugDetectionsystem1.git
cd bugDetectionsystem1

# Sanal ortam oluşturun (tercihen)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Gerekli paketleri yükleyin
pip install -r requirements.txt

# Notebook’u açın
jupyter notebook

# Web uygulamasını çalıştırın (örneğin)
python app.py
