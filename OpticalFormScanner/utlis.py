import cv2
import numpy as np


    # Birden fazla görüntüyü tek bir pencerede yatay ve dikey olarak birleştirir
    # Parametreler:
    # resimDizisi: Birleştirilecek görüntülerin 2D dizisi
    # olcek: Görüntülerin yeniden boyutlandırma ölçeği
    # etiketler: Her görüntü için gösterilecek metin etiketleri
def goruntuleriYiginla(goruntuDizisi, olcek, etiketler=[]):
    # Giriş parametreleri:
    # goruntuDizisi: İşlenecek görüntülerin 2D dizisi
    # olcek: Görüntülerin yeniden boyutlandırma ölçeği
    # etiketler: Her görüntü için gösterilecek metin etiketleri (opsiyonel)

    # 1. Temel değişkenlerin hazırlanması
    # Dizi boyutlarını ve görüntü özelliklerini belirler
    satirlar = len(goruntuDizisi)  # Dizi satır sayısı
    sutunlar = len(goruntuDizisi[0])  # Dizi sütun sayısı
    satirlarMevcut = isinstance(goruntuDizisi[0], list)  # 2D dizi kontrolü
    genislik = goruntuDizisi[0][0].shape[1]  # İlk görüntünün genişliği
    yukseklik = goruntuDizisi[0][0].shape[0]  # İlk görüntünün yüksekliği
    
    # 2. 2D dizi durumunda işlemler
    # Eğer görüntü dizisi 2 boyutlu ise:
    # - Her görüntüyü yeniden boyutlandırır
    # - Gri tonlamalı görüntüleri BGR'ye dönüştürür
    # - Görüntüleri yatay ve dikey olarak birleştirir
    if satirlarMevcut:
        for x in range(0, satirlar):
            for y in range(0, sutunlar):
                goruntuDizisi[x][y] = cv2.resize(goruntuDizisi[x][y], (0, 0), None, olcek, olcek)
                if len(goruntuDizisi[x][y].shape) == 2: 
                    goruntuDizisi[x][y] = cv2.cvtColor(goruntuDizisi[x][y], cv2.COLOR_GRAY2BGR)
        #altta yapılan işlem görüntüleri yatay ve dikey olarak birleştirir
        bosGoruntu = np.zeros((yukseklik, genislik, 3), np.uint8)
        yatay = [bosGoruntu]*satirlar
        yatay_birlesmis = [bosGoruntu]*satirlar
        for x in range(0, satirlar):
            yatay[x] = np.hstack(goruntuDizisi[x])  #hstack yatay birleştirme işlemi
            yatay_birlesmis[x] = np.concatenate(goruntuDizisi[x])   #concatenate dizi birleştirme işlemi
        dikey = np.vstack(yatay)  #vstack dikey birleştirme işlemi
        dikey_birlesmis = np.concatenate(yatay)  #concatenate dizi birleştirme işlemi
    else:
        for x in range(0, satirlar):
            #altta yapılan işlem görüntüleri yeniden boyutlandırır
            goruntuDizisi[x] = cv2.resize(goruntuDizisi[x], (0, 0), None, olcek, olcek)
            if len(goruntuDizisi[x].shape) == 2: 
                goruntuDizisi[x] = cv2.cvtColor(goruntuDizisi[x], cv2.COLOR_GRAY2BGR)
        yatay = np.hstack(goruntuDizisi)  #hstack yatay birleştirme işlemi
        yatay_birlesmis = np.concatenate(goruntuDizisi)  #concatenate dizi birleştirme işlemi
        dikey = yatay

    if len(etiketler) != 0:
        #altta yapılan işlem görüntülerin üzerine etiketleri yazdırır
        goruntuGenisligi = int(dikey.shape[1] / sutunlar)
        goruntuYuksekligi = int(dikey.shape[0] / satirlar)
        for d in range(0, satirlar):
            for c in range(0, sutunlar):
                cv2.rectangle(dikey,
                            (c*goruntuGenisligi, goruntuYuksekligi*d),
                            (c*goruntuGenisligi+len(etiketler[d][c])*13+27, 30+goruntuYuksekligi*d),
                            (255,255,255), cv2.FILLED)
                cv2.putText(dikey, etiketler[d][c],
                           (goruntuGenisligi*c+10, goruntuYuksekligi*d+20),
                           cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
    return dikey



 # Dörtgenin köşe noktalarını yeniden düzenler
    # Parametreler:
    # noktalar: Düzenlenecek köşe noktaları
    # İşlem: Sol üst, sağ üst, sol alt ve sağ alt şeklinde sıralar
def noktalariYenidenDuzenle(noktalar):
    # Fazla parantezi kaldır ve 4x2'lik matrise dönüştür
    noktalar = noktalar.reshape((4, 2))
    print(noktalar)
    
    # Düzenlenmiş noktalar için yeni matris oluştur
    yeniNoktalar = np.zeros((4, 1, 2), np.int32)
    
    # Her satırın toplamını hesapla
    toplam = noktalar.sum(1)
    print(toplam)
    print(np.argmax(toplam))
    
    # Sol üst köşe (en küçük toplam)
    yeniNoktalar[0] = noktalar[np.argmin(toplam)]  #[0,0]
    # Sağ alt köşe (en büyük toplam)
    yeniNoktalar[3] = noktalar[np.argmax(toplam)]  #[w,h]
    
    # Noktalar arasındaki farkı hesapla
    fark = np.diff(noktalar, axis=1)
    # Sağ üst köşe (en küçük fark)
    yeniNoktalar[1] = noktalar[np.argmin(fark)]    #[w,0]
    # Sol alt köşe (en büyük fark)
    yeniNoktalar[2] = noktalar[np.argmax(fark)]    #[h,0]

    return yeniNoktalar



 # Dikdörtgen şeklindeki konturları filtreler
    # Parametreler:
    # konturlar: İşlenecek kontur listesi
    # İşlem: Alan > 50 olan ve 4 köşeli konturları seçer
def dortgenKonturBul(konturlar):
    dortgenKonturlar = []
    for kontur in konturlar:
        alan = cv2.contourArea(kontur)
        if alan > 50:
            cevre = cv2.arcLength(kontur, True)
            yaklasikSekil = cv2.approxPolyDP(kontur, 0.02 * cevre, True)
            if len(yaklasikSekil) == 4:
                dortgenKonturlar.append(kontur)
    dortgenKonturlar = sorted(dortgenKonturlar, key=cv2.contourArea, reverse=True)
    #print(len(dortgenKonturlar))
    return dortgenKonturlar

 # Konturun köşe noktalarını bulur
    # Parametreler:
    # kontur: İşlenecek kontur
    # İşlem: Konturun çevresini hesaplayıp köşe noktalarını yaklaşık olarak bulur
def koseNoktalariAl(kontur):
    cevre = cv2.arcLength(kontur, True) # KONTURUN ÇEVRESİ
    yaklasik = cv2.approxPolyDP(kontur, 0.02 * cevre, True) # KÖŞE NOKTALARINI BULMAK İÇİN POLİGONU YAKLAŞTIR
    return yaklasik


 # Görüntüyü 5x5 kutulara böler
    # Parametreler:
    # goruntu: Bölünecek görüntü
    # İşlem: Görüntüyü önce yatay sonra dikey olarak 5'e böler
def kutuBol(goruntu):
    satirlar = np.vsplit(goruntu,5)
    kutular=[]
    for satir in satirlar:
        sutunlar = np.hsplit(satir,5)
        for kutu in sutunlar:
            kutular.append(kutu)
    return kutular



 # Görüntü üzerine ızgara çizer
    # Parametreler:
    # goruntu: Izgara çizilecek görüntü
    # sorular: Yatay bölme sayısı
    # secenekler: Dikey bölme sayısı
def GridCiz(goruntu, sorular=5, secenekler=5):
    bolumGenislik = int(goruntu.shape[1]/sorular)
    bolumYukseklik = int(goruntu.shape[0]/secenekler)
    
    # 9 yerine max(sorular, secenekler) kullanarak dinamik çizgi sayısı
    for i in range(0, max(sorular, secenekler) + 1):
        nokta1 = (0, bolumYukseklik * i)
        nokta2 = (goruntu.shape[1], bolumYukseklik * i)
        nokta3 = (bolumGenislik * i, 0)
        nokta4 = (bolumGenislik * i, goruntu.shape[0])
        cv2.line(goruntu, nokta1, nokta2, (255, 255, 0), 2)
        cv2.line(goruntu, nokta3, nokta4, (255, 255, 0), 2)

    return goruntu


 # Cevapları görüntü üzerinde gösterir
    # Parametreler:
    # goruntu: İşaretleme yapılacak görüntü
    # cevapIndeksleri: Kullanıcının cevapları
    # notlandirma: Doğru/yanlış bilgisi
    # dogruCevaplar: Gerçek cevap anahtarı
    # sorular: Soru sayısı
    # secenekler: Seçenek sayısı
def cevaplariGoster(goruntu, cevapIndeksleri, notlandirma, dogruCevaplar, sorular=5, secenekler=5):
    secW = int(goruntu.shape[1] / sorular)
    secH = int(goruntu.shape[0] / secenekler)

    for x in range(0, sorular):
        benimCevabim = cevapIndeksleri[x]
        cX = (benimCevabim * secW) + secW // 2
        cY = (x * secH) + secH // 2
        if notlandirma[x] == 1:
            benimRengim = (0, 255, 0)
            cv2.circle(goruntu, (cX, cY), 50, benimRengim, cv2.FILLED)
        else:
            benimRengim = (0, 0, 255)
            cv2.circle(goruntu, (cX, cY), 50, benimRengim, cv2.FILLED)

            # DOĞRU CEVAP
            benimRengim = (0, 255, 0)
            dogruCevap = dogruCevaplar[x]
            cv2.circle(goruntu, ((dogruCevap * secW) + secW // 2, (x * secH) + secH // 2),
                       20, benimRengim, cv2.FILLED)

