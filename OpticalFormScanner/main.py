import cv2
import numpy as np
import utlis

#
cameraFeed = True
pathImage = "A.png"
cap = cv2.VideoCapture(0)
cap.set(10,160)
imgYukseklik = 700
imgGenislik  = 700
soruSayisi=5
secenekSayisi=5
dogruCevaplar = [1,2,0,2,4]



count=0

while True:

    if cameraFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
    img = cv2.resize(img, (imgGenislik, imgYukseklik)) #resize image. görüntüyü belirtilen genişlik ve yükseklik boyutuna getirir.
    imgFinal = img.copy()
    bosImage = np.zeros((imgYukseklik,imgGenislik, 3), np.uint8) # boş görüntü oluşturur. 3= RGB temsil eder.
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # renkli görüntüyü siyah beyaz görüntüye dönüştürür.
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # Gaussian blur uygular. Görüntüdeki gürültüyü azaltır.
    imgCanny = cv2.Canny(imgBlur,10,70) # Kenar tespiti yapar.

    try:
        ## Tüm konturları bul.
        imgContours = img.copy() # DISPLAY İÇİN GÖRÜNTÜYÜ KOPYALA
        imgBigContour = img.copy() # DISPLAY İÇİN GÖRÜNTÜYÜ KOPYALA
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # imgCanny üzerinde (kenar tespiti yapılmış görüntü) konturları bulur. 
        # cv2.RETR_EXTERNAL: En dıştaki konturları bulur. # cv2.CHAIN_APPROX_NONE: Tüm kontur noktalarını korur.
        #contours: konturların listesi # hierarchy: konturların hiyerarşisi, dizi.
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # TESPİT EDİLEN TÜM KONTURLARI ÇİZ
        dortgenKonturlar = utlis.dortgenKonturBul(contours)
        enBuyukDortgenNoktalari = utlis.koseNoktalariAl(dortgenKonturlar[0]) # büyük dörtgenin köşe noktalarını al
        ikinciBuyukDortgenNoktalari = utlis.koseNoktalariAl(dortgenKonturlar[1]) # ikinci büyük dörtgenin köşe noktalarını al

        # büyük dörtgenin köşe noktaları ve ikinci büyük dörtgenin köşe noktaları bulunduysa
        if enBuyukDortgenNoktalari.size != 0 and ikinciBuyukDortgenNoktalari.size != 0:

            # büyük dörtgenin köşe noktalarını yeniden düzenleme işlemi.
            enBuyukDortgenNoktalari=utlis.noktalariYenidenDuzenle(enBuyukDortgenNoktalari) # yeniden düzenleme
            cv2.drawContours(imgBigContour, enBuyukDortgenNoktalari, -1, (0, 255, 0), 20) # büyük dörtgeni çiz
            pts1 = np.float32(enBuyukDortgenNoktalari) # Kaynak görüntüdeki 4 nokta (orijinal görüntüdeki eğik dörtgenin köşeleri)
            pts2 = np.float32([[0, 0],[imgGenislik, 0], [0, imgYukseklik],[imgGenislik, imgYukseklik]]) # hedef görüntüdeki 4 nokta (hedef görüntüdeki dörtgenin köşeleri)
            #pts1 burada eğik duran dörtgenin köşeleri, pts2 ise düzgün duran dörtgenin köşeleri.
            matrix = cv2.getPerspectiveTransform(pts1, pts2) # perspektif dönüşüm matrisi al
            imgWarpColored = cv2.warpPerspective(img, matrix, (imgGenislik, imgYukseklik)) # perspektif dönüşüm uygula
            #optik formu düz şekle getirir. imgWarpColored düzgün duran dörtgeni temsil eder.

            # ikinci büyük dörtgenin köşe noktalarını yeniden düzenleme işlemi.
            cv2.drawContours(imgBigContour, ikinciBuyukDortgenNoktalari, -1, (255, 0, 0), 20) # ikinci büyük dörtgeni çiz
            ikinciBuyukDortgenNoktalari = utlis.noktalariYenidenDuzenle(ikinciBuyukDortgenNoktalari) # köşe noktalarını yeniden düzenle
            ptsG1 = np.float32(ikinciBuyukDortgenNoktalari)  # kaynak görüntüdeki 4 nokta
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  # hedef görüntüdeki 4 nokta
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)# perspektif dönüşüm matrisi al
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150)) # perspektif dönüşüm uygula

            # EŞİKLEME(THRESHOLD) UYGULAMA
            #imgWarpColored düzgün duran dörtgeni temsil eder.
            imgPerspektifGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) # GrayScale'a dönüştür
            imgPerspektifThresh = cv2.threshold(imgPerspektifGray, 170, 255,cv2.THRESH_BINARY_INV )[1] # eşikleme uygula ve tersini al
            #threshold işlemi görüntüyü binary formata çevirir.
            #THRESH_BINARY_INV: Eşik değerinden büyük olan pikselleri 0, küçük olan pikselleri 255 yapar.
            kutular = utlis.kutuBol(imgPerspektifThresh) # her bir kutuyu 5x5'e böler.
            cv2.imshow("Split Test ", kutular[3])
            countR=0    #soru sayısı (satır sayısı)
            countC=0    #cevap sayısı (sütun sayısı)
            kutudakiSifirOlmayanPikselSayisi = np.zeros((soruSayisi,secenekSayisi)) # her bir kutudaki sıfır olmayan değerleri saklar
            # image in boxes -> 25 adet kutu taraması. 
            for image in kutular:
                #cv2.imshow(str(countR)+str(countC),image)
                totalPixels = cv2.countNonZero(image)   #siyah olmayan piksel sayısı
                kutudakiSifirOlmayanPikselSayisi[countR][countC]= totalPixels
                #her bir kutudaki sıfır olmayan piksel sayısını saklar.
                countC += 1 # sütun sayısını artırır.
                if (countC==secenekSayisi):
                    countC=0; # sütun sayısını sıfırlar.
                    countR +=1 # sütun sayısı bitince sıfırlar ve satır sayısını artırır.

            # optik form cevaplarını bulup listeye koyma işlemi.
            formCevaplar=[]  # Kullanıcı cevaplarını tutacak boş liste
            for x in range (0,soruSayisi):  # Her soru için döngü
                arr = kutudakiSifirOlmayanPikselSayisi[x]  # O sorunun tüm şıklarındaki piksel değerlerini al 
                formCevaplar.append(np.where(arr == np.amax(arr))[0][0])  # En yüksek piksel değerine sahip şıkkın indeksini bul
            #print("Kullanıcı Cevapları",myIndex)

            # Doğru cevapları bulma işlemi.
            dogruYanlis =[]  # Doğru/yanlış sonuçlarını tutacak boş liste
            for x in range(0,soruSayisi):  # Her soru için kontrol   
                if dogruCevaplar[x] == formCevaplar[x]:  # Doğru cevap (ans) ile kullanıcı cevabı (myIndex) karşılaştırması
                    dogruYanlis.append(1)      # Doğruysa 1 ekle
                else:
                    dogruYanlis.append(0)      # Yanlışsa 0 ekle
            #print("GRADING",grading)
            score = (sum(dogruYanlis)/soruSayisi)*100  # Toplam puanı hesapla
            #print("SCORE",score)

            # Cevapları görüntüleme İşlemi
            utlis.cevaplariGoster(imgWarpColored,formCevaplar,dogruYanlis,dogruCevaplar) # Taranan cevapları görüntüle
            utlis.GridCiz(imgWarpColored) # Izgara çiz
            imgRawDrawings = np.zeros_like(imgWarpColored) # YENİ BOŞ GÖRÜNTÜ OLUŞTUR
            utlis.cevaplariGoster(imgRawDrawings, formCevaplar, dogruYanlis, dogruCevaplar) # YENİ GÖRÜNTÜYE ÇİZ
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1) # TERS DÖNÜŞÜM MATRİSİ
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (imgGenislik, imgYukseklik)) # INV IMAGE WARP

            # Puanı görüntüleme İşlemi
            imgRawGrade = np.zeros_like(imgGradeDisplay,np.uint8) # YENİ BOŞ GÖRÜNTÜ OLUŞTUR
            cv2.putText(imgRawGrade,str(int(score)),(70,100)
                        ,cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),3) # YENİ GÖRÜNTÜYE PUAN EKLE
            invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1) # TERS DÖNÜŞÜM MATRİSİ
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (imgGenislik, imgYukseklik)) # TERS DÖNÜŞÜM UYGULAMA

            # Son görüntüye cevapları ve puanı ekleme işlemi
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0)

            # Son görüntüye cevapları ve puanı ekleme işlemi
            imageArray = ([img,imgGray,imgCanny,imgContours],
                          [imgBigContour,imgPerspektifThresh,imgWarpColored,imgFinal])
            cv2.imshow("Sonuc", imgFinal)
    except (cv2.error, IndexError) as e:
        print(f"Hata oluştu: {e}")
        imageArray = ([img,imgGray,imgCanny,imgContours],
                      [bosImage, bosImage, bosImage, bosImage])

    # GÖRÜNTÜLERİ GÖRÜNTÜLEME İŞLEMİ
    lables = [["Orijinal","Gri Tonlamalı","Kenarlar","Konturlar"],
              ["Buyuk Kontur","Etikleme","Perspektif","Sonuc"]]

    imgStack = utlis.goruntuleriYiginla(imageArray,0.5,lables)
    cv2.imshow("Sonuç",imgStack)


    if cv2.waitKey(1) & 0xFF == ord('q'):  # q tuşuna basıldığında çık
        break

# While döngüsünden sonra ekleyin:
cap.release()
cv2.destroyAllWindows()