import os
import pandas as pd
from analysis import metin_analiz_et

metin_klasoru = "texts"
metin_listesi = []

print("Metin dosyalari isleniyor...")
for dosya_adi in os.listdir(metin_klasoru):
    if dosya_adi.endswith(".txt"):
        try:
            if "_" in dosya_adi:
                zorluk = dosya_adi.split("_")[0].lower()
            else:
                zorluk = "bilinmiyor"
                
            dosya_yolu = os.path.join(metin_klasoru, dosya_adi)
            with open(dosya_yolu, "r", encoding="utf-8") as f:
                icerik = f.read()
                metin_listesi.append({"dosya_adi": dosya_adi, "zorluk": zorluk, "icerik": icerik})
                print(f"'{dosya_adi}' dosyasi basariyla yuklendi.")
        except Exception as e:
            print(f"Hata: {dosya_adi} dosyasi islenirken bir sorun olustu: {e}")

sonuclar = []
print("\nMetin analizleri yapiliyor...")
for metin in metin_listesi:
    try:
        sonuc = metin_analiz_et(metin["icerik"])
        sonuclar.append({"dosya_adi": metin["dosya_adi"], **sonuc})
        print(f"'{metin['dosya_adi']}' dosyasi analiz edildi.")
    except Exception as e:
        print(f"Hata: '{metin['dosya_adi']}' dosyasi analiz edilirken sorun olustu: {e}")

sonuc_df = pd.DataFrame(sonuclar)

print("\nANALIZ SONUCLARI:")
if len(sonuc_df) > 0:
    print(sonuc_df.to_string(index=False))
else:
    print("Hic analiz sonucu bulunamadi!")
