import textstat
from collections import Counter
from text_processing import turkce_cumle_tokenize, turkce_kelime_tokenize

def metin_analiz_et(metin):
    cumleler = turkce_cumle_tokenize(metin)
    kelimeler = turkce_kelime_tokenize(metin)
    
    kelime_sayisi = len(kelimeler)
    cumle_sayisi = len(cumleler)
    ort_cumle_uzunlugu = kelime_sayisi / cumle_sayisi if cumle_sayisi > 0 else 0
    
    try:
        flesch_skoru = textstat.flesch_reading_ease(metin)
    except:
        flesch_skoru = 0
        
    kelime_sayisi_dict = Counter(kelimeler)
    tekrar_eden_sayisi = sum(1 for count in kelime_sayisi_dict.values() if count > 1)
    
    if flesch_skoru > 60:
        flesch_puan = 0.2
    elif flesch_skoru > 40:
        flesch_puan = 0.5
    else:
        flesch_puan = 0.8
    
    if ort_cumle_uzunlugu < 10:
        cumle_uzunlugu_puan = 0.2
    elif ort_cumle_uzunlugu < 20:
        cumle_uzunlugu_puan = 0.5
    else:
        cumle_uzunlugu_puan = 0.8
    
    if kelime_sayisi < 100:
        kelime_puan = 0.2
    elif kelime_sayisi < 500:
        kelime_puan = 0.5
    else:
        kelime_puan = 0.8
        
    if tekrar_eden_sayisi > 10:
        tekrar_puan = 0.8  
    elif tekrar_eden_sayisi > 5:
        tekrar_puan = 0.5  
    else:
        tekrar_puan = 0.2  

    zorluk = (0.4 * flesch_puan) + (0.3 * cumle_uzunlugu_puan) + (0.2 * kelime_puan) + (0.1 * tekrar_puan)
    
    if zorluk < 0.3:
        zorluk_seviyesi = "Kolay"
    elif zorluk < 0.6:
        zorluk_seviyesi = "Orta"
    else:
        zorluk_seviyesi = "Zor"
    
    return {
        "zorluk": zorluk_seviyesi,
        "kelime_sayisi": kelime_sayisi,
        "cumle_sayisi": cumle_sayisi,
        "ortalama_cumle_uzunlugu": ort_cumle_uzunlugu,
        "flesch_skoru": flesch_skoru,
        "tekrar_eden_kelime_sayisi": tekrar_eden_sayisi
    }
