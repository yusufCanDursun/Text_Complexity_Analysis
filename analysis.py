import textstat
from collections import Counter
import numpy as np
from text_processing import turkce_cumle_tokenize, turkce_kelime_tokenize
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def safe_flesch_score(text):
    """Hata yonetimli Flesch skoru hesaplama"""
    try:
        return textstat.flesch_reading_ease(text)
    except Exception as e:
        logger.warning(f"Flesch skoru hesaplanamadi: {str(e)}")
        return 0.0

def calculate_difficulty(flesch, cumle_uzunluk, kelime_sayisi, tekrar):
    """Vektorel zorluk puani hesaplama"""
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    scores = np.array([
        0.8 if flesch < 40 else 0.5 if flesch < 60 else 0.2,
        0.8 if cumle_uzunluk >= 20 else 0.5 if cumle_uzunluk >= 10 else 0.2,
        0.8 if kelime_sayisi >= 500 else 0.5 if kelime_sayisi >= 100 else 0.2,
        0.8 if tekrar > 10 else 0.5 if tekrar > 5 else 0.2
    ])
    return float(np.dot(weights, scores))

def metin_analiz_et(metin):
    """Spark uyumlu metin analiz fonksiyonu"""
    try:
        if not metin or not isinstance(metin, str):
            return default_analiz_sonucu()
        
        cumleler = turkce_cumle_tokenize(metin)
        kelimeler = turkce_kelime_tokenize(metin)
        
        kelime_sayisi = len(kelimeler)
        cumle_sayisi = max(len(cumleler), 1)
        ort_cumle_uzunlugu = kelime_sayisi / cumle_sayisi
        flesch_skoru = safe_flesch_score(metin)
        tekrar_eden = sum(1 for c in Counter(kelimeler).values() if c > 1)
        
        zorluk = calculate_difficulty(
            flesch_skoru, ort_cumle_uzunlugu, 
            kelime_sayisi, tekrar_eden
        )
        
        zorluk_seviyesi = "Zor" if zorluk >= 0.6 else "Orta" if zorluk >= 0.3 else "Kolay"
        
        return (
            zorluk_seviyesi,
            int(kelime_sayisi),
            int(cumle_sayisi),
            float(round(ort_cumle_uzunlugu, 2)),
            float(round(flesch_skoru, 2)),
            int(tekrar_eden)
        )
        
    except Exception as e:
        logger.error(f"Analiz hatasi: {str(e)}")
        return default_analiz_sonucu()

def default_analiz_sonucu():
    """Hata durumu icin varsayilan degerler"""
    return ("Hata", 0, 0, 0.0, 0.0, 0)