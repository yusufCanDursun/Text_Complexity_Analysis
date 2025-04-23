import nltk #dogal dil isleme icin kullanilan populer bir python kutuphanesi
import numpy as np #sayisal islemler ve diziler icin kullanilir
from nltk.tokenize import TreebankWordTokenizer #metni, noktalama isaretleri ve benzeri kurallara gore kelimelere (token) ayirir.
from nltk.corpus import stopwords #the,is,and gibi metin analizinde anlam tasimayan kelimeleri barindirir.
from nltk.stem.porter import PorterStemmer #kelimeleri koklerine indirger.
from nltk.stem import WordNetLemmatizer #gramer kurallarina gore kelimeyi kok haline getirir (fiil,isim,sifat olarak)
import string

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CSV Okuma") \
    .getOrCreate()

df = spark.read.csv("train.csv", header=True, inferSchema=True)
df.show()


def normalize_token(x):
    tokenizer = TreebankWordTokenizer()
    lower = x.text.lower()
    text_token = tokenizer.tokenize(lower)
    count_char = len(lower)
    return x+(count_char,text_token,)

rdd_normalize_token = df.rdd.map(lambda x: normalize_token(x))
rdd_normalize_token.first()

def remove_stop_word_and_changer_number(x, seuil, stop_words, list_punct) :
    tab_result = []
    count_punct = 0
    for elt in x[4]:
        if elt in list_punct:
            count_punct += 1
        try :
            number = float(elt)
            tab_result.append("number")
        except:
            if(len(elt)>seuil) &(elt not in stop_words):
                tab_result.append(elt)
    return x[:4]+(count_punct, tab_result,)

seuil = 1
stop_words = set(stopwords.words('turkish'))
list_punct = list(string.punctuation)
rdd_stop_word = rdd_normalize_token.map(lambda x : remove_stop_word_and_changer_number(x,seuil,stop_words,list_punct))
rdd_stop_word.first()