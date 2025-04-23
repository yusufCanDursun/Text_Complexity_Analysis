import os

os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from analysis import metin_analiz_et





spark = SparkSession.builder \
    .appName("Turkce Metin Analizi") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.python.worker.memory", "2g") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate()

analiz_schema = StructType([
    StructField("zorluk", StringType()),
    StructField("kelime_sayisi", IntegerType()),
    StructField("cumle_sayisi", IntegerType()),
    StructField("ortalama_cumle_uzunlugu", DoubleType()),
    StructField("flesch_skoru", DoubleType()),
    StructField("tekrar_eden_kelime_sayisi", IntegerType())
])

metin_analiz_udf = udf(metin_analiz_et, analiz_schema)

def dosya_yukle(klasor_yolu):
    """Dosyalari Spark DataFrame'e yukleme"""
    from pyspark.sql import Row
    
    veriler = []
    for dosya in os.listdir(klasor_yolu):
        if dosya.endswith(".txt"):
            try:
                with open(os.path.join(klasor_yolu, dosya), "r", encoding="utf-8") as f:
                    icerik = f.read()
                    zorluk = dosya.split("_")[0].lower() if "_" in dosya else "belirsiz"
                    veriler.append(Row(
                        dosya_adi=dosya,
                        zorluk_etiket=zorluk,
                        icerik=icerik
                    ))
            except Exception as e:
                print(f"Hata: {dosya} yuklenemedi - {str(e)}")
    
    return spark.createDataFrame(veriler)

def main():
    df = dosya_yukle("texts")
    
    analiz_df = df.withColumn("analiz", metin_analiz_udf(col("icerik"))) \
        .select(
            col("dosya_adi"),
            col("zorluk_etiket"),
            col("analiz.zorluk").alias("tahmini_zorluk"),
            col("analiz.kelime_sayisi"),
            col("analiz.cumle_sayisi"),
            col("analiz.ortalama_cumle_uzunlugu"),
            col("analiz.flesch_skoru"),
            col("analiz.tekrar_eden_kelime_sayisi")
        )
    
    analiz_df.show(truncate=False)
    
    analiz_df.write.mode("overwrite").csv("sonuclar", header=True)
    
    spark.stop()

if __name__ == "__main__":
    main()