import os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, VectorAssembler
from pyspark.sql.functions import col, size, length, regexp_replace, split, expr, lower, trim, monotonically_increasing_id
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import IntegerType

# Ortam değişkenleri
os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-11.0.17"
os.environ["SPARK_HOME"] = "C:\\spark"
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] += os.pathsep + "C:\\spark\\bin"
os.environ["PATH"] += os.pathsep + "C:\\hadoop\\bin"
os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"

spark = SparkSession.builder \
    .appName("MetinKarmasikAnalis") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 1. Veri okuma ve temizleme
df = spark.read.text('metinler/*.txt')
df = df.withColumn("id", monotonically_increasing_id())
df = df.withColumn("Metin", trim(lower(col("value"))))
df = df.filter(col("Metin") != "")

# 2. Tokenizer ve StopWordsRemover
tokenizer = Tokenizer(inputCol="Metin", outputCol="kelimeler")
remover = StopWordsRemover(inputCol="kelimeler", outputCol="filtered")

# 3. Özellik çıkarımı fonksiyonu
def feature_engineering(df):
    df = df.withColumn("kelime_sayisi", size(col("filtered")))
    df = df.withColumn("karakter_sayisi", length(col("Metin")))
    df = df.withColumn("ortalama_kelime_uzunlugu", col("karakter_sayisi") / (col("kelime_sayisi") + 1))
    df = df.withColumn("noktalama_sayisi", length(col("Metin")) - length(regexp_replace(col("Metin"), r"[.,;:!?]", "")))
    df = df.withColumn("uzun_kelime_sayisi", expr("size(filter(filtered, x -> length(x) > 7))"))
    df = df.withColumn("cumle_sayisi", size(split(col("Metin"), r"[.!?]")))
    df = df.withColumn("farkli_kelime_sayisi", size(expr("array_distinct(filtered)")))
    df = df.withColumn("kelime_ceşitliliği", col("farkli_kelime_sayisi") / (col("kelime_sayisi") + 1))
    # Rastgele karmaşıklık seviyesi (0, 1, 2)
    df = df.withColumn("karmasik_seviyesi", (expr("rand()") * 3).cast(IntegerType()))
    return df

# 4. Özellikleri birleştir
assembler = VectorAssembler(
    inputCols=[
        "kelime_sayisi", "karakter_sayisi", "ortalama_kelime_uzunlugu",
        "noktalama_sayisi", "uzun_kelime_sayisi", "cumle_sayisi",
        "farkli_kelime_sayisi", "kelime_ceşitliliği"
    ],
    outputCol="features"
)

# 5. Model
rf = RandomForestClassifier(featuresCol="features", labelCol="karmasik_seviyesi", numTrees=50)

# 6. Pipeline
pipeline = Pipeline(stages=[tokenizer, remover])
df = pipeline.fit(df).transform(df)
df = feature_engineering(df)
df = assembler.transform(df)
model = rf.fit(df)

# 7. Tahmin ve değerlendirme
tahminler = model.transform(df)
evaluator = MulticlassClassificationEvaluator(labelCol="karmasik_seviyesi", predictionCol="prediction", metricName="f1")
f1 = evaluator.evaluate(tahminler)
print(f"F1 Skoru: {f1}")

evaluator_acc = MulticlassClassificationEvaluator(labelCol="karmasik_seviyesi", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_acc.evaluate(tahminler)
print(f"Modelin doğruluğu: {accuracy}")

tahminler.select("Metin", "karmasik_seviyesi", "prediction").show(20)
