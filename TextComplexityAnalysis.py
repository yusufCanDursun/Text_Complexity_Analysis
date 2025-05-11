import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, VectorAssembler, StopWordsRemover
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, size, length, regexp_replace, split, expr, trim
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import IntegerType

os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] += os.pathsep + "C:\\hadoop\\bin"
os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"

spark = SparkSession.builder \
    .appName("MetinKarmasikAnalis") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

metinler_df = spark.read.csv('etiketli_metin.csv', header=True, inferSchema=True)
metinler_df = metinler_df.filter(trim(col("Metin")) != "")

tokenizer = Tokenizer(inputCol="Metin", outputCol="kelimeler")
metinler_df = tokenizer.transform(metinler_df)

metinler_df = metinler_df.withColumn("kelime_sayisi", size(col("kelimeler")))
metinler_df = metinler_df.withColumn("karakter_sayisi", length(col("Metin")))
metinler_df = metinler_df.withColumn("ortalama_kelime_uzunlugu", (col("karakter_sayisi") / (col("kelime_sayisi") + 1)))              
metinler_df = metinler_df.withColumn("noktalama_sayisi", length(col("Metin")) - length(regexp_replace(col("Metin"), r"[.,;:!?]", "")))
metinler_df = metinler_df.withColumn("uzun_kelime_sayisi", expr("size(filter(kelimeler, x -> length(x) > 7))"))

remover = StopWordsRemover(inputCol="kelimeler", outputCol="filtered")
metinler_df = remover.transform(metinler_df)
metinler_df = metinler_df.withColumn("stopword_sayisi", col("kelime_sayisi") - size(col("filtered")))
metinler_df = metinler_df.withColumn("cumle_sayisi", size(split(col("Metin"), r"[.!?]")))
metinler_df = metinler_df.withColumn("karmasik_seviyesi", col("Etiket").cast(IntegerType()))

assembler = VectorAssembler(
    inputCols=[
        "kelime_sayisi", 
        "karakter_sayisi", 
        "ortalama_kelime_uzunlugu", 
        "noktalama_sayisi",
        "uzun_kelime_sayisi",
        "stopword_sayisi",
        "cumle_sayisi",
    ],
    outputCol="features"
)
metinler_df = assembler.transform(metinler_df)

lr = LogisticRegression(featuresCol="features", labelCol="karmasik_seviyesi")
lr_model = lr.fit(metinler_df)

dogrulama_df = spark.read.csv('dogrulama_metin.csv', header=True, inferSchema=True)
dogrulama_df = dogrulama_df.filter(trim(col("Metin")) != "")
dogrulama_df = tokenizer.transform(dogrulama_df)

dogrulama_df = dogrulama_df.withColumn("kelime_sayisi", size(col("kelimeler")))
dogrulama_df = dogrulama_df.withColumn("karakter_sayisi", length(col("Metin")))
dogrulama_df = dogrulama_df.withColumn("ortalama_kelime_uzunlugu", (col("karakter_sayisi") / (col("kelime_sayisi") + 1)))              
dogrulama_df = dogrulama_df.withColumn("noktalama_sayisi", length(col("Metin")) - length(regexp_replace(col("Metin"), r"[.,;:!?]", "")))
dogrulama_df = dogrulama_df.withColumn("uzun_kelime_sayisi", expr("size(filter(kelimeler, x -> length(x) > 7))"))

dogrulama_df = remover.transform(dogrulama_df)
dogrulama_df = dogrulama_df.withColumn("stopword_sayisi", col("kelime_sayisi") - size(col("filtered")))
dogrulama_df = dogrulama_df.withColumn("cumle_sayisi", size(split(col("Metin"), r"[.!?]")))
dogrulama_df = dogrulama_df.withColumn("karmasik_seviyesi", col("Etiket").cast(IntegerType()))

dogrulama_df = assembler.transform(dogrulama_df)
tahminler = lr_model.transform(dogrulama_df)

tahminler.select("Metin", "karmasik_seviyesi", "prediction").show(30)

evaluator = MulticlassClassificationEvaluator(
    labelCol="karmasik_seviyesi", 
    predictionCol="prediction", 
    metricName="accuracy"
)
accuracy = evaluator.evaluate(tahminler)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="karmasik_seviyesi",
    predictionCol="prediction",
    metricName="f1"
)
f1_score = f1_evaluator.evaluate(tahminler)

print(f"F1 Skoru: {f1_score}")
print(f"Modelin dogrulugu: {accuracy}")
