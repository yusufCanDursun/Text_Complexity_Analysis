import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, size, monotonically_increasing_id
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import IntegerType

#ortam degiskenleri deistirmek gerekebilir
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] += os.pathsep + "C:\\hadoop\\bin"
os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"

spark = SparkSession.builder \
    .appName("MetinKarmasikAnalis") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

metinler_df = spark.read.text('metinler/*.txt')
metinler_df = metinler_df.withColumn("id", monotonically_increasing_id())
metinler_df.show(5)

tokenizer = Tokenizer(inputCol="value", outputCol="kelimeler")
pipeline = Pipeline(stages=[tokenizer])
model = pipeline.fit(metinler_df)
processed_df = model.transform(metinler_df)

processed_df = processed_df.withColumn("kelime_sayisi", size(col("kelimeler")))

from pyspark.sql.functions import rand
processed_df = processed_df.withColumn("karmasik_seviyesi", (rand() * 2).cast(IntegerType()))

assembler = VectorAssembler(inputCols=["kelime_sayisi"], outputCol="features")
processed_df = assembler.transform(processed_df)

lr = LogisticRegression(featuresCol="features", labelCol="karmasik_seviyesi")
model = lr.fit(processed_df)

predictions = model.transform(processed_df)
predictions.select("value", "kelime_sayisi", "prediction").show(5)

evaluator = MulticlassClassificationEvaluator(labelCol="karmasik_seviyesi", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Modelin dogrulugu: {accuracy}")
