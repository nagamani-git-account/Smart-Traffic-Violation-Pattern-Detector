from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, dayofweek, month, year, to_timestamp
from pyspark.sql.functions import count

spark = SparkSession.builder.appName("Milestone2").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet("cleaned_traffic_data.parquet")
df.show(5)

# make sure timestamp column is in correct format
df = df.withColumn("Timestamp", to_timestamp("Timestamp"))

# add new columns
df = df.withColumn("Hour", hour("Timestamp"))
df = df.withColumn("DayOfWeek", dayofweek("Timestamp"))
df = df.withColumn("Month", month("Timestamp"))
df = df.withColumn("Year", year("Timestamp"))

df.show(5)


violations_per_hour = df.groupBy("Hour").agg(count("*").alias("Total_Violations"))
violations_per_hour.show()

violations_per_day = df.groupBy("DayOfWeek").agg(count("*").alias("Total_Violations"))
violations_per_day.show()

violations_by_type = df.groupBy("Violation_Type").agg(count("*").alias("Total_Violations"))
violations_by_type.show()

violations_by_location = df.groupBy("Location").agg(count("*").alias("Total_Violations"))
violations_by_location.show()

top_locations = violations_by_location.orderBy("Total_Violations", ascending=False).limit(5)
top_locations.show()

violations_per_hour.write.mode("overwrite").parquet("violations_per_hour.parquet")
violations_per_day.write.mode("overwrite").parquet("violations_per_day.parquet")
violations_by_type.write.mode("overwrite").parquet("violations_by_type.parquet")
violations_by_location.write.mode("overwrite").parquet("violations_by_location.parquet")

spark.stop()