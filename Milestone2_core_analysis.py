from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd

# ---------- Spark ----------
spark = SparkSession.builder.appName("Milestone2").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ---------- Read cleaned data ----------
# read the simulated cleaned parquet
df_sim = spark.read.parquet("cleaned_traffic_data.parquet")

# read YOLO parquet if it exists, else create empty df with same schema
import os
if os.path.exists("detected_violations.parquet"):
    df_yolo = spark.read.parquet("detected_violations.parquet")
else:
    # create empty DF with same schema as df_sim
    df_yolo = spark.createDataFrame([], schema=df_sim.schema)

# Ensure both have same columns & types: cast timestamp to timestamp type
df_sim = df_sim.withColumn("Timestamp", F.to_timestamp("Timestamp"))
df_yolo = df_yolo.withColumn("Timestamp", F.to_timestamp("Timestamp"))

# union safely by name
df = df_sim.unionByName(df_yolo.select(df_sim.columns))

# now derive time features and continue the rest of your aggregations...
df.show(5, truncate=False)
 
# ---------- Derive time features ----------
df = (
    df.withColumn("Hour", F.hour("Timestamp"))
      # Spark's dayofweek: 1=Sun ... 7=Sat
      .withColumn("DowNum", F.dayofweek("Timestamp"))
      .withColumn("DayOfWeek", F.date_format("Timestamp", "E"))  # Mon, Tue, ...
      .withColumn("Month", F.month("Timestamp"))
      .withColumn("Year", F.year("Timestamp"))
)

df.select("Timestamp", "Hour", "DowNum", "DayOfWeek", "Month", "Year").show(5, truncate=False)

# ---------- Aggregations ----------
# Per hour (sorted 0..23)
violations_per_hour = (
    df.groupBy("Hour")
      .agg(F.count("*").alias("Total_Violations"))
      .orderBy("Hour")
)
violations_per_hour.show(24, truncate=False)

# Per day of week (sorted using DowNum)
violations_per_day = (
    df.groupBy("DayOfWeek", "DowNum")
      .agg(F.count("*").alias("Total_Violations"))
      .orderBy("DowNum")
      .drop("DowNum")
)
violations_per_day.show(truncate=False)

# By offense type
violations_by_type = (
    df.groupBy("Violation_Type")
      .agg(F.count("*").alias("Total_Violations"))
      .orderBy(F.desc("Total_Violations"))
)
violations_by_type.show(truncate=False)

# Cross-tab: Violation type × Hour (pivot 0..23, fill nulls with 0)
xtab_type_by_hour = (
    df.groupBy("Violation_Type")
      .pivot("Hour", list(range(24)))
      .agg(F.count("*"))
      .na.fill(0)
      .orderBy("Violation_Type")
)
xtab_type_by_hour.show(truncate=False)

# By location + Top N
violations_by_location = (
    df.groupBy("Location")
      .agg(F.count("*").alias("Total_Violations"))
      .orderBy(F.desc("Total_Violations"))
)
violations_by_location.show(truncate=False)

TOP_N = 5
top_locations = violations_by_location.limit(TOP_N)
top_locations.show(truncate=False)

# ---------- Write outputs (organized) ----------
violations_per_hour.write.mode("overwrite").parquet("out/agg_by_hour")
violations_per_day.write.mode("overwrite").parquet("out/agg_by_dow")
violations_by_type.write.mode("overwrite").parquet("out/agg_by_type")
xtab_type_by_hour.write.mode("overwrite").parquet("out/xtab_type_by_hour")
violations_by_location.write.mode("overwrite").parquet("out/agg_by_location")
top_locations.write.mode("overwrite").parquet("out/top_locations")

spark.stop()