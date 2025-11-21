from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# Spark Session

spark = SparkSession.builder.appName("Milestone3").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


# Load cleaned data (from Milestone 1)

df = spark.read.parquet("cleaned_traffic_data.parquet")

# Ensure Timestamp exists
df = df.filter(F.col("Timestamp").isNotNull())


# --- Advanced Time Analysis----

# 1) 3-hour windows (0–3, 3–6, ... 21–24)
df = df.withColumn(
    "Hour", F.hour("Timestamp")
).withColumn(
    "Hour_Window",
    (F.col("Hour") / 3).cast("int") * 3
)

# 2) Weekday vs Weekend
# Spark: 1 = Sunday ... 7 = Saturday
df = df.withColumn("DowNum", F.dayofweek("Timestamp"))
df = df.withColumn(
    "Day_Type",
    F.when(F.col("DowNum").isin(1,7), "Weekend").otherwise("Weekday")
)

# 3) Aggregations
violations_by_window = (
    df.groupBy("Hour_Window")
      .agg(F.count("*").alias("Total_Violations"))
      .orderBy("Hour_Window")
)

violations_by_daytype = (
    df.groupBy("Day_Type")
      .agg(F.count("*").alias("Total_Violations"))
)

# 4) Relationship between Violation Type and Peak Times (3-hr windows)
type_vs_window = (
    df.groupBy("Violation_Type", "Hour_Window")
      .agg(F.count("*").alias("Total_Violations"))
      .orderBy("Violation_Type", "Hour_Window")
)


#  ----Hotspot Identification----

# 1) Find locations with unusually high violation counts
total_violations = df.count()

violations_by_location = (
    df.groupBy("Location")
      .agg(F.count("*").alias("Total_Violations"))
      .orderBy(F.desc("Total_Violations"))
)

# simple hotspot rule: location with > average
avg_per_location = violations_by_location.agg(F.avg("Total_Violations")).first()[0]

hotspots = violations_by_location.filter(
    F.col("Total_Violations") > avg_per_location
)


# Save Results (Parquet)

violations_by_window.write.mode("overwrite").parquet("out3/violations_by_3hr_window")
violations_by_daytype.write.mode("overwrite").parquet("out3/violations_by_daytype")
type_vs_window.write.mode("overwrite").parquet("out3/type_vs_timewindow")
violations_by_location.write.mode("overwrite").parquet("out3/violations_by_location")
hotspots.write.mode("overwrite").parquet("out3/hotspots")


# Quick Sample Output

print("=== 3-Hour Window Violations ===")
violations_by_window.show()

print("=== Weekday vs Weekend ===")
violations_by_daytype.show()

print("=== Type vs Time Window ===")
type_vs_window.show(20, truncate=False)

print("=== All Locations Ranked ===")
violations_by_location.show(10, truncate=False)

print("=== Hotspots Identified ===")
hotspots.show(10, truncate=False)

spark.stop()