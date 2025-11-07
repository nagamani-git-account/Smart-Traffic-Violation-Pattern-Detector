import os
import random
import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# --------------------------
# Step 1: Environment setup
# --------------------------
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["hadoop.home.dir"] = "C:\\hadoop"
os.environ["PATH"] += os.pathsep + os.path.join("C:\\hadoop", "bin")
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# --------------------------
# Step 2: Create Spark Session
# --------------------------
spark = (
    SparkSession.builder
    .appName("TrafficViolation")
    .config("spark.sql.warehouse.dir", "file:///C:/tmp/hive/warehouse")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
    .config("spark.driver.extraJavaOptions", "-Djava.library.path=C:/hadoop/bin")
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .getOrCreate()
)


# --------------------------
# Step 3: Define schema
# --------------------------
schema = StructType([
    StructField("Violation_ID", StringType(), True),
    StructField("Timestamp", StringType(), True),
    StructField("Location", StringType(), True),
    StructField("Violation_Type", StringType(), True),
    StructField("Vehicle_Type", StringType(), True),
    StructField("Severity", IntegerType(), True)
])

# --------------------------
# Step 4: Create sample dataset and save to CSV
# --------------------------
violation_types = ["Speeding", "Red Light", "No Helmet", "Wrong Lane"]
vehicle_types = ["Car", "Bike", "Bus", "Truck"]

data = []
for i in range(1000):
    data.append({
        "Violation_ID": f"V{i+1:04d}",
        "Timestamp": str(datetime.datetime.now() - datetime.timedelta(minutes=random.randint(0, 10000))),
        "Location": f"Intersection_{random.randint(1, 20)}",
        "Violation_Type": random.choice(violation_types),
        "Vehicle_Type": random.choice(vehicle_types),
        "Severity": random.randint(1, 5)
    })

df = pd.DataFrame(data)
df.to_csv("traffic_data.csv", index=False)

# --------------------------
# Step 5: Load CSV into Spark DataFrame
# --------------------------
df_spark = spark.read.csv("traffic_data.csv", header=True, schema=schema)
df_spark.show(5)

# --------------------------
# Step 6: Data cleaning
# --------------------------
# Fill missing values
df_clean = df_spark.na.fill({"Violation_Type": "Unknown", "Severity": 0})

# Convert timestamp
df_clean = df_clean.withColumn("Timestamp", to_timestamp("Timestamp", "yyyy-MM-dd HH:mm:ss.SSSSSS"))

# Filter valid violation types
valid_types = ["Speeding", "Red Light", "No Helmet", "Wrong Lane"]
df_clean = df_clean.filter(df_clean.Violation_Type.isin(valid_types))

# --------------------------
# Step 7: Write to Parquet
# --------------------------
df_clean.write.mode("overwrite").parquet("cleaned_traffic_data.parquet")

# Quick read-back check
df_check = spark.read.parquet("cleaned_traffic_data.parquet")
print("Parquet read-back sample:")
df_check.show(5, truncate=False)

spark.stop()