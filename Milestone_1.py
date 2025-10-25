import random
import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType, DoubleType

spark = SparkSession.builder.appName("TrafficViolation").getOrCreate()

schema = StructType([
    StructField("Violation_ID", StringType(), True),
    StructField("Timestamp", StringType(), True),
    StructField("Location", StringType(), True),
    StructField("Violation_Type", StringType(), True),
    StructField("Vehicle_Type", StringType(), True),
    StructField("Severity", IntegerType(), True)
])

violation_types = ["Speeding", "Red Light", "No Helmet", "Wrong Lane"]
vehicle_types = ["Car", "Bike", "Bus", "Truck"]


data = []
for i in range(1000):  # 1000 sample records
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

df_spark = spark.read.csv("traffic_data.csv", header=True, schema=schema)
df_spark.show(5)

df_clean = df_spark.na.fill({"Violation_Type": "Unknown", "Severity": 0})

df_clean = df_clean.withColumn("Timestamp", to_timestamp("Timestamp", "yyyy-MM-dd HH:mm:ss"))

valid_types = ["Speeding", "Red Light", "No Helmet", "Wrong Lane"]
df_clean = df_clean.filter(df_clean.Violation_Type.isin(valid_types))

df_clean.write.parquet("cleaned_traffic_data.parquet")

