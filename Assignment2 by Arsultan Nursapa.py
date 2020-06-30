''' 
From the analysis made it can be said that there were 2 hackers only as clustering algorithm with K=2 created two same size clusters
Please follow the code:
'''

from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler 

spark = SparkSession.builder.appName('k-means clustering').getOrCreate()
dataset = spark.read.csv("C:/6290Internetofthings/hack_data.txt",header=True,inferSchema=True)
dataset.head()
'''
Row(Session_Connection_Time=8.0, Bytes Transferred=391.09, Kali_Trace_Used=1, Servers_Corrupted=2.96, Pages_Corrupted=7.0, Location='Slovenia', WPM_Typing_Speed=72.37)
'''
dataset.describe().show()
'''
+-------+-----------------------+------------------+------------------+-----------------+------------------+-----------+------------------+
|summary|Session_Connection_Time| Bytes Transferred|   Kali_Trace_Used|Servers_Corrupted|   Pages_Corrupted|   Location|  WPM_Typing_Speed|
+-------+-----------------------+------------------+------------------+-----------------+------------------+-----------+------------------+
|  count|                    334|               334|               334|              334|               334|        334|               334|
|   mean|     30.008982035928145| 607.2452694610777|0.5119760479041916|5.258502994011977|10.838323353293413|       null|57.342395209580864|
| stddev|     14.088200614636158|286.33593163576757|0.5006065264451406| 2.30190693339697|  3.06352633036022|       null| 13.41106336843464|
|    min|                    1.0|              10.0|                 0|              1.0|               6.0|Afghanistan|              40.0|
|    max|                   60.0|            1330.5|                 1|             10.0|              15.0|   Zimbabwe|              75.0|
+-------+-----------------------+------------------+------------------+-----------------+------------------+-----------+------------------+
'''
dataset.columns
'''
['Session_Connection_Time',
 'Bytes Transferred',
 'Kali_Trace_Used',
 'Servers_Corrupted',
 'Pages_Corrupted',
 'WPM_Typing_Speed']
'''
feat_cols = ['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used', 'Servers_Corrupted', 'Pages_Corrupted','WPM_Typing_Speed']

vec_assembler = VectorAssembler(inputCols = feat_cols, outputCol='features')
feature_data = vec_assembler.transform(dataset)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scalerModel = scaler.fit(feature_data)
final_data = scalerModel.transform(feature_data)

kmeans2 = KMeans(featuresCol='scaledFeatures',k=2)
kmeans3 = KMeans(featuresCol='scaledFeatures',k=3) 
model2 = kmeans2.fit(final_data)
model3 = kmeans3.fit(final_data)
wssse2 = model2.computeCost(final_data)
wssse3 = model3.computeCost(final_data) 

print ("If K=2, Within Set Sum of Squared Errors = " + str(wssse2)) 

'''
If K=2, Within Set Sum of Squared Errors = 601.7707512676716
'''

print ("If K=3, Within Set Sum of Squared Errors = " + str(wssse3))

'''
If K=3, Within Set Sum of Squared Errors = 434.75507308487647
'''

model2.transform(final_data).groupBy('prediction').count().show()

'''
+----------+-----+
|prediction|count|
+----------+-----+
|         1|  167|
|         0|  167|
+----------+-----+
'''

model3.transform(final_data).groupBy('prediction').count().show()
'''
+----------+-----+
|prediction|count|
+----------+-----+
|         1|   88|
|         2|   79|
|         0|  167|
+----------+-----+
'''

