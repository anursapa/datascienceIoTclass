#!/usr/bin/env python
# coding: utf-8LinearRegression


# Following is the assginment 1 by Arsultan Nursapa:
# Please execute all the code blocks below one-by-one
# Summary of the work is evaluated at the end of the file in Conclusion section (cell #10)


# In[1]:
# # Initiating Spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder\
           .appName("Python Spark Linear Regression example")\
           .config("spark.some.config.option", "some-value")\
           .getOrCreate()

# In[2]:
# Import data from cruise_ship_info.txt into variable ad using spark.read.csv method

ad = spark.read.csv("C:/6290Internetofthings/cruise_ship_info.txt", header=True, inferSchema=True)
ad.show(5)
# In[3]:
# Transform data structure. Cut down columns, by slicing out columns 3 to 7 and name them 'features' and crew members amount to be named 'label' 

from pyspark.ml.linalg import Vectors
ad_df = ad.rdd.map(lambda x: [Vectors.dense(x[3:8]), x[-1]]).toDF(['features', 'label'])
ad_df.show(5)



# In[4]:
# Build linear regression model, by associating 'features' and 'labels' to featuresCol and labelCol

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol = 'label')


# In[5]:
# Fit the model by using LinearRegression().fit() method
lr_model = lr.fit(ad_df)


# In[6]:
# Prediction
pred = lr_model.transform(ad_df)
pred.show(5)


# In[7]:
# Module evaluation

from pyspark.ml.evaluation import RegressionEvaluator 
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='label')
evaluator.setMetricName('r2').evaluate(pred)

# In[8]:
# Define function modelSummary to visualize the regression model summary

def modelsummary(model, param_names):
    import numpy as np
    print ("Note: the last rows are the information for Intercept")
    print ("##","-------------------------------------------------")
    print ("##","  Estimate   |   Std.Error | t Values  |  P-value")
    coef = np.append(list(model.coefficients), model.intercept)
    Summary=model.summary
    param_names.append('intercept')

    for i in range(len(Summary.pValues)):
        print ("##",'{:10.6f}'.format(coef[i]),\
        '{:14.6f}'.format(Summary.coefficientStandardErrors[i]),\
        '{:12.3f}'.format(Summary.tValues[i]),\
        '{:12.6f}'.format(Summary.pValues[i]), \
        param_names[i])

    print ("##",'---')
    print ("##","Mean squared error: % .6f" \
           % Summary.meanSquaredError, ", RMSE: % .6f" \
           % Summary.rootMeanSquaredError )
    print ("##","Multiple R-squared: %f" % Summary.r2, "," )
    print ("##","Multiple Adjusted R-squared: %f" % Summary.r2adj, ", \
            Total iterations: %i"% Summary.totalIterations)

param_names = ad.columns[3:8]
modelsummary(lr_model, param_names)
# In[9]:
# ## Linear regression with cross-validation


training, test = ad_df.randomSplit([0.8, 0.2], seed=123)
lr_model = lr.fit(training)
pred = lr_model.transform(training)
pred.show(5)
param_names = ad.columns[3:8]
modelsummary(lr_model, param_names)

# Make predictions.
pred_test = lr_model.transform(test)
pred_test.show(5)

from pyspark.ml.evaluation import RegressionEvaluator
# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(labelCol="label",
                                predictionCol="prediction",
                                metricName="rmse")

rmse = evaluator.evaluate(pred_test)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
# In[10]: 
# Conclusion:

# By doing the regression model following can be determined:
#    - among the 5 independent variables (age wasn't taken into account), 3 variables are statistically significant (Passengers, Length, Cabins), because their p-value is lower than 0.05;
#    - prediction column can be used as optimal crew number;
#    - Root Mean Squared Error (RMSE)  is equal to 0.89779 which is reletively good.