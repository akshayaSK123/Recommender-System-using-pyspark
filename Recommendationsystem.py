import os

import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()

"""# Option 1: Recommender System"""

# Importing all required libraries
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql import Row

"""Step 1: Import the MovieLens dataset"""

# Importing the input file
from google.colab import files
uploaded = files.upload()

#Read data from u.data file 
from pyspark import SparkConf,SparkContext
from pyspark import SparkContext
from pyspark.sql import functions as func
import pyspark.sql.functions as f
from pyspark.sql import *
from pyspark.sql.functions import col
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import desc
from pyspark import SparkContext
sc =SparkContext.getOrCreate()

import collections
df = spark.read.format('com.databricks.spark.csv').\
                               options(header='true', \
                                       delimiter='\t',\
                               inferschema='true').\
                load("u.data",header=False)
#Renaming the column header                
df2 = df.withColumnRenamed("_c0","userid").withColumnRenamed("_c1","itemid").withColumnRenamed("_c2","rating").withColumnRenamed("_c3","timestamp")
#Displaying the source file
df2.show()

"""Step 2: Build a recommendation model using Alternating Least Squares"""

# split training and testing
(training, test) = df2.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
als = ALS(userCol="userid", itemCol="itemid", ratingCol="rating", nonnegative=True)
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Model results in RMSE = nan due to cold start problem

"""RMSE for first model is nan due to cold start problem

Step 4 : Resolving Cold start problem and improving performance by cross validation

Resolving cold start problem
"""

# Build new recommendation model using ALS on the training data resolving cold start problem
als_basic = ALS(userCol="userid", itemCol="itemid", ratingCol="rating",coldStartStrategy="drop", nonnegative=True)
model_basic = als_basic.fit(training)
# Evaluate the model by computing the RMSE on the test data
predictions = model_basic.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Basic model gives RMSE value of 0.92

"""RMSE for basic model after resolving cold start problem is 0.92"""

als = ALS(userCol="userid", itemCol="itemid", ratingCol="rating",coldStartStrategy="drop", nonnegative=True)

#Tuning model using ParamGridBuilder
param_grid=ParamGridBuilder()\
            .addGrid(als.rank,(12,13,14))\
            .addGrid(als.maxIter,(5,10,15))\
            .addGrid(als.regParam,[0.01,0.05,0.10])\
            .build()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
#CrossValidation
tvs=TrainValidationSplit(estimator=als,estimatorParamMaps=param_grid,evaluator=evaluator)
model = tvs.fit(training)          

best_model=model.bestModel
predictions = best_model.transform(test)

rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

"""Root-mean-square error = 0.916

Improving performance by cross validation
"""

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

als = ALS(userCol="userid", itemCol="itemid", ratingCol="rating",coldStartStrategy="drop", nonnegative=True)

#Tuning model using ParamGridBuilder
param_grid=ParamGridBuilder()\
            .addGrid(als.rank,(3,5,8))\
            .addGrid(als.maxIter,(10,15,25))\
            .addGrid(als.regParam,[0.01,0.05,0.10])\
            .build()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
#CrossValidation
tvs=CrossValidator(estimator=als,estimatorParamMaps=param_grid,evaluator=evaluator,numFolds=5)
model = tvs.fit(training)          

best_model=model.bestModel
predictions = best_model.transform(test)

rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
print("Best Rank:",best_model.rank)
print("Best MaxIter:",best_model._java_obj.parent().getMaxIter())
print("Best RegParam:",best_model._java_obj.parent().getRegParam())

"""Improved RMSE
Root-mean-square error = 0.914
Best Rank: 5
Best MaxIter: 25
Best RegParam: 0.1

Step 5 : Output top 10 movies for all the users and write output into file
"""

import pandas as pd
Top10allusers=best_model.recommendForAllUsers(10)
Top10allusers=Top10allusers.toPandas()
user=[]
items=[]
for row in range(len(Top10allusers)):
  user.append(Top10allusers.iloc[row,0])
  rec=""
  for item in Top10allusers.iloc[row,1]:
    rec=rec+","+str(item.asDict()["itemid"])
  items.append(rec[2:])

Toprecdf=pd.DataFrame(data=zip(user,items),columns=["userid","items_recommended"])
Toprecdf.to_csv("Top10rec.txt", sep='\t')

#Displaying 10 rows of final result
Toprecdf.head(10)