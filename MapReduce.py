import pyspark
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from collections import Counter
import itertools
from pyspark.sql import *
from pyspark.sql.functions import *

# create the Spark Session
spark = SparkSession.builder.getOrCreate()

# create the Spark Context
sc = spark.sparkContext

filepath='/Users/rupsachakraborty/Desktop/Spring23/MDM/homework1/data/soc-LiveJournal1Adj.txt'

def friends_and_mutuals(line):
    minimum = -34589849
    user, friends = line.split('\t')
    friends = friends.split(',')
    friends = [((user, friend), minimum) for friend in friends]
    mutuals = [(pair, 1) for pair in itertools.permutations(friends, 2)]
    return friends + mutuals

N=10 

result= (sc
                  .textFile(filepath)
                  .flatMap(friends_and_mutuals)
                  .reduceByKey( lambda total, current: total + current )
                  .filter(lambda tup: tup[1] > 0)  #tup=(pairs,counts)
                  .map(lambda a: (a[0][0], (a[1], a[0][1])))  #a=((from,to),counts) here to is a friend as m>0 in previous step
                  .groupByKey()
                  .map(lambda b:(b[0], Counter( dict( (friend, count) for count, friend in b[1])).most_common(N))) #b=(from,recommendations)
                  #.cache()
                   )



#recommendations
for i in ['924','8941','8942','9019','9020','9021','9022','9990','9992','9993']:
    print("Recommendations for "+i)
    for j in result.lookup(i):
        for j in i:
            print(j[0])