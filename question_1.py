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

result.lookup('924')  #[[('11860', 1),('15416', 1),('43748', 1),('6995', 1),('439', 1),('45881', 1),('2409', 1)]]
result.lookup('8941') #[[('8944', 2), ('8943', 2), ('8940', 1)]]
result.lookup('8942') #[[('8939', 3), ('8940', 1), ('8943', 1), ('8944', 1)]]
result.lookup('9019') #[[('9022', 2), ('317', 1), ('9023', 1)]]
result.lookup('9020') #[[('9021', 3), ('9016', 2), ('9017', 2), ('9022', 2), ('317', 1), ('9023', 1)]]
result.lookup('9021') #[[('9020', 3), ('9022', 2), ('9016', 2), ('9017', 2), ('317', 1), ('9023', 1)]]
result.lookup('9022') #[[('9019', 2),('9021', 2),('9020', 2),('9016', 1),('9017', 1),('317', 1),('9023', 1)]]
result.lookup('9990') #[[('34642', 1),('34485', 1),('13478', 1),('34299', 1),('13134', 1),('13877', 1),('37941', 1)]]
result.lookup('9992') #[[('9989', 4), ('9987', 4), ('35667', 3), ('9991', 2)]]
result.lookup('9993') #[[('9991', 5),('13877', 1),('34485', 1),('34642', 1),('13478', 1),('34299', 1),('13134', 1),('37941', 1)]]

#recommendations
for i in ['924','8941','8942','9019','9020','9021','9022','9990','9992','9993']:
    print("Recommendations for "+i)
    for j in result.lookup(i):
        for j in i:
            print(j[0])