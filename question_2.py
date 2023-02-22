import pyspark
from pyspark import SparkContext, SparkConf
from collections import Counter
import itertools
from pyspark.sql import *
from pyspark.sql.functions import *
import re, sys, operator
import itertools


# create the Spark Session
spark = SparkSession.builder.getOrCreate()

# create the Spark Context
sc = spark.sparkContext

filepath='/Users/rupsachakraborty/Desktop/Spring23/MDM/homework1/data/browsing.txt'
lines=sc.textFile(filepath)

baskets = lines.map(lambda l: l.split())
N = baskets.count()
baskets = baskets.map(lambda b: sorted(set(b)))

def single(basket):
    return [(item,1) for item in basket]


individual_filtered = baskets.flatMap(single).reduceByKey(operator.add).filter(lambda x: x[1] >= 100)
individual_filtered_dict = {}

for item, support in individual_filtered.collect():
    individual_filtered_dict[item] = support

individual_filtered_dict = sc.broadcast(individual_filtered_dict)

def pair(basket):
    singles = individual_filtered_dict.value
    ret = []
    for i in range(len(basket)):
        if basket[i] in singles:
            for j in range(i):
                if basket[j] in singles:
                    ret.append(((basket[j], basket[i]), 1)) # basket is sorted
    return ret

pairs_filtered = baskets.flatMap(pair).reduceByKey(operator.add).filter(lambda x: x[1] >= 100)

def confidence_pair(double_support):
    double, support = double_support
    support = float(support)
    u, v = double
    singles = individual_filtered_dict.value
    uv_conf = support / singles[u]
    vu_conf = support / singles[v]
    return (('%s -> %s' % (u, v), uv_conf),
            ('%s -> %s' % (v, u), vu_conf))


pairs_conf = pairs_filtered.flatMap(confidence_pair).sortBy(lambda x: (-x[1], x[0]))
pairs_filtered_dict = {}
for entry, support in pairs_filtered.collect():
    pairs_filtered_dict[entry] = support

pairs_filtered_dict = sc.broadcast(pairs_filtered_dict)


def triplets(basket):
    pairs = pairs_filtered_dict.value
    singles = individual_filtered_dict.value
    ret = []
    for i in range(len(basket)):
        if basket[i] not in singles:
            continue
        for j in range(i):
            if basket[j] not in singles:
                continue
            if (basket[j], basket[i]) not in pairs:
                continue
            for k in range(j):
                if basket[k] not in singles:
                    continue
                if (basket[k], basket[j]) not in pairs:
                    continue
                if (basket[k], basket[i]) not in pairs:
                    continue
                ret.append(((basket[k], basket[j], basket[i]), 1))
    return ret

triplets_filtered = baskets.flatMap(triplets).reduceByKey(operator.add).filter(lambda x: x[1] >= 100)

def confidence_triplets(triple_support):
    pairs = pairs_filtered_dict.value
    triple, support = triple_support
    support = float(support)
    u, v, w = triple
    uv_w = support / pairs[u, v]
    uw_v = support / pairs[u, w]
    vw_u = support / pairs[v, w]
    return (('(%s, %s) -> %s' % (u, v, w), uv_w),
            ('(%s, %s) -> %s' % (u, w, v), uw_v),
            ('(%s, %s) -> %s' % (v, w, u), vw_u))

triples_conf = triplets_filtered.flatMap(confidence_triplets).sortBy(lambda x: (-x[1], x[0]))


print(pairs_conf.take(5))
print(triples_conf.take(5))