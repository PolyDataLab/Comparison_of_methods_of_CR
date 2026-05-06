#! /usr/bin/python3 -u

import os
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from scipy.io import mmwrite as save
from scipy.sparse import csr_matrix as sparse
import random




dir = 'edm&ipsj/stu_nets_EXT/'

kg_dir = 'edm&ipsj/course_KG_EXT'

if not os.path.exists(dir):
    os.makedirs(dir)

########################################################################################################################

trainMap = dict()
itemMap = dict()

########################################################################################################################

def loadKG(item):

    global itemMap
    global features

    # 使得itemMap 存 item-feature 的key-value
    with open(os.path.join(kg_dir, item), 'r', encoding='utf-8') as file:
        results = file.read().splitlines()

    for result in results:
        features.add(result)
        #set.add(result["object"]["value"])
        itemMap[item].add(result)


########################################################################################################################

class Rate:
    itemId = 0
    rating = 0

    def __init__(self, itemId, rating):
        self.itemId = itemId
        self.rating = float(rating)

    def getRating(self):
        return self.rating

    def getItemId(self):
        return self.itemId

########################################################################################################################

print("Loading...")

# course map
filename = 'edm&ipsj/course_map.tsv'
file = open(filename, "r", encoding="utf-8")
lines = file.read().splitlines()
file.close()
courseMap = dict(( line.split("\t")[0], line.split("\t")[1]) for line in lines[1:])

print(filename + " loaded.")

# training file
filename = 'edm&ipsj/mooc_data/Data/mooc.train.rating'
file = open(filename, "r", encoding="utf-8")
lines = file.read().splitlines()
file.close()

items = set()
users = set()

for line in lines:
    words = line.split("\t")
    # user, item, rate, timestamp = line.split("\t")
    user, item, rate = [words[0], words[1], words[2]]
    if(user not in trainMap):
        trainMap[user] = []
    if(item in courseMap):
        trainMap[user].append(Rate(item, rate))
        items.add(item)
    users.add(user)

usersList = list(users)
itemsList = list(items)

print(filename + " loaded. \n\tItems: {} \n\tUsers: {}".format(len(items), len(users)))

########################################################################################################################

for item in itemsList:
    itemMap[item] = set()


print("Fetching resources...")

features = set()

p = Pool()
p.map(loadKG, itemsList)

p.close()
p.join()

# print(len(features))
print("Resources fetched.")

# 全部的feature
featuresList = list(features)

########################################################################################################################

# 设计负实例

num_negative_samples = 4  # 每个用户的负实例数量

# 为每个用户选择负实例
for user in usersList:
    user_items = set([rate.getItemId() for rate in trainMap[user]])
    all_items = set(itemsList)
    non_interacted_items = list(all_items - user_items)
    negative_samples = random.sample(non_interacted_items, min(num_negative_samples, len(non_interacted_items)))

    # 将负实例加入到trainMap中，并分配低评分
    for item in negative_samples:
        trainMap[user].append(Rate(item, 0))  # 评分为0表示负实例








print("Creating matrix...")

for user in usersList:
    rates = trainMap[user]
    items = [rate.getItemId() for rate in rates]
    ratings = np.array([rate.getRating() for rate in rates])
    features = set()
    # 每个user 的 评分过的物品 的 feature 集合
    for item in items:
        features.update(itemMap[item])

    # 物品 X 特征 矩阵
    weights = np.zeros((len(items), len(features)))
    features = list(features)
    # 将feature 映射成ID
    index = dict((feature, count) for count, feature in enumerate(features))

    for count, item in enumerate(items):
        row = count
        for feature in itemMap[item]:
            # 获取 每个物品的 feature 的ID
            col = int(index[feature])
            weights[row, col] = 1

    dir2 = dir + str(user) + "/"

    if not(os.path.exists(dir2)):
        os.makedirs(dir2)

    # mask: 物品×特征权重矩阵
    # matrix: 每个用户单独一个文件,评分矩阵
    # 行索引 列索引 值(稀疏矩阵保存方法)
    save(dir2 + "mask", sparse(weights))
    save(dir2 + "matrix", sparse(ratings))

    # features: 用户在意的特征
    file = open(dir2 + "features", "w", encoding='utf-8')
    file.write("\n".join(features))
    file.close()

########################################################################################################################

print("\nDone.")
