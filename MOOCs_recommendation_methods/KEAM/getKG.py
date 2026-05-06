#! /usr/bin/python3 -u

from os.path import exists, join
from os import makedirs
from multiprocessing.pool import ThreadPool as Pool
import pandas as pd
import os




# 得到知识图谱 每个item对应类别


dir = 'edm&ipsj/course_KG_EXT' + "/"


########################################################################################################################

trainMap = dict()
itemMap = dict()
# featureMap = dict()

########################################################################################################################

print("Loading...")


csv_file_path = 'edm&ipsj/mooc_data/data_EXT.csv'
df = pd.read_csv(csv_file_path, encoding='utf-8')  # 或者使用适当的编码


# 初始化物品的知识图谱映射
itemMap = {}
df['sub_type'] = df['sub_type'].astype(str)

# 为data.csv中的每个物品创建知识图谱
for index, row in df.iterrows():
    item = row['course_index']
    types = set(row['type'].split(' '))  # 从类型字段分割并创建集合
    sub_types = set(row['sub_type'].replace(',', ' ').split())  # 替换逗号并分割创建集合
    combined_set = types.union(sub_types)  # 合并两个集合
    # 合并 type 和 sub_type 的值，存入 itemMap
    itemMap[item] = combined_set

# 创建目录用于存储知识图谱
if not os.path.exists(dir):
    os.makedirs(dir)

# 保存每个物品的知识图谱到文件
for item in itemMap:
    with open(join(dir, str(item)), 'w', encoding='utf-8') as file:
        for feat in itemMap[item]:
            file.write("{}\n".format(feat))

print("\nDone.")


