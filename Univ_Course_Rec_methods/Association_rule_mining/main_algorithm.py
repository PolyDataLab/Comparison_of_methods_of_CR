import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import random
import tqdm
from preprocess import *
from utils import *
from offered_courses import *
import math
from preprocess import *
from sklearn.cluster import KMeans
import numpy as np
from apyori import apriori
#from mlxtend.frequent_patterns import apriori, association_rules
# from mlxtend.frequent_patterns import apriori, association_rules


def generate_rules(data, reversed_item_dict):
    
    list_of_instances = []
    for x in range(len(data)):
        baskets = data['baskets'][x] # contains course registration history of each student (a list of lists taken in semester-by-semester)
        list_of_items = []
        for basket in baskets:
            for item in basket:
                list_of_items.append(reversed_item_dict[item])
        list_of_instances.append(list_of_items)
    
    transactions = list_of_instances.copy()
    rules = apriori(transactions, min_support = 0.15, min_confidence = 0.4, min_lift = 1, min_length = 1)
    min_support1 = 0.15
    min_confidence1 = 0.4
    print("minimum support: ", min_support1)
    print("minimum confidence: ", min_confidence1)
    #print(rules)
    
    list_rules_all = list(rules)
    list_rules_new = []
    for rule in list_rules_all: 
        #for item in rule:
            # pair = list(item[0])
            # if len(pair)==2:
            pair = [list(rule[2][0][0]), list(rule[2][0][1])]
            if (len(pair[0])>0 and len(pair[1])>0):
                items = [x for x in pair]
                # print("Rule: " + str(items[0]) + "-->" + str(items[1]))
                # print("support: ", float(rule[1]))
                # print("confidence: ", float(rule[2][0][2]))
                # print("lift: ", float(rule[2][0][3]))
                row = [items[0], items[1], rule[1], float(rule[2][0][2]), float(rule[2][0][3])]
                list_rules_new.append(row)
                #print("==========")

    list_rules_df = pd.DataFrame(list_rules_new, columns=['lhs', 'rhs', 'sup', 'con', 'lift'])
    #list_rules_df = pd.DataFrame(list_rules_all)
    list_rules_df.to_csv('./Association_rules/rules_v9_extended_15_40.csv') 
    list_rules_df.to_json('./Association_rules/rules_v9_extended_15_40.json',  orient='records', lines= True) 
    
    return list_rules_df 


if __name__ == '__main__':
   
   train_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/train_data_all.json', orient='records', lines= True)
   train_data, item_dict, user_dict, reversed_item_dict, reversed_user_dict = preprocess_train_data_part1(train_data)
   
   train_all, train_set_without_target, target, max_len = preprocess_train_data_part2(train_data)   
   rules = generate_rules(train_all, reversed_item_dict)
