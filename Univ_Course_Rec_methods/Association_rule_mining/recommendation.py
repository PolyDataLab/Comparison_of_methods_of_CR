#!/usr/bin/env python
from utils import *
import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Course enrollment data:')

import numpy as np
import pandas as pd
import json
import random
from tqdm import tqdm
import tensorflow as tf
#import gensim
from preprocess import *
from offered_courses import *
import math
import time


def recommending_target_courses_valid_test(data_set_without_target, dataTest, offered_courses, item_list, item_dict, reversed_item_dict, reversed_user_dict3, rules, match_thr):
    #top_k= 5
    top_items_per_user= []
    # Making list of user baskets for each user
    time_baskets = data_set_without_target.baskets.values
    user_baskets_all= []
    prior_items_list = []
    for baskets in tqdm(time_baskets):
        user_baskets_all.append(baskets)
        prior_items = []
        for basket in baskets:
            for item in basket:
                if item not in prior_items:
                    prior_items.append(item)
        prior_items_list.append(prior_items)

    target_basket=[]
    time_baskets = dataTest.baskets.values
    #userID_values = dataTest.userID.values
    for baskets in tqdm(time_baskets):
        target_basket_items= []
        #ann_target_basket_list1=[]
        #list2=[]
        for item in baskets:
            if item in item_list:
                target_basket_items.append(item)
        target_basket.append(target_basket_items)
    last_semester_values = dataTest.last_semester.values
    target_semesters= []
    for semester in tqdm(last_semester_values):
        target_semesters.append(semester)

    
    #print(target_basket)

    for i in range(len(data_set_without_target)):
        index_ij_dict= {}
        user_baskets = user_baskets_all[i]
        #len_user_baskets = len(user_baskets)
        len_target_basket = len(target_basket[i])
        target_semester = target_semesters[i]
        prior_courses = prior_items_list[i]

        #using filtering function to get the offered courses in the target semester and remove prior courses taken by user
        items_fil = []
        for item4 in item_list:
            if not filtering(item4, user_baskets, offered_courses[target_semester]):
                items_fil.append(item4)
        #using column sum for scores for all prior items of last prior basket
        score = {}
        #for item1 in range(len(item_list)):
        index3 = 0
        for j in range(len(rules)):
            lhs = rules['lhs'][index3]
            rhs = rules['rhs'][index3]
            support = float(rules['sup'][index3])
            conf = float(rules['con'][index3])
            lift = float(rules['lift'][index3])
            n_match= len((set(prior_courses) & set(lhs)))
            per_match = (n_match/ len(lhs)) * 100
            if per_match>= match_thr: #and conf>=0.4
                for item in rhs:
                    if item in items_fil:
                        if item not in score:
                            score[item] = conf
                        else:
                            score[item] += conf
                # for item in lhs:
                #     if item in items_fil:
                #         if item not in score:
                #             score[item] = conf
                #         else:
                #             score[item] += conf
            index3 += 1
        
        score = dict(sorted(score.items(), key= lambda item: item[1], reverse= True))
        #print(score)
        #print(index_ij_dict)
        top_k_count= 0
        #list_key= []
        list2= []
        #list3 =[]
        top_k = len_target_basket
        # top_items_per_user.append(list1)
        for item7 in score.keys():
            #print(index_j)
            # if reversed_item_dict[item7] not in list2:
            if item7 not in list2:
                list2.append(item7)
                top_k_count += 1
                if(top_k_count==top_k): break
        top_items_per_user.append(list2)

    return top_items_per_user


#calculate recall
def recall_cal(top_item_list, target_item_list, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred):
        t_length= len(target_item_list)
        correct_preds= len((set(top_item_list) & set(target_item_list)))
        actual_bsize= t_length
        if(correct_preds>=1): count_at_least_one_cor_pred += 1
        if correct_preds>=2: count_at_least_two_cor_pred+= 1
        if correct_preds>=3: count_at_least_three_cor_pred+= 1
        if correct_preds>=4: count_at_least_four_cor_pred+= 1
        if correct_preds>=5: count_at_least_five_cor_pred+= 1
        if correct_preds==actual_bsize: count_all_cor_pred+= 1

        if (actual_bsize>=6): 
            if(correct_preds==1): count_cor_pred[6,1]+= 1
            if(correct_preds==2): count_cor_pred[6,2]+= 1
            if(correct_preds==3): count_cor_pred[6,3]+= 1
            if(correct_preds==4): count_cor_pred[6,4]+= 1
            if(correct_preds==5): count_cor_pred[6,5]+= 1
            if(correct_preds>=6): count_cor_pred[6,6]+= 1
        else:
            if(correct_preds==1): count_cor_pred[actual_bsize,1]+= 1
            if(correct_preds==2): count_cor_pred[actual_bsize,2]+= 1
            if(correct_preds==3): count_cor_pred[actual_bsize,3]+= 1
            if(correct_preds==4): count_cor_pred[actual_bsize,4]+= 1
            if(correct_preds==5): count_cor_pred[actual_bsize,5]+= 1
        return float(correct_preds/actual_bsize), count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred

def recall_calculation(top_items, dataTest, item_list, output_path, output_dir):
    f1= open(output_path, "w")
    recall_test_main= 0.0
    #last basket for all users for testing 
    target_basket=[]
    time_baskets = dataTest.baskets.values
    for baskets in tqdm(time_baskets):
        target_basket_items= []
        
        for item in baskets:
                if item in item_list:
                    target_basket_items.append(item)
        target_basket.append(target_basket_items)
    #print("target basket: ", target_basket)
    userID_values = dataTest.userID.values
    target_user= []
    for userID in tqdm(userID_values):
        target_user.append(userID)
    #print("userID: ", target_user)
    last_semester_values = dataTest.last_semester.values
    target_semester= []
    for semester in tqdm(last_semester_values):
        target_semester.append(semester)

    #recall_calculation
    count= 0
    count_at_least_one_cor_pred = 0
    recall_test_for_one_cor_pred = 0.0
    count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred  = 0, 0, 0, 0, 0
    count_actual_bsize_at_least_2, count_actual_bsize_at_least_3, count_actual_bsize_at_least_4, count_actual_bsize_at_least_5, count_actual_bsize_at_least_6 = 0, 0, 0, 0, 0
    recall_temp =0.0
    count_cor_pred = {}
    for x5 in range(1,7):
        for y5 in range(1,7):
            count_cor_pred[x5,y5] = 0
    
    term_dict = {}
    #count_course = {}
    rec_info = []
    for i in range(len(top_items)):
        f1.write("UserID: ")
        f1.write(str(target_user[i])+ "| ")
        f1.write("target basket: "+ str(target_basket[i]))
        #print("target basket: "+ str(target_basket[i]))
        f1.write(", Recommended basket: ")
        for item4 in top_items[i]:
            f1.write(str(item4)+ " ")
        f1.write("\n") 
        t_length = len(target_basket[i])
        #calculate recall
        #if len(target_courses_CIS)>0:
        recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred = recall_cal(top_items[i], target_basket[i], count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred)
        #recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred = recall_cal(pred_courses_CIS, target_courses_CIS, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred)
        
        recall_test_main += recall_temp
        if t_length>=2: count_actual_bsize_at_least_2 += 1
        if t_length>=3: count_actual_bsize_at_least_3 += 1
        if t_length>=4: count_actual_bsize_at_least_4 += 1
        if t_length>=5: count_actual_bsize_at_least_5 += 1
        if t_length>=6: count_actual_bsize_at_least_6 += 1
        if recall_temp>0:  recall_test_for_one_cor_pred += recall_temp
        rel_rec = len((set(top_items[i]) & set(target_basket[i])))
        row = [t_length, target_basket[i], top_items[i], rel_rec, recall_temp, target_semester[i]]
        rec_info.append(row)
        if t_length>=6: target_basket_size[6] += 1 
        else: target_basket_size[t_length] += 1 
        count += 1
    recall_test = recall_test_main/ count
    #print("test recall: ", recall_test)
    f1.write("recall@n: "+ str(recall_test)+"\n")
    # percentage_of_at_least_one_cor_pred = (count_at_least_one_cor_pred/ len(top_items))* 100
    percentage_of_at_least_one_cor_pred = (count_at_least_one_cor_pred/ count)* 100
    print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred)
    f1.write("percentage_of_at_least_one_cor_pred: "+str(percentage_of_at_least_one_cor_pred))
    f1.close()
    return recall_test, percentage_of_at_least_one_cor_pred

if __name__ == '__main__':
    
    start = time.time()
    print("Start")
    #train_data = pd.read_json('./train_data_all.json', orient='records', lines= True)
    train_data, item_dict, user_dict, reversed_item_dict, reversed_user_dict = preprocess_train_data_part1(train_data)
    train_all, train_set_without_target, target, max_len = preprocess_train_data_part2(train_data) 
   
    valid_data = pd.read_json('./valid_sample_all.json', orient='records', lines= True)
    valid_data, user_dict2, reversed_user_dict2 = preprocess_valid_data_part1(valid_data, reversed_user_dict, item_dict)
    valid_all, valid_set_without_target, valid_target = preprocess_valid_data_part2(valid_data) #  #, 
    
    test_data = pd.read_json('./test_sample_all.json', orient='records', lines= True)
    test_data, user_dict3, reversed_user_dict3 = preprocess_test_data_part1(test_data, reversed_user_dict, item_dict, reversed_user_dict2)
    test_all, test_set_without_target, test_target = preprocess_test_data_part2(test_data) #, item_dict, user_dict, reversed_item_dict, reversed_user_dict #
    print("step 4 done")
    offered_courses = offered_course_cal('./all_data.json')

    print("Step 1 done")

    #creating the list of items
    item_list= list(item_dict.keys())
    rules = pd.read_json('./Association_rules/rules_v9_extended_15_40.json', orient="records", lines = True)
    #recommending top-k items for the last basket of each user
    match_thr = 50
    top_items = recommending_target_courses_valid_test(test_set_without_target, test_target, offered_courses, item_list, item_dict, reversed_item_dict, reversed_user_dict3, rules, match_thr)
    data_dir= './Association_rules/'
    output_dir = data_dir + "/output_dir"
    create_folder(output_dir)
    output_path= output_dir+ "/prediction_test_extended_prec_rules_v7_extended_15_40.txt"

    recall_score, percentage_of_at_least_one_cor_pred = recall_calculation(top_items, test_target, item_list, output_path, output_dir)
    print("test recall@n: ", recall_score)
    print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred)
    end2 = time.time()
    #print("time for recommendation for test data:", end2-end)


    #validating with valid data
    #recommending top-k items for the last basket of each user
    top_items2 = recommending_target_courses_valid_test(valid_set_without_target, valid_target, offered_courses, item_list, item_dict, reversed_item_dict, reversed_user_dict2, rules, match_thr)
    
    output_path= output_dir+ "/prediction_valid_extended_prec_rules_v7_extended_15_40.txt"
    #calculate recall for top-k predicted items
    recall_score, percentage_of_at_least_one_cor_pred = recall_calculation(top_items2, valid_target, item_list, output_path, output_dir)
    print("validation recall@n: ", recall_score)
    print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred)
    end3 = time.time()
    print("time for recommendation for validation data:", end3-end2)
   
    print("minimum support: ", 0.15)
    print("minimum confidence: ", 0.4)
    print("per match_thr: ", 50)
    print("recommendation from consequences only")
