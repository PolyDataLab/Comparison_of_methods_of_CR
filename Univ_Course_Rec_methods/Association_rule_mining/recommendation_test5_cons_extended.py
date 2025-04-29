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

def course_CIS_dept(basket):
    list_of_terms = ["CAP", "CDA", "CEN", "CGS", "CIS", "CNT", "COP", "COT", "CTS", "IDC","IDS"]
    basket1 = []
    for course in basket:
        flag = 0
        for term in list_of_terms:
            if course.find(term)!= -1:
                flag = 1
        if(flag==1):
            basket1.append(course)
    return basket1 

def recommending_target_courses_train(data_set_without_target, dataTest, offered_courses, item_list, item_dict, reversed_item_dict, reversed_user_dict, rules, match_thr):
    #top_k= 5
    #top_k= 5
    top_items_per_user= []
    # Making list of user baskets for each user
    time_baskets = data_set_without_target.baskets.values
    user_baskets_all= []
    prior_items_all = []
    for baskets in tqdm(time_baskets):
        baskets2 = []
        prior_items= []
        for basket in baskets:
            basket1 = []
            for item in basket:
                basket1.append(reversed_item_dict[item])
                if reversed_item_dict[item] not in prior_items:
                    prior_items.append(reversed_item_dict[item])
            baskets2.append(basket1)
        prior_items_all.append(prior_items)
        user_baskets_all.append(baskets2)

    target_basket=[]
    time_baskets = dataTest.baskets.values
    #userID_values = dataTest.userID.values
    for baskets in tqdm(time_baskets):
        target_basket_items= []
        #ann_target_basket_list1=[]
        #list2=[]
        for item in baskets:
                target_basket_items.append(reversed_item_dict[item])
        target_basket.append(target_basket_items)
    last_semester_values = dataTest.last_semester.values
    target_semesters= []
    for semester in tqdm(last_semester_values):
        target_semesters.append(semester)
    
    #print(target_basket)

    for i in range(len(data_set_without_target)):
        #index_ij_dict= {}
        user_baskets = user_baskets_all[i]
        #len_user_baskets = len(user_baskets)
        len_target_basket = len(target_basket[i])
        target_semester = target_semesters[i]
        prior_courses = prior_items_all[i]

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
            # lhs = list(rules['antecedents'][index3])
            # rhs = list(rules['consequents'][index3])
            # support = float(rules['support'][index3])
            # conf = float(rules['confidence'][index3])
            # lift = float(rules['lift'][index3])
            lhs = rules['lhs'][index3]
            rhs = rules['rhs'][index3]
            support = float(rules['sup'][index3])
            conf = float(rules['con'][index3])
            lift = float(rules['lift'][index3])
            n_match= len((set(prior_courses) & set(lhs)))
            per_match = (n_match/ len(lhs)) * 100
            if per_match>= match_thr:
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
            # lhs = list(rules['antecedents'][index3])
            # rhs = list(rules['consequents'][index3])
            # support = float(rules['support'][index3])
            # conf = float(rules['confidence'][index3])
            # lift = float(rules['lift'][index3])
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

def calculate_term_dict(term_dict, semester, basket, reversed_item_dict):
    for item in basket:
        if semester not in term_dict:
            count_course = {}
        else:
            count_course = term_dict[semester]
        if reversed_item_dict[item] not in count_course:
            count_course[reversed_item_dict[item]] = 1
        else:
            count_course[reversed_item_dict[item]] = count_course[reversed_item_dict[item]]+ 1
        term_dict[semester] = count_course
    return term_dict

def calculate_term_dict_2(term_dict, semester, basket):
    for item in basket:
        if semester not in term_dict:
            count_course = {}
        else:
            count_course = term_dict[semester]
        if item not in count_course:
            count_course[item] = 1
        else:
            count_course[item] = count_course[item] + 1
        #if semester==1221 and item=="COP4710": print("Count of course COP4710 in 1221 semester:", count_course[item])
        term_dict[semester] = count_course
    return term_dict

def calculate_term_dict_true(term_dict_true, semester, t_basket, pred_basket):
    for item in pred_basket:
        if item in t_basket:
            if semester not in term_dict_true:
                count_course = {}
            else:
                count_course = term_dict_true[semester]
            if item not in count_course:
                count_course[item] = 1
            else:
                count_course[item] = count_course[item]+ 1
            term_dict_true[semester] = count_course
    return term_dict_true

def calculate_term_dict_false(term_dict_false, semester, t_basket, pred_basket):
    for item in pred_basket:
        if item not in t_basket:
            if semester not in term_dict_false:
                count_course = {}
            else:
                count_course = term_dict_false[semester]
            if item not in count_course:
                count_course[item] = 1
            else:
                count_course[item] = count_course[item]+ 1
            term_dict_false[semester] = count_course
    return term_dict_false

def calculate_avg_n_actual_courses(input_data, reversed_item_dict):
    data = input_data
    frequency_of_courses = {}
    for baskets in data["baskets"]:
        for basket in baskets:
            for item in basket:
                if item not in frequency_of_courses:
                    frequency_of_courses[item] = 1
                else:
                    frequency_of_courses[item] += 1
    term_dict_all = {}
    for x in range(len(data)):
        baskets = data['baskets'][x]
        ts = data['timestamps'][x]
        #index1 =0 
        for x1 in range(len(ts)):
            basket = baskets[x1]
            semester = ts[x1]
            term_dict_all = calculate_term_dict_2(term_dict_all, semester, basket)
    count_course_all = {}
    for keys, values in term_dict_all.items():
        count_course = values
        for item, cnt in count_course.items():
            if item not in count_course_all:
                count_course_all[item] = [cnt, 1]
            else:
                # list1 = count_course_all[item]
                # list1[0] = list1[0]+ cnt
                # list1[1] = list1[0]+ 1
                cnt1, n1 = count_course_all[item]
                cnt1 += cnt
                n1 += 1
                #count_course_all[item] = list1
                count_course_all[item] = [cnt1, n1]
    count_course_avg = {}
    for course, n in count_course_all.items():
        #count_course_avg[course] = float(n[0]/n[1])
        cnt2, n2 = n
        count_course_avg[course] = float(cnt2/n2)
    #calculate standard deviation
    course_sd = {}
    for keys, values in term_dict_all.items():
        count_course = values
        for item, cnt in count_course.items():
            if item not in course_sd:
                course_sd[item] = [pow((cnt-count_course_avg[item]),2), 1]
            else:
                # list1 = count_course_all[item]
                # list1[0] = list1[0]+ cnt
                # list1[1] = list1[0]+ 1
                cnt1, n1 = course_sd[item]
                cnt1 = cnt1+ pow((cnt-count_course_avg[item]),2)
                n1 += 1
                #count_course_all[item] = list1
                course_sd[item] = [cnt1, n1]
    course_sd_main = {}
    course_number_terms = {}
    for course, n in course_sd.items():
        #count_course_avg[course] = float(n[0]/n[1])
        cnt2, n2 = n
        if(n2==1): course_sd_main[course] = float(math.sqrt(cnt2/n2))
        else: course_sd_main[course] = float(math.sqrt(cnt2/(n2-1)))
        course_number_terms[course] = n2
    
    return term_dict_all, frequency_of_courses, count_course_avg, course_sd_main, course_number_terms

def find_prior_term(course, prior_semester, term_dict_all_prior):
    flag = 0
    count_course_prior_2 = {}
    while(flag!=1):
        #print("prior_semester: ", prior_semester)
        if prior_semester in term_dict_all_prior:
            count_course_prior_2 = term_dict_all_prior[prior_semester]
        if course in count_course_prior_2:
            flag =1
        if prior_semester %5==0:
            prior_semester = prior_semester-4
        else:
            prior_semester = prior_semester-3
    return count_course_prior_2 

def calculate_std_dev(error_list):
    sum_err = 0.0
    for err in error_list:
        sum_err += err
    avg_err = sum_err/ len(error_list)
    sum_diff = 0.0
    for err in error_list:
        sum_diff += pow((err-avg_err), 2)
    std_dev = math.sqrt((sum_diff/len(error_list)))
    return avg_err, std_dev

def calculate_mse_for_course_allocation(term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, count_total_course, count_course_avg, course_sd_main, course_number_terms, term_dict_all_prior, output_dir):
    mse_for_course_allocation = 0.0
    mse_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mse_for_course_allocation_considering_not_predicted_courses = 0.0
    mae_for_course_allocation = 0.0
    mae_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mae_for_course_allocation_considering_not_predicted_courses = 0.0
    msse_for_course_allocation = 0.0
    msse_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_msse_for_course_allocation_considering_not_predicted_courses = 0.0
    mase_for_course_allocation = 0.0
    mase_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mase_for_course_allocation_considering_not_predicted_courses = 0.0
    #count1= 0
    count2 = 0
    output_path1= output_dir+ "/test_course_allocation_v2.txt"
    f = open(output_path1, "w") #generating text file with recommendation using filtering function
    course_allocation = []
    error_list = []
    ab_error_list = []
    st_error_list = []
    for keys in term_dict.keys():
        semester = keys
        count_course = term_dict[semester]
        # number of students in the previous offering
        if semester %5==0:
            prior_semester = semester-4
        else:
            prior_semester = semester-3

        if semester in term_dict_predicted:
            count_course_predicted = term_dict_predicted[semester]
            count_course_predicted_true = term_dict_predicted_true[semester]
            count_course_predicted_false = term_dict_predicted_false[semester]
            #print("Done 5")

            
            for item1 in count_course.keys():
                f.write("Semester: ")
                f.write(str(semester)+ " ")
                f.write("Course ID: ")
                f.write(str(item1)+ " ")
                count_course_prior = find_prior_term(item1, prior_semester, term_dict_all_prior)
                #print("Done 6")

                if item1 in count_course_predicted:
                    #mse_for_course_allocation += pow(((count_course[item1]/count_total_course[semester])-(count_course_predicted[item1]/count_total_course[semester])), 2)
                    mse_for_course_allocation += pow((count_course_predicted[item1]-count_course[item1]), 2)
                    mae_for_course_allocation += abs(count_course_predicted[item1]-count_course[item1])
                    msse_for_course_allocation += pow(abs((count_course_predicted[item1]-count_course[item1])/count_course[item1]), 2)
                    mase_for_course_allocation += abs((count_course_predicted[item1]-count_course[item1])/count_course[item1])
                    error_list.append(count_course_predicted[item1]-count_course[item1])
                    ab_error_list.append(abs(count_course_predicted[item1]-count_course[item1]))
                    st_error_list.append(abs((count_course_predicted[item1]-count_course[item1])/count_course[item1]))
                    f.write("actual: ")
                    f.write(str(count_course[item1])+ " ")
                    f.write("predicted: ")
                    f.write(str(count_course_predicted[item1]))
                    f.write("\n")
                    if item1 in count_course_predicted_true and item1 in count_course_predicted_false:
                        row = [semester, item1, count_course[item1], count_course_predicted[item1], count_course_predicted_true[item1], count_course_predicted_false[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                        #course_allocation.append(row)
                    elif item1 in count_course_predicted_true and item1 not in count_course_predicted_false:
                        row = [semester, item1, count_course[item1], count_course_predicted[item1], count_course_predicted_true[item1], 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                        #course_allocation.append(row)
                    elif item1 not in count_course_predicted_true and item1 in count_course_predicted_false:
                        row = [semester, item1, count_course[item1], count_course_predicted[item1], 0, count_course_predicted_false[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                        #course_allocation.append(row)
                    #count1 += 1
                    if item1 in count_course_prior:
                        row.append(count_course_prior[item1])
                    else:
                        row.append(0)
                    course_allocation.append(row)
                    count2 += 1
                else:
                    mse_for_course_allocation_2 += pow((count_course[item1]-0), 2)
                    mae_for_course_allocation_2 += abs((count_course[item1]-0))
                    msse_for_course_allocation_2 += pow(((count_course[item1]-0)/count_course[item1]), 2)
                    mase_for_course_allocation_2 += abs((count_course[item1]-0)/count_course[item1])
                    error_list.append(0-count_course[item1])
                    ab_error_list.append(abs(0-count_course[item1]))
                    st_error_list.append(abs((0-count_course[item1])/count_course[item1]))
                    
                    f.write("actual: ")
                    f.write(str(count_course[item1])+ " ")
                    f.write("predicted: ")
                    f.write(str(0))
                    f.write("\n")
                    if item1 in count_course_prior:
                        row = [semester, item1, count_course[item1], 0, 0, 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1], count_course_prior[item1]]
                    else:
                        row = [semester, item1, count_course[item1], 0, 0, 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1], 0]
                    course_allocation.append(row)
                    count2 += 1
    #print("Done 7")
    #avg_mse_for_course_allocation = mse_for_course_allocation/ count1
    avg_mse_for_course_allocation_considering_not_predicted_courses = (mse_for_course_allocation+ mse_for_course_allocation_2 )/ count2
    avg_mae_for_course_allocation_considering_not_predicted_courses = (mae_for_course_allocation+ mae_for_course_allocation_2 )/ count2
    avg_msse_for_course_allocation_considering_not_predicted_courses = (msse_for_course_allocation+ msse_for_course_allocation_2 )/ count2
    avg_mase_for_course_allocation_considering_not_predicted_courses = (mase_for_course_allocation+ mase_for_course_allocation_2 )/ count2
   
    f.close()
    course_allocation_actual_predicted = pd.DataFrame(course_allocation, columns=['Semester', 'Course_ID', 'actual_n', 'predicted_n', 'predicted_n_true', 'predicted_n_false', 'avg_n_actual', 'st_dev_actual', 'number_of_terms', 'n_sts_last_offering'])
    course_allocation_actual_predicted.to_csv(output_dir+'/course_allocation_actual_predicted_updated_new_v2.csv')
    return avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list


# #calculate recall
# def recall_cal(top_item_list, target_item_list, count_at_least_one_cor_pred):
#         t_length= len(target_item_list)
#         correct_preds= len((set(top_item_list) & set(target_item_list)))
#         actual_bsize= t_length
#         if(correct_preds>=1): count_at_least_one_cor_pred += 1
#         return (correct_preds/actual_bsize), count_at_least_one_cor_pred

# def recall_calculation(top_items, dataTest, item_list, output_path):
#     f1= open(output_path, "w")
#     recall_test_main= 0.0
#     #last basket for all users for testing 
#     target_basket=[]
#     time_baskets = dataTest.baskets.values
#     for baskets in tqdm(time_baskets):
#         target_basket_items= []
#         #ann_target_basket_list1=[]
#         #list2=[]
#         for basket in baskets:
#             for item in basket:
#                 if item in item_list:
#                     target_basket_items.append(item)
#         target_basket.append(target_basket_items)
#     #print("target basket: ", target_basket)
#     userID_values = dataTest.userID.values
#     target_user= []
#     for userID in tqdm(userID_values):
#         target_user.append(userID)
#     #print("userID: ", target_user)

#     #recall_calculation
#     count= 0
#     count_at_least_one_cor_pred = 0
#     recall_temp =0.0
#     for i in range(len(top_items)):
#         f1.write("UserID: ")
#         f1.write(str(target_user[i])+ "| ")
#         f1.write("target basket: "+ str(target_basket[i]))
#         #print("target basket: "+ str(target_basket[i]))
#         f1.write(", Recommended basket: ")
#         for item4 in top_items[i]:
#             f1.write(str(item4)+ " ")
#         f1.write("\n") 
#         if len(target_basket[i])>0:
#             recall_temp, count_at_least_one_cor_pred = recall_cal(top_items[i], target_basket[i], count_at_least_one_cor_pred)
#             recall_test_main += recall_temp
#             count += 1
#     recall_test = recall_test_main/ count
#     #print("test recall: ", recall_test)
#     f1.write("recall@n: "+ str(recall_test)+"\n")
#     percentage_of_at_least_one_cor_pred = count_at_least_one_cor_pred/ len(top_items)
#     f1.write("percentage_of_at_least_one_cor_pred: "+str(percentage_of_at_least_one_cor_pred))
#     f1.close()
#     return recall_test, percentage_of_at_least_one_cor_pred

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

def recall_calculation_train(top_items, dataTest, item_list, output_path, output_dir, reversed_item_dict):
    f1= open(output_path, "w")
    recall_test_main= 0.0
    #last basket for all users for testing 
    target_basket=[]
    time_baskets = dataTest.baskets.values
    for baskets in tqdm(time_baskets):
        target_basket_items= []
        #ann_target_basket_list1=[]
        #list2=[]
        for item in baskets:
                    target_basket_items.append(reversed_item_dict[item])
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
    target_basket_size = {}
    target_basket_size[1] = 0
    target_basket_size[2] = 0
    target_basket_size[3] = 0
    target_basket_size[4] = 0
    target_basket_size[5] = 0
    target_basket_size[6] = 0
    count_cor_pred = {}
    for x5 in range(1,7):
        for y5 in range(1,7):
            count_cor_pred[x5,y5] = 0
    
    term_dict = {}
    #count_course = {}
    term_dict_predicted = {}
    term_dict_predicted_true = {}
    term_dict_predicted_false = {}

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
        # target_courses= target_basket[i]
        # target_courses_CIS = course_CIS_dept(target_courses)
        # pred_courses = top_items[i]
        # pred_courses_CIS = course_CIS_dept(pred_courses)
        #calculate recall
        #if len(target_courses_CIS)>0:
        recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred = recall_cal(top_items[i], target_basket[i], count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred)
        #recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred = recall_cal(pred_courses_CIS, target_courses_CIS, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred)
        term_dict = calculate_term_dict_2(term_dict, target_semester[i], target_basket[i])
        #course allocation for predicted courses
        term_dict_predicted = calculate_term_dict_2(term_dict_predicted, target_semester[i], top_items[i])
        term_dict_predicted_true = calculate_term_dict_true(term_dict_predicted_true, target_semester[i], target_basket[i], top_items[i])
        term_dict_predicted_false = calculate_term_dict_false(term_dict_predicted_false, target_semester[i], target_basket[i], top_items[i])
        recall_test_main += recall_temp
        if t_length>=2: count_actual_bsize_at_least_2 += 1
        if t_length>=3: count_actual_bsize_at_least_3 += 1
        if t_length>=4: count_actual_bsize_at_least_4 += 1
        if t_length>=5: count_actual_bsize_at_least_5 += 1
        if t_length>=6: count_actual_bsize_at_least_6 += 1
        if recall_temp>0:  recall_test_for_one_cor_pred += recall_temp
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

    percentage_of_at_least_two_cor_pred = (count_at_least_two_cor_pred/ count_actual_bsize_at_least_2) *100
    print("percentage_of_at_least_two_cor_pred: ", percentage_of_at_least_two_cor_pred)
    f1.write("percentage_of_at_least_two_cor_pred: "+ str(percentage_of_at_least_two_cor_pred)+ "\n")
    percentage_of_at_least_three_cor_pred = (count_at_least_three_cor_pred/ count_actual_bsize_at_least_3) *100
    print("percentage_of_at_least_three_cor_pred: ", percentage_of_at_least_three_cor_pred)
    f1.write("percentage_of_at_least_three_cor_pred: "+ str(percentage_of_at_least_three_cor_pred)+ "\n")
    percentage_of_at_least_four_cor_pred = (count_at_least_four_cor_pred/ count_actual_bsize_at_least_4) * 100
    print("percentage_of_at_least_four_cor_pred: ", percentage_of_at_least_four_cor_pred)
    f1.write("percentage_of_at_least_four_cor_pred: "+ str(percentage_of_at_least_four_cor_pred)+ "\n")
    percentage_of_at_least_five_cor_pred = (count_at_least_five_cor_pred/ count_actual_bsize_at_least_5) *100
    print("percentage_of_at_least_five_cor_pred: ", percentage_of_at_least_five_cor_pred)
    f1.write("percentage_of_at_least_five_cor_pred: "+ str(percentage_of_at_least_five_cor_pred)+ "\n")
    percentage_of_all_cor_pred = (count_all_cor_pred/ len(top_items)) *100
    print("percentage_of_all_cor_pred: ", percentage_of_all_cor_pred)
    f1.write("percentage_of_all_cor_pred: "+ str(percentage_of_all_cor_pred)+ "\n")
    #calculate Recall@n for whom we generated at least one correct prediction in test data
    test_recall_for_one_cor_pred = recall_test_for_one_cor_pred/ count_at_least_one_cor_pred
    print("Recall@n for whom we generated at least one correct prediction in test data: ", test_recall_for_one_cor_pred)
    f1.write("Recall@n for whom we generated at least one correct prediction in test data:"+ str(test_recall_for_one_cor_pred))
    f1.write("\n") 
    for x6 in range(1,7):
        percentage_of_one_cor_pred = (count_cor_pred[x6,1]/ target_basket_size[x6]) *100
        print("percentage of_one cor pred for target basket size {}: {}".format(x6, percentage_of_one_cor_pred))
        percentage_of_two_cor_pred = (count_cor_pred[x6,2]/ target_basket_size[x6]) *100
        print("percentage of_two cor pred for target basket size {}: {}".format(x6, percentage_of_two_cor_pred))
        percentage_of_three_cor_pred = (count_cor_pred[x6,3]/ target_basket_size[x6]) *100
        print("percentage of_three cor pred for target basket size {}: {}".format(x6, percentage_of_three_cor_pred))
        percentage_of_four_cor_pred = (count_cor_pred[x6,4]/ target_basket_size[x6]) *100
        print("percentage of_four cor pred for target basket size {}: {}".format(x6, percentage_of_four_cor_pred))
        percentage_of_five_cor_pred = (count_cor_pred[x6,5]/ target_basket_size[x6]) *100
        print("percentage of_five cor pred for target basket size {}: {}".format(x6, percentage_of_five_cor_pred))
        percentage_of_at_least_six_cor_pred = (count_cor_pred[x6,6]/ target_basket_size[x6]) *100
        print("percentage of_at_least_six cor pred for target basket size {}: {}".format(x6, percentage_of_at_least_six_cor_pred))

    for x7 in range(1,7):
        print("total count of target basket size of {}: {}".format(x7, target_basket_size[x7]))

    for x6 in range(1,7):
        print("one cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,1]))
        print("two cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,2]))
        print("three cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,3]))
        print("four cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,4]))
        print("five cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,5]))
        print("six or more cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,6]))
    f1.close()
    return recall_test, percentage_of_at_least_one_cor_pred

def recall_calculation(top_items, dataTest, item_list, output_path, output_dir):
    f1= open(output_path, "w")
    recall_test_main= 0.0
    #last basket for all users for testing 
    target_basket=[]
    time_baskets = dataTest.baskets.values
    for baskets in tqdm(time_baskets):
        target_basket_items= []
        #ann_target_basket_list1=[]
        #list2=[]
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
    target_basket_size = {}
    target_basket_size[1] = 0
    target_basket_size[2] = 0
    target_basket_size[3] = 0
    target_basket_size[4] = 0
    target_basket_size[5] = 0
    target_basket_size[6] = 0
    count_cor_pred = {}
    for x5 in range(1,7):
        for y5 in range(1,7):
            count_cor_pred[x5,y5] = 0
    
    term_dict = {}
    #count_course = {}
    term_dict_predicted = {}
    term_dict_predicted_true = {}
    term_dict_predicted_false = {}
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
        # target_courses= target_basket[i]
        # target_courses_CIS = course_CIS_dept(target_courses)
        # pred_courses = top_items[i]
        # pred_courses_CIS = course_CIS_dept(pred_courses)
        #calculate recall
        #if len(target_courses_CIS)>0:
        recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred = recall_cal(top_items[i], target_basket[i], count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred)
        #recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred = recall_cal(pred_courses_CIS, target_courses_CIS, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred)
        term_dict = calculate_term_dict_2(term_dict, target_semester[i], target_basket[i])
        #course allocation for predicted courses
        term_dict_predicted = calculate_term_dict_2(term_dict_predicted, target_semester[i], top_items[i])
        term_dict_predicted_true = calculate_term_dict_true(term_dict_predicted_true, target_semester[i], target_basket[i], top_items[i])
        term_dict_predicted_false = calculate_term_dict_false(term_dict_predicted_false, target_semester[i], target_basket[i], top_items[i])
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

    percentage_of_at_least_two_cor_pred = (count_at_least_two_cor_pred/ count_actual_bsize_at_least_2) *100
    print("percentage_of_at_least_two_cor_pred: ", percentage_of_at_least_two_cor_pred)
    f1.write("percentage_of_at_least_two_cor_pred: "+ str(percentage_of_at_least_two_cor_pred)+ "\n")
    percentage_of_at_least_three_cor_pred = (count_at_least_three_cor_pred/ count_actual_bsize_at_least_3) *100
    print("percentage_of_at_least_three_cor_pred: ", percentage_of_at_least_three_cor_pred)
    f1.write("percentage_of_at_least_three_cor_pred: "+ str(percentage_of_at_least_three_cor_pred)+ "\n")
    percentage_of_at_least_four_cor_pred = (count_at_least_four_cor_pred/ count_actual_bsize_at_least_4) * 100
    print("percentage_of_at_least_four_cor_pred: ", percentage_of_at_least_four_cor_pred)
    f1.write("percentage_of_at_least_four_cor_pred: "+ str(percentage_of_at_least_four_cor_pred)+ "\n")
    percentage_of_at_least_five_cor_pred = (count_at_least_five_cor_pred/ count_actual_bsize_at_least_5) *100
    print("percentage_of_at_least_five_cor_pred: ", percentage_of_at_least_five_cor_pred)
    f1.write("percentage_of_at_least_five_cor_pred: "+ str(percentage_of_at_least_five_cor_pred)+ "\n")
    percentage_of_all_cor_pred = (count_all_cor_pred/ len(top_items)) *100
    print("percentage_of_all_cor_pred: ", percentage_of_all_cor_pred)
    f1.write("percentage_of_all_cor_pred: "+ str(percentage_of_all_cor_pred)+ "\n")
    #calculate Recall@n for whom we generated at least one correct prediction in test data
    test_recall_for_one_cor_pred = recall_test_for_one_cor_pred/ count_at_least_one_cor_pred
    print("Recall@n for whom we generated at least one correct prediction in test data: ", test_recall_for_one_cor_pred)
    f1.write("Recall@n for whom we generated at least one correct prediction in test data:"+ str(test_recall_for_one_cor_pred))
    f1.write("\n") 
    for x6 in range(1,7):
        percentage_of_one_cor_pred = (count_cor_pred[x6,1]/ target_basket_size[x6]) *100
        print("percentage of_one cor pred for target basket size {}: {}".format(x6, percentage_of_one_cor_pred))
        percentage_of_two_cor_pred = (count_cor_pred[x6,2]/ target_basket_size[x6]) *100
        print("percentage of_two cor pred for target basket size {}: {}".format(x6, percentage_of_two_cor_pred))
        percentage_of_three_cor_pred = (count_cor_pred[x6,3]/ target_basket_size[x6]) *100
        print("percentage of_three cor pred for target basket size {}: {}".format(x6, percentage_of_three_cor_pred))
        percentage_of_four_cor_pred = (count_cor_pred[x6,4]/ target_basket_size[x6]) *100
        print("percentage of_four cor pred for target basket size {}: {}".format(x6, percentage_of_four_cor_pred))
        percentage_of_five_cor_pred = (count_cor_pred[x6,5]/ target_basket_size[x6]) *100
        print("percentage of_five cor pred for target basket size {}: {}".format(x6, percentage_of_five_cor_pred))
        percentage_of_at_least_six_cor_pred = (count_cor_pred[x6,6]/ target_basket_size[x6]) *100
        print("percentage of_at_least_six cor pred for target basket size {}: {}".format(x6, percentage_of_at_least_six_cor_pred))

    for x7 in range(1,7):
        print("total count of target basket size of {}: {}".format(x7, target_basket_size[x7]))

    for x6 in range(1,7):
        print("one cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,1]))
        print("two cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,2]))
        print("three cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,3]))
        print("four cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,4]))
        print("five cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,5]))
        print("six or more cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,6]))
    test_rec_info = pd.DataFrame(rec_info, columns=['bsize', 'target_courses', 'rec_courses', 'n_rel_rec', 'recall_score', 'target_semester'])
    test_rec_info.to_json('/Users/mkhan149/Downloads/Experiments/Others/Association_rules/ARM_test_rec_info_without_summer.json', orient='records', lines=True)
    test_rec_info.to_csv('/Users/mkhan149/Downloads/Experiments/Others/Association_rules/ARM_test_rec_info_without_summer.csv')
    # count_total_course = {}
    # for keys, values in term_dict.items():
    #     count_course_dict = values
    #     count_course_dict = dict(sorted(count_course_dict.items(), key=lambda item: item[1], reverse= True))
    #     count3 = 0
    #     for cnt in count_course_dict.values():
    #         count3 += cnt
    #     count_total_course[keys] = count3
    #     term_dict[keys] = count_course_dict
    # #print(term_dict)
    # #sorting the courses in term dictionary based on number of occurences of courses in descending order
    # for keys, values in term_dict_predicted.items():
    #     count_course_dict = values
    #     count_course_dict = dict(sorted(count_course_dict.items(), key=lambda item: item[1], reverse= True))
    #     term_dict_predicted[keys] = count_course_dict
    # for keys, values in term_dict_predicted_true.items():
    #     count_course_dict = values
    #     count_course_dict = dict(sorted(count_course_dict.items(), key=lambda item: item[1], reverse= True))
    #     term_dict_predicted_true[keys] = count_course_dict
    
    # for keys, values in term_dict_predicted_false.items():
    #     count_course_dict = values
    #     count_course_dict = dict(sorted(count_course_dict.items(), key=lambda item: item[1], reverse= True))
    #     term_dict_predicted_false[keys] = count_course_dict
    # all_data_en_pred = pd.read_json('/Users/mkhan149/Downloads/Experiments/all_data_en_pred_filtered.json', orient='records', lines= True)
    # term_dict_all, frequency_of_courses, count_course_avg, course_sd_main, course_number_terms = calculate_avg_n_actual_courses(all_data_en_pred, reversed_item_dict)

    # # valid_data_unique = pd.read_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/valid_sample_filtered_unique.json', orient='records', lines= True)
    # # term_dict_valid, frequency_of_courses2, count_course_avg2, course_sd_main2, course_number_terms2 = calculate_avg_n_actual_courses(valid_data_unique, reversed_item_dict)

    # # avg_mse_for_course_allocation, avg_mse_for_course_allocation_considering_not_predicted_courses = calculate_mse_for_course_allocation(term_dict, term_dict_predicted)
    # # avg_rmse_for_course_allocation, avg_rmse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_mse_for_course_allocation), math.sqrt(avg_mse_for_course_allocation_considering_not_predicted_courses)
    # avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list = calculate_mse_for_course_allocation(term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, count_total_course, count_course_avg, course_sd_main, course_number_terms, term_dict_all, output_dir)
    # print("Done4")
    # avg_rmse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_mse_for_course_allocation_considering_not_predicted_courses)
    # avg_rmsse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_msse_for_course_allocation_considering_not_predicted_courses)
    # mean_error, std_dev_error = calculate_std_dev(error_list)
    # mean_ab_error, std_dev_ab_error = calculate_std_dev(ab_error_list)
    # mean_st_error, std_dev_st_error = calculate_std_dev(st_error_list)

    # #print("avg mse for # of allocated course where we are predicting a course at least once: ",avg_mse_for_course_allocation)
    # #print("avg_mse_for_course_allocation_considering all courses available in test data: ",avg_mse_for_course_allocation_considering_not_predicted_courses)
    # #print("avg rmse for # of allocated course where we are predicting a course at least once: ",avg_rmse_for_course_allocation)
    # print("avg_mae_for_course_allocation_considering all courses available in test data: ",avg_mae_for_course_allocation_considering_not_predicted_courses)
    # print("avg_rmse_for_course_allocation_considering all courses available in test data: ",avg_rmse_for_course_allocation_considering_not_predicted_courses)
    # print("avg_mase_for_course_allocation_considering all courses available in test data: ",avg_mase_for_course_allocation_considering_not_predicted_courses)
    # print("avg_rmsse_for_course_allocation_considering all courses available in test data: ",avg_rmsse_for_course_allocation_considering_not_predicted_courses)
    # print("mean of errors: ", mean_error)
    # print("standard_deviation for errors: ", std_dev_error)
    # print("mean of absolute errors: ", mean_ab_error)
    # print("standard_deviation for absolute errors: ", std_dev_ab_error)
    # print("mean of standardized errors: ", mean_st_error)
    # print("standard_deviation for standardized errors: ", std_dev_st_error)
    f1.close()
    return recall_test, percentage_of_at_least_one_cor_pred

if __name__ == '__main__':
    #train_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/train_data_all.json', orient='records', lines= True)
    start = time.time()
    print("Start")
    #train_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/train_data_all_CR.json', orient='records', lines= True)
    train_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/train_data_all_without_summer.json', orient='records', lines= True)
    train_data, item_dict, user_dict, reversed_item_dict, reversed_user_dict = preprocess_train_data_part1(train_data)
    train_all, train_set_without_target, target, max_len = preprocess_train_data_part2(train_data) 
    #print(len(item_dict))
    #    print(train_all)
    #    print("max_len:", max_len)
    #print(target)
    #valid_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/valid_data_all.json', orient='records', lines= True)
    # valid_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/valid_sample_all_CR.json', orient='records', lines= True)
    valid_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/valid_sample_all_without_summer.json', orient='records', lines= True)
    valid_data, user_dict2, reversed_user_dict2 = preprocess_valid_data_part1(valid_data, reversed_user_dict, item_dict)
    valid_all, valid_set_without_target, valid_target = preprocess_valid_data_part2(valid_data) #  #, 
    #print("reversed_user_dict2: ", reversed_user_dict2)
    #print(valid_all)
    # test_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/test_sample_all_CR.json', orient='records', lines= True)
    test_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/test_sample_all_without_summer.json', orient='records', lines= True)
    test_data, user_dict3, reversed_user_dict3 = preprocess_test_data_part1(test_data, reversed_user_dict, item_dict, reversed_user_dict2)
    test_all, test_set_without_target, test_target = preprocess_test_data_part2(test_data) #, item_dict, user_dict, reversed_item_dict, reversed_user_dict #
    print("step 4 done")
    # offered_courses = offered_course_cal('/Users/mkhan149/Downloads/Experiments/all_data_CR.json')
    offered_courses = offered_course_cal('/Users/mkhan149/Downloads/Experiments/all_data_without_summer.csv')

    print("Step 1 done")

    #creating the list of items
    item_list= list(item_dict.keys())
    # rules = pd.read_csv('/Users/mkhan149/Downloads/Experiments/Others/Association_rules/rules_apriori_v2_sup_0.2.csv')
    # rules = pd.read_json('/Users/mkhan149/Downloads/Experiments/Others/Association_rules/rules_v7_extended.json', orient="records", lines = True)
    # rules = pd.read_json('/Users/mkhan149/Downloads/Experiments/Others/Association_rules/rules_v7_extended_15_40.json', orient="records", lines = True)
    rules = pd.read_json('/Users/mkhan149/Downloads/Experiments/Others/Association_rules/rules_v9_extended_15_40_without_summer.json', orient="records", lines = True)
    #rules_v9_extended_15_40_without_summer
    #recommending top-k items for the last basket of each user
    # top_items = recommending_target_courses(A1, R1, 5, u1, k, offered_courses, train_basket, dataTrain, item_list, dataTest, item_dict_all, reversed_item_dict)
    match_thr = 50
    top_items = recommending_target_courses_valid_test(test_set_without_target, test_target, offered_courses, item_list, item_dict, reversed_item_dict, reversed_user_dict3, rules, match_thr)
    #top_items = recommending_top_k(A1, R1, 5, k, offered_courses, train_basket, dataTrain, item_list)
    # for i in top_items:
    #     for x in i:
    #         print(x," ")
    #     print("\n")
    data_dir= '/Users/mkhan149/Downloads/Experiments/Others/Association_rules/'
    output_dir = data_dir + "/output_dir"
    create_folder(output_dir)
    output_path= output_dir+ "/prediction_test_extended_prec_rules_v7_extended_15_40_without_summer.txt"

    recall_score, percentage_of_at_least_one_cor_pred = recall_calculation(top_items, test_target, item_list, output_path, output_dir)
    print("test recall@n: ", recall_score)
    print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred)
    end2 = time.time()
    #print("time for recommendation for test data:", end2-end)


    #valid_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/valid_data_all.json', orient='records', lines= True)
    # dataValid_prev, dataValid_target, dataValid_Total = preprocess_valid_data(valid_data, item_list5)

    #validating with valid data
    #recommending top-k items for the last basket of each user
    top_items2 = recommending_target_courses_valid_test(valid_set_without_target, valid_target, offered_courses, item_list, item_dict, reversed_item_dict, reversed_user_dict2, rules, match_thr)
    #top_items = recommending_top_k(A1, R1, 5, k, offered_courses, train_basket, dataTrain, item_list)
    # for i in top_items2:
    #     for x in i:
    #         print(x," ")
    #     print("\n")
    output_path= output_dir+ "/prediction_valid_extended_prec_rules_v7_extended_15_40_without_summer.txt"
    #calculate recall for top-k predicted items
    recall_score, percentage_of_at_least_one_cor_pred = recall_calculation(top_items2, valid_target, item_list, output_path, output_dir)
    print("validation recall@n: ", recall_score)
    print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred)
    end3 = time.time()
    print("time for recommendation for validation data:", end3-end2)
   
    # # # test_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/test_data_all.json', orient='records', lines= True)
    # # # dataTest_prev, dataTest_target, datatest_Total = preprocess_test_data(test_data, item_list5)

    # # #recommending top-k items for the last basket of each user
    # top_items3 = recommending_target_courses_train(train_set_without_target, target, offered_courses, item_list, item_dict, reversed_item_dict, reversed_user_dict, rules, match_thr)
    # #top_items = recommending_top_k(A1, R1, 5, k, offered_courses, train_basket, dataTrain, item_list)
    # # for i in top_items2:
    # #     for x in i:
    # #         print(x," ")
    # #     print("\n")
    # output_path= output_dir+ "/prediction_train_extended_prec_rules_v7_extended_15_40.txt"
    # #calculate recall for top-k predicted items
    # recall_score, percentage_of_at_least_one_cor_pred = recall_calculation_train(top_items3, target, item_list, output_path, output_dir, reversed_item_dict)
    # print("train recall@n: ", recall_score)
    # #print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred)
    # end4 = time.time()
    # print("time for recommendation for training data:", end4-end3)
    
    print("minimum support: ", 0.15)
    print("minimum confidence: ", 0.4)
    print("per match_thr: ", 50)
    print("recommendation from consequences only")