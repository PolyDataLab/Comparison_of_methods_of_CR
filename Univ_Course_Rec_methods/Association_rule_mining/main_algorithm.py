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

#This function counts the number of apperances of a course in a particular semester and store them in the dictionary
#Semester_dict consists of all possible semester numbers as keys and a dictionary of count of courses as values
def calculate_dict(semester_dict, index2, basket):
    if index2 in semester_dict:
        count_item= semester_dict[index2]
    else:
        count_item = {}
    for item2 in basket:
        count_item[item2]= count_item.get(item2, 0)+ 1
    semester_dict[index2] = count_item
    return semester_dict

#training process to measure popular courses
#it returns a dictionary of count of courses

#recommend top-k courses based on highest score of courses
def recommend_top_k(main_dict, ts, user_baskets, offered_course_list, top_k, item_list):
     top_k1= 0
     #print(prob_dict)
     top_items= []
     for keys, values in main_dict.items():
            if(keys==ts):
                count_dict= values
                top_k1=0
                for item in count_dict.keys():
                    if not filtering(item, user_baskets, offered_course_list, item_list):
                        top_items.append(item)
                        top_k1+=1
                        if(top_k1==top_k): break
     return top_items

#calculate recall 
def recall_cal(top_item_list, target_item_list, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred):
    t_length= len(target_item_list)
    correct_preds= len((set(top_item_list) & set(target_item_list)))
    #print(correct_preds)
    actual_bsize= t_length
    if correct_preds>=1: count_at_least_one_cor_pred+= 1
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
# def course_CIS_dept(basket):
#     list_of_terms = ["CAP", "CDA", "CEN", "CGS", "CIS", "CNT", "COP", "COT", "CTS", "IDC","IDS"]
#     basket1 = []
#     for course in basket:
#         flag = 0
#         for term in list_of_terms:
#             if course.find(term)!= -1:
#                 flag = 1
#         if(flag==1):
#             basket1.append(course)
#     return basket1  

#validating the model 
def validate(main_dict, valid, valid_target, offered_courses, item_list):
    count1= 0
    recall_validation = 0.0
    target_basket= []
    count_at_least_one_cor_pred = 0
    count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred  = 0, 0, 0, 0, 0
    recall_temp =0
    count_cor_pred = {}
    for x5 in range(1,7):
        for y5 in range(1,7):
            count_cor_pred[x5,y5] = 0
    for basket in valid_target['baskets']:
        target_basket.append(basket)
    #offered courses in each semester for validation data 
    #offered_courses = calculate_offered_courses(valid_all)
    index = 0
    for user in valid.userID.values:
        #index = valid[valid['userID'] == user].index.values[0]
        len_baskets= valid.iloc[index]['num_baskets']
        user_baskets = valid.iloc[index]['baskets']
        top_k = len(target_basket[index])
        target_semester =  valid.iloc[index]['last_semester']
        pred_basket= recommend_top_k(main_dict, len_baskets+1, user_baskets, offered_courses[target_semester], top_k, item_list)
        #CIS dept
        # target_courses= target_basket[index]
        # target_courses_CIS = course_CIS_dept(target_courses)
        # pred_courses = pred_basket
        # pred_courses_CIS = course_CIS_dept(pred_courses)
        # #calculate recall
        # if len(target_courses_CIS)>0:
        recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred = recall_cal(pred_basket, target_basket[index], count_at_least_one_cor_pred,  count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred) 
        #recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred = recall_cal(pred_courses_CIS, target_courses_CIS, count_at_least_one_cor_pred,  count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred) 
        recall_validation += recall_temp
        #recall_validation += recall_cal(pred_basket, target_basket[index], count_at_least_one_cor_pred)
        count1 += 1
        index +=1
    
    valid_recall = recall_validation/ count1
    print("Recall@n for validation: ", valid_recall)

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
                count_course[item] = count_course[item] + 1
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
# def course_allocation(data, reversed_item_dict):
#     term_dict = {}
#     count_course = {}
#     for x in range(len(data)):
#         semester = data['last_semester'][x]
#         #if semester not in term_dict:
#         term_dict, count_course = calculate_term_dict(term_dict, semester, count_course, data['baskets'][x], reversed_item_dict)
#         # else:
#         #     term_dict[semester], count_course = calculate_term_dict(term_dict, semester, count_course, data['baskets'][x], reversed_item_dict)
#     return term_dict
def calculate_avg_n_actual_courses(input_data):
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

def cluster_based_on_grades(data, item_dict):
    list_of_grades = []
    item_list = list(item_dict.values())
    list_of_grades_2 = []
    for x in range(len(data)):
        list_of_grades_3 = []
        for y in range(len(item_list)):
            list_of_grades_3.append(0)
        list_of_grades.append(list_of_grades_3)
    index2= 0
    for baskets in data['baskets']:
        list_of_grades_2 = list_of_grades[index2] 
        list_of_grades_main = data['grades'][index2]
        index3= 0
        for basket in baskets:
            list_grade_basket = list_of_grades_main[index3]
            index4 = 0
            for item in basket:
                grade = list_grade_basket[index4]
                if grade!= 'P':
                    list_of_grades_2[item]= float(grade)
                index4 += 1
            index3 += 1
        list_of_grades[index2] = list_of_grades_2
        index2 += 1

    # for grades in data['grades']:
    #     grades_list_new = []
    #     for grade_basket in grades:
    #         for grade in grade_basket:
    #             if grade!= 'P' and grade!='F':
    #                 grades_list_new.append(float(grade))
    #     cnt += 1 
    #     list_of_grades.append(grades_list_new)
        #if cnt==2: break
    #print(list_of_grades)
    X = np.array(list_of_grades)
    kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(X)
    print(kmeans.labels_)
    print ("step 5 done")
    #kmeans.predict([[2, 4, 3], [2, 3, 3]])
    return kmeans.labels_

# def generate_rules(data, label_of_students, reversed_item_dict):
#     #records = []
#     records_1 = []
#     records_0 = []
#     records_2 = []
#     records_3 = []
#     records_4 =[]
#     for x in range(len(data)):
#         baskets = data['baskets'][x]
#         list_of_items = []
#         for basket in baskets:
#             for item in basket:
#                 list_of_items.append(reversed_item_dict[item])
#         if label_of_students[x]==0:
#             records_0.append(list_of_items)  #clustered group 0
#         elif label_of_students[x]==1:
#             records_1.append(list_of_items)  #clustered group 1
#         elif label_of_students[x]==2:
#             records_2.append(list_of_items)  #clustered group 2
#         elif label_of_students[x]==3:
#             records_3.append(list_of_items)  #clustered group 3
#         elif label_of_students[x]==4:
#             records_4.append(list_of_items)  #clustered group 4
#     records_all = [records_0, records_1, records_2, records_3, records_4]
#     # item_list = list(reversed_item_dict.keys())
#     # list_of_courses_2 = []
#     # list_of_courses = []
#     # for x in range(len(data)):
#     #     list_of_courses_3 = []
#     #     for y in range(len(item_list)):
#     #         list_of_courses_3.append(0)
#     #     list_of_courses.append(list_of_courses_3)
#     # index2= 0
#     # for baskets in data['baskets']:
#     #     list_of_courses_2 = list_of_courses[index2] 
#     #     index3= 0
#     #     for basket in baskets:
#     #         # list_grade_basket = list_of_grades_main[index3]
#     #         index4 = 0
#     #         for item in basket:
#     #             # grade = list_grade_basket[index4]
#     #             list_of_courses_2[item]= 1
#     #             index4 += 1
#     #         index3 += 1
#     #     list_of_courses[index2] = list_of_courses_2
#     #     index2 += 1
#     # for i in range(len(label_of_students)):
#     #     if label_of_students[i]==1:
#     #         records.append(list_of_courses[i])
    
#     # records2 = np.array(records)
    
#     # Building the model
#     # frq_items = apriori(records2, min_support = 0.05, use_colnames = True)
    
#     # # Collecting the inferred rules in a dataframe
#     # rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
#     # rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
#     # print(rules.head())
#     rules_all = []
#     cnt = 10
#     for records in records_all:
#         rules = apriori(records, min_support = 0.05, min_confidence = 0.8, min_lift = 3, min_length = 10)
#         rules_all.append(rules)
#         print("step number "+ str(cnt)+ "done")
#         cnt+= 1
#     # results = list(rules)
#     # results = pd.DataFrame(results)
#     #print(results.head(5))
#     list_rules_all = []
#     for rules in rules_all: 
#         list_rules = []
#         for item in rules:
#             pair = item[0]
#             items = [x for x in pair]
#             print("Rule: " + items[0] + "-->" + items[1])
#             print("support: ", str(item[1]))
#             print("confidence: ", str(item[2][0][2]))
#             print("lift: ", str(item[2][0][3]))
#             row = [items[0], items[1], item[1], item[2][0][2], item[2][0][3]]
#             list_rules.append(row)
#             print("==========")
#         list_rules_all.append(list_rules)
#     index2 =0
#     list_rules_cls = []
#     for rules_2 in list_rules_all:
#         for rules in rules_2:
#             v, w, x, y, z = rules
#             row = [index2, v, w, x, y, z]
#             list_rules_cls.append(row)
#         index2 += 1

#     list_rules_df = pd.DataFrame(list_rules_cls, columns=['cluster_no', 'lhs', 'rhs', 'sup', 'con', 'lift'])
#     list_rules_df.to_csv('/Users/mkhan149/Downloads/Experiments/others/Association_rules/rules.csv') 
    
#     return list_rules_all 
def generate_rules(data, reversed_item_dict):
    #records = []
    # records_1 = []
    # records_0 = []
    # records_2 = []
    # records_3 = []
    # records_4 =[]
    list_of_instances = []
    for x in range(len(data)):
        baskets = data['baskets'][x]
        list_of_items = []
        for basket in baskets:
            for item in basket:
                list_of_items.append(reversed_item_dict[item])
        list_of_instances.append(list_of_items)
    #records_all = [records_0, records_1, records_2, records_3, records_4]
    # item_list = list(reversed_item_dict.keys())
    # list_of_courses_2 = []
    # list_of_courses = []
    # for x in range(len(data)):
    #     list_of_courses_3 = []
    #     for y in range(len(item_list)):
    #         list_of_courses_3.append(0)
    #     list_of_courses.append(list_of_courses_3)
    # index2= 0
    # for baskets in data['baskets']:
    #     list_of_courses_2 = list_of_courses[index2] 
    #     index3= 0
    #     for basket in baskets:
    #         # list_grade_basket = list_of_grades_main[index3]
    #         index4 = 0
    #         for item in basket:
    #             # grade = list_grade_basket[index4]
    #             list_of_courses_2[item]= 1
    #             index4 += 1
    #         index3 += 1
    #     list_of_courses[index2] = list_of_courses_2
    #     index2 += 1
    # for i in range(len(label_of_students)):
    #     if label_of_students[i]==1:
    #         records.append(list_of_courses[i])
    
    # records2 = np.array(records)
    
    # Building the model
    # frq_items = apriori(records2, min_support = 0.05, use_colnames = True)
    
    # # Collecting the inferred rules in a dataframe
    # rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
    # rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
    # print(rules.head())
    #rules_all = []
    #cnt = 10
    transactions = list_of_instances.copy()
    rules = apriori(transactions, min_support = 0.15, min_confidence = 0.4, min_lift = 1, min_length = 1)
    min_support1 = 0.15
    min_confidence1 = 0.4
    print("minimum support: ", min_support1)
    print("minimum confidence: ", min_confidence1)
    #print(rules)
    #rules = list(apriori(list_of_instances))
    # results = list(rules)
    # results = pd.DataFrame(results)
    #print(results.head(5))
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
    # index2 =0
    # list_rules_cls = []
    # for rules_2 in list_rules_all:
    #     for rules in rules_2:
    #         v, w, x, y, z = rules
    #         row = [index2, v, w, x, y, z]
    #         list_rules_cls.append(row)
    #     index2 += 1

    list_rules_df = pd.DataFrame(list_rules_new, columns=['lhs', 'rhs', 'sup', 'con', 'lift'])
    #list_rules_df = pd.DataFrame(list_rules_all)
    list_rules_df.to_csv('/Users/mkhan149/Downloads/Experiments/others/Association_rules/rules_v9_extended_15_40_without_summer.csv') 
    list_rules_df.to_json('/Users/mkhan149/Downloads/Experiments/others/Association_rules/rules_v9_extended_15_40_without_summer.json',  orient='records', lines= True) 
    
    return list_rules_df 



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

            
            for item1 in count_course.keys():
                f.write("Semester: ")
                f.write(str(semester)+ " ")
                f.write("Course ID: ")
                f.write(str(item1)+ " ")
                count_course_prior = find_prior_term(item1, prior_semester, term_dict_all_prior)

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
    #avg_mse_for_course_allocation = mse_for_course_allocation/ count1
    avg_mse_for_course_allocation_considering_not_predicted_courses = (mse_for_course_allocation+ mse_for_course_allocation_2 )/ count2
    avg_mae_for_course_allocation_considering_not_predicted_courses = (mae_for_course_allocation+ mae_for_course_allocation_2 )/ count2
    avg_msse_for_course_allocation_considering_not_predicted_courses = (msse_for_course_allocation+ msse_for_course_allocation_2 )/ count2
    avg_mase_for_course_allocation_considering_not_predicted_courses = (mase_for_course_allocation+ mase_for_course_allocation_2 )/ count2
   
    f.close()
    course_allocation_actual_predicted = pd.DataFrame(course_allocation, columns=['Semester', 'Course_ID', 'actual_n', 'predicted_n', 'predicted_n_true', 'predicted_n_false', 'avg_n_actual', 'st_dev_actual', 'number_of_terms', 'n_sts_last_offering'])
    course_allocation_actual_predicted.to_csv(output_dir+'/course_allocation_actual_predicted_updated_new_v2.csv')
    return avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list

  
#evaluate the model with test data 
#last basket is target basket
# def evaluate_with_test_data(main_dict, test, test_target, offered_courses, output_path, item_list, output_dir):
#     f = open(output_path, "w")
#     count1= 0
#     recall_test = 0.0
#     recall_test_for_one_cor_pred = 0.0
#     count_at_least_one_cor_pred = 0
#     count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred  = 0, 0, 0, 0, 0
#     count_actual_bsize_at_least_2, count_actual_bsize_at_least_3, count_actual_bsize_at_least_4, count_actual_bsize_at_least_5, count_actual_bsize_at_least_6 = 0, 0, 0, 0, 0
#     recall_temp =0
#     target_basket_size = {}
#     target_basket_size[1] = 0
#     target_basket_size[2] = 0
#     target_basket_size[3] = 0
#     target_basket_size[4] = 0
#     target_basket_size[5] = 0
#     target_basket_size[6] = 0
#     count_cor_pred = {}
#     for x5 in range(1,7):
#         for y5 in range(1,7):
#             count_cor_pred[x5,y5] = 0
    
#     term_dict = {}
#     #count_course = {}
#     term_dict_predicted = {}
#     term_dict_predicted_true = {}
#     term_dict_predicted_false = {}
#     target_basket= []
#     for basket in test_target['baskets']:
#         target_basket.append(basket)
#     #offered_courses = calculate_offered_courses(test_all)
#     #print(offered_courses)
#     index = 0
#     for user in test.userID.values:
#         #index = test[test['userID'] == user].index.values[0]
#         #print(index)
#         len_baskets= test.iloc[index]['num_baskets']
#         user_baskets = test.iloc[index]['baskets']
#         target_semester = test.iloc[index]['last_semester']
#         top_k = len(target_basket[index])
#         pred_basket= recommend_top_k(main_dict, len_baskets+1, user_baskets, offered_courses[target_semester], top_k, item_list)
#         f.write("UserID: ")
#         f.write(str(user)+ "| ")
#         f.write("target basket: "+ str(target_basket[index]))
#         f.write(", Recommended basket: ")
#         for item3 in pred_basket:
#             f.write(str(item3)+ " ")
#         f.write("\n") 
#         #calculate recall
#         #recall_test += recall_cal(pred_basket, target_basket[index])
#         #CIS dept
#         recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred = recall_cal(pred_basket, target_basket[index], count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred) 
#         #recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred = recall_cal(pred_courses_CIS, target_courses_CIS, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred) 
#         term_dict = calculate_term_dict_2(term_dict, target_semester, target_basket[index])

#         #course allocation for predicted courses
#         term_dict_predicted = calculate_term_dict_2(term_dict_predicted, target_semester, pred_basket)
#         term_dict_predicted_true = calculate_term_dict_true(term_dict_predicted_true, target_semester, target_basket[index], pred_basket)
#         term_dict_predicted_false = calculate_term_dict_false(term_dict_predicted_false, target_semester, target_basket[index], pred_basket)
#         if top_k>=2: count_actual_bsize_at_least_2 += 1
#         if top_k>=3: count_actual_bsize_at_least_3 += 1
#         if top_k>=4: count_actual_bsize_at_least_4 += 1
#         if top_k>=5: count_actual_bsize_at_least_5 += 1
#         if top_k>=6: count_actual_bsize_at_least_6 += 1

#         recall_test += recall_temp
#         if recall_temp>0:  recall_test_for_one_cor_pred += recall_temp
#         t_length = top_k
#         if t_length>=6: target_basket_size[6] += 1 
#         else: target_basket_size[t_length] += 1 
#         count1 += 1
#         index += 1
    
#     test_recall = recall_test/ count1
#     print("Recall@n for test data: ", test_recall)
#     f.write("Recall@n for test data: "+ str(test_recall))
#     print("count_at_least_one_cor_pred ", count_at_least_one_cor_pred)
#     # percentage_of_at_least_one_cor_pred = (count_at_least_one_cor_pred/ len(test)) *100
#     percentage_of_at_least_one_cor_pred = (count_at_least_one_cor_pred/ count1) *100
#     print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred)
#     f.write("percentage_of_at_least_one_cor_pred: "+ str(percentage_of_at_least_one_cor_pred)+ "\n")
#     percentage_of_at_least_two_cor_pred = (count_at_least_two_cor_pred/ count_actual_bsize_at_least_2) *100
#     print("percentage_of_at_least_two_cor_pred: ", percentage_of_at_least_two_cor_pred)
#     f.write("percentage_of_at_least_two_cor_pred: "+ str(percentage_of_at_least_two_cor_pred)+ "\n")
#     percentage_of_at_least_three_cor_pred = (count_at_least_three_cor_pred/ count_actual_bsize_at_least_3) *100
#     print("percentage_of_at_least_three_cor_pred: ", percentage_of_at_least_three_cor_pred)
#     f.write("percentage_of_at_least_three_cor_pred: "+ str(percentage_of_at_least_three_cor_pred)+ "\n")
#     percentage_of_at_least_four_cor_pred = (count_at_least_four_cor_pred/ count_actual_bsize_at_least_4) * 100
#     print("percentage_of_at_least_four_cor_pred: ", percentage_of_at_least_four_cor_pred)
#     f.write("percentage_of_at_least_four_cor_pred: "+ str(percentage_of_at_least_four_cor_pred)+ "\n")
#     percentage_of_at_least_five_cor_pred = (count_at_least_five_cor_pred/ count_actual_bsize_at_least_5) *100
#     print("percentage_of_at_least_five_cor_pred: ", percentage_of_at_least_five_cor_pred)
#     f.write("percentage_of_at_least_five_cor_pred: "+ str(percentage_of_at_least_five_cor_pred)+ "\n")
#     percentage_of_all_cor_pred = (count_all_cor_pred/ len(test)) *100
#     print("percentage_of_all_cor_pred: ", percentage_of_all_cor_pred)
#     f.write("percentage_of_all_cor_pred: "+ str(percentage_of_all_cor_pred)+ "\n")
#     #calculate Recall@n for whom we generated at least one correct prediction in test data
#     test_recall_for_one_cor_pred = recall_test_for_one_cor_pred/ count_at_least_one_cor_pred
#     print("Recall@n for whom we generated at least one correct prediction in test data: ", test_recall_for_one_cor_pred)
#     f.write("Recall@n for whom we generated at least one correct prediction in test data:"+ str(test_recall_for_one_cor_pred))

#     for x6 in range(1,7):
#         percentage_of_one_cor_pred = (count_cor_pred[x6,1]/ target_basket_size[x6]) *100
#         print("percentage of_one cor pred for target basket size {}: {}".format(x6, percentage_of_one_cor_pred))
#         percentage_of_two_cor_pred = (count_cor_pred[x6,2]/ target_basket_size[x6]) *100
#         print("percentage of_two cor pred for target basket size {}: {}".format(x6, percentage_of_two_cor_pred))
#         percentage_of_three_cor_pred = (count_cor_pred[x6,3]/ target_basket_size[x6]) *100
#         print("percentage of_three cor pred for target basket size {}: {}".format(x6, percentage_of_three_cor_pred))
#         percentage_of_four_cor_pred = (count_cor_pred[x6,4]/ target_basket_size[x6]) *100
#         print("percentage of_four cor pred for target basket size {}: {}".format(x6, percentage_of_four_cor_pred))
#         percentage_of_five_cor_pred = (count_cor_pred[x6,5]/ target_basket_size[x6]) *100
#         print("percentage of_five cor pred for target basket size {}: {}".format(x6, percentage_of_five_cor_pred))
#         percentage_of_at_least_six_cor_pred = (count_cor_pred[x6,6]/ target_basket_size[x6]) *100
#         print("percentage of_at_least_six cor pred for target basket size {}: {}".format(x6, percentage_of_at_least_six_cor_pred))

#     for x7 in range(1,7):
#         print("total count of target basket size of {}: {}".format(x7, target_basket_size[x7]))

#     for x6 in range(1,7):
#         print("one cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,1]))
#         print("two cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,2]))
#         print("three cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,3]))
#         print("four cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,4]))
#         print("five cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,5]))
#         print("six or more cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,6]))
#     count_total_course = {}
#     for keys, values in term_dict.items():
#         count_course_dict = values
#         count_course_dict = dict(sorted(count_course_dict.items(), key=lambda item: item[1], reverse= True))
#         count3 = 0
#         for cnt in count_course_dict.values():
#             count3 += cnt
#         count_total_course[keys] = count3
#         term_dict[keys] = count_course_dict
#     #sorting the courses in term dictionary based on number of occurences of courses in descending order
#     for keys, values in term_dict_predicted.items():
#         count_course_dict = values
#         count_course_dict = dict(sorted(count_course_dict.items(), key=lambda item: item[1], reverse= True))
#         term_dict_predicted[keys] = count_course_dict
#     for keys, values in term_dict_predicted_true.items():
#         count_course_dict = values
#         count_course_dict = dict(sorted(count_course_dict.items(), key=lambda item: item[1], reverse= True))
#         term_dict_predicted_true[keys] = count_course_dict
    
#     for keys, values in term_dict_predicted_false.items():
#         count_course_dict = values
#         count_course_dict = dict(sorted(count_course_dict.items(), key=lambda item: item[1], reverse= True))
#         term_dict_predicted_false[keys] = count_course_dict
#     all_data_en_pred = pd.read_json('/Users/mkhan149/Downloads/Experiments/all_data_en_pred_filtered.json', orient='records', lines= True)
#     term_dict_all, frequency_of_courses, count_course_avg, course_sd_main, course_number_terms = calculate_avg_n_actual_courses(all_data_en_pred)

#     # valid_data_unique = pd.read_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/valid_sample_filtered_unique.json', orient='records', lines= True)
#     # term_dict_valid, frequency_of_courses2, count_course_avg2, course_sd_main2, course_number_terms2 = calculate_avg_n_actual_courses(valid_data_unique, reversed_item_dict)

#     # avg_mse_for_course_allocation, avg_mse_for_course_allocation_considering_not_predicted_courses = calculate_mse_for_course_allocation(term_dict, term_dict_predicted)
#     # avg_rmse_for_course_allocation, avg_rmse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_mse_for_course_allocation), math.sqrt(avg_mse_for_course_allocation_considering_not_predicted_courses)
#     avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list = calculate_mse_for_course_allocation(term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, count_total_course, count_course_avg, course_sd_main, course_number_terms, term_dict_all, output_dir)
#     avg_rmse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_mse_for_course_allocation_considering_not_predicted_courses)
#     avg_rmsse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_msse_for_course_allocation_considering_not_predicted_courses)
#     mean_error, std_dev_error = calculate_std_dev(error_list)
#     mean_ab_error, std_dev_ab_error = calculate_std_dev(ab_error_list)
#     mean_st_error, std_dev_st_error = calculate_std_dev(st_error_list)

#     #print("avg mse for # of allocated course where we are predicting a course at least once: ",avg_mse_for_course_allocation)
#     #print("avg_mse_for_course_allocation_considering all courses available in test data: ",avg_mse_for_course_allocation_considering_not_predicted_courses)
#     #print("avg rmse for # of allocated course where we are predicting a course at least once: ",avg_rmse_for_course_allocation)
#     print("avg_mae_for_course_allocation_considering all courses available in test data: ",avg_mae_for_course_allocation_considering_not_predicted_courses)
#     print("avg_rmse_for_course_allocation_considering all courses available in test data: ",avg_rmse_for_course_allocation_considering_not_predicted_courses)
#     print("avg_mase_for_course_allocation_considering all courses available in test data: ",avg_mase_for_course_allocation_considering_not_predicted_courses)
#     print("avg_rmsse_for_course_allocation_considering all courses available in test data: ",avg_rmsse_for_course_allocation_considering_not_predicted_courses)
#     print("mean of errors: ", mean_error)
#     print("standard_deviation for errors: ", std_dev_error)
#     print("mean of absolute errors: ", mean_ab_error)
#     print("standard_deviation for absolute errors: ", std_dev_ab_error)
#     print("mean of standardized errors: ", mean_st_error)
#     print("standard_deviation for standardized errors: ", std_dev_st_error)

#     f.close()


if __name__ == '__main__':
   #train, test, valid = split_data('/Users/mdakibzabedkhan/Downloads/Experiments/Others/DREAM_2/train_sample.csv')
    #train, test, valid = split_data('/Users/mkhan149/Downloads/Experiments/Others/Association_rules_2/train_sample.csv')
#    train_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/train_data_all_CR.json', orient='records', lines= True)
   train_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/train_data_all_without_summer.json', orient='records', lines= True)
   train_data, item_dict, user_dict, reversed_item_dict, reversed_user_dict = preprocess_train_data_part1(train_data)
   
   train_all, train_set_without_target, target, max_len = preprocess_train_data_part2(train_data) 
#    train_df = pd.read_json('/Users/mkhan149/Downloads/Experiments/Others/CDREAM_LGCN/train_data_cnpc_v4.json', orient='records', lines=True)
#    num_users_train = len(train_df['user'].unique())
#    train_users = set(train_df['user'].unique())
#    train_data_2 = []
#    for user in train_users:
#             actual_items = train_df[train_df['user'] == user]['item'].tolist()
#             row1= [user, actual_items, len(actual_items)]
#             train_data_2.append(row1)
#    train_data = pd.DataFrame(train_data_2, columns=['userID', 'baskets', 'num_baskets'])
#    train_set, target_set, total_set, item_list = preprocess_data(train_data)
   #print(len(item_dict))
#    print(train_all)
#    print("max_len:", max_len)
   #print(target)
   #valid_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/valid_data_all.json', orient='records', lines= True)
#    valid_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/valid_sample_all_CR.json', orient='records', lines= True)
#    valid_data, user_dict2, reversed_user_dict2 = preprocess_valid_data_part1(valid_data, reversed_user_dict, item_dict)
#    valid_all, valid_set_without_target, valid_target = preprocess_valid_data_part2(valid_data) #  #, 
#    #print("reversed_user_dict2: ", reversed_user_dict2)
#    #print(valid_all)
#    test_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/test_sample_all_CR.json', orient='records', lines= True)
#    test_data, user_dict3, reversed_user_dict3 = preprocess_test_data_part1(test_data, reversed_user_dict, item_dict, reversed_user_dict2)
#    test_all, test_set_without_target, test_target = preprocess_test_data_part2(test_data) #, item_dict, user_dict, reversed_item_dict, reversed_user_dict #
   print("step 4 done")
   #offered_courses = offered_course_cal('/Users/mkhan149/Downloads/Experiments/all_data_CR.json')

   #label_of_students = cluster_based_on_grades(train_all, item_dict)
   #rules = generate_rules(train_all, label_of_students, reversed_item_dict)
   #freq_itemsets = generate_freq_itemsets(train_all, reversed_item_dict)
   rules = generate_rules(train_all, reversed_item_dict)
#    main_dict = measure_popular_courses(train_all_unique, train_set_without_target_set2, train_target2, offered_courses, item_list1) #training 
#    print("step 5 done")
   #print(main_dict)
#    validate(main_dict, valid, valid_target, offered_courses, item_list1)
#    print("step 6 done")
#    data_dir= '/Users/mkhan149/Downloads/Experiments/Others/Non_sequential_baseline/'
#    output_dir = data_dir + "/output_dir"
#    create_folder(output_dir)
#    output_path= output_dir+ "/prediction.txt"
#    evaluate_with_test_data(main_dict, test, test_target, offered_courses, output_path, item_list1, output_dir)
#    print("step 7 done")
