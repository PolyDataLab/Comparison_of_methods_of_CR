#! /usr/bin/python3 -u



import numpy as np
import sklearn.metrics.pairwise as skl
import pandas as pd
import configparser as cfg



feat_file = 'edm&ipsj/feature_space_C_EXT.txt'
train_file = 'edm&ipsj/mooc_data/Data/mooc.train.rating'
save = 'edm&ipsj/predictions_EXT_n'

senteces_file = 'edm&ipsj/sentences_EXT.tsv'

features_dict_file = 'edm&ipsj/feature_map_EXT.dict'

user_dir = 'edm&ipsj/w2vSP_EXT/'

user_file = 'edm&ipsj/stu_sentences_EXT.tsv'

knn_file = 'edm&ipsj/knns_test.txt'

############################################################################################################

file = open(knn_file, "r", encoding="utf-8")
knns = file.read().splitlines()
file.close()

knns = [int(x) for x in knns]

############################################################################################################

print("Loading users...")

file = open(user_file, "r", encoding="utf-8")
users = file.read().splitlines()
file.close()

dicto = dict()

for user in users:

    file = open(user_dir + user, "r", encoding="utf-8")
    feats = file.read().splitlines()
    file.close()

    dicto[user] = []

    for feat in feats:
        uri, w = feat.split("\t")
        dicto[user].append((uri, float(w)))

#############################################################################################################

print("Loading features...")

file = open(feat_file, "r", encoding="utf-8")
features = file.read().splitlines()
file.close()

index = dict()

for i, feat in enumerate(features):
    index[feat] = i

#############################################################################################################

print("Loading training file...")

file = open(train_file, "r", encoding="utf-8")
lines = file.read().splitlines()
file.close()

items = set()
train = dict()

for line in lines:

    words = line.split("\t")
    user = words[0]
    item = int(words[1])
    rating = float(words[2])

    items.add(item)

    if user not in train:
        train[user] = []
    train[user].append((item, rating))

items = list(items)
it_index = dict()
for i, item in enumerate(items):
    it_index[item] = i

rating_mtx = np.zeros((len(users), len(items)))

for i, user in enumerate(users):
    its = train[user]
    for it in its:
        j = it_index[it[0]]
        rating_mtx[i, j] = it[1]

#############################################################################################################

print("Creating matrix...")

matrix = np.zeros((len(users), len(features)))

for i, user in enumerate(users):

    feats = dicto[user]

    for feat in feats:
        j = index[feat[0]]
        matrix[i, j] = feat[1]

#############################################################################################################

def getRating(similarity, matrix, rated_index):
    matrix = np.delete(matrix, rated_index, axis=1)
    weight = np.sum(np.absolute(similarity))
    ratings = np.dot(similarity, matrix)
    ratings = ratings / weight
    return ratings


def getRecs(rating_matrix, users, similar_users, items, save):

    users_dict = {}
    for index, user in enumerate(users):
        users_dict[user] = index

    recom = open(save, "w")

    for count, user in enumerate(users):

        array = np.copy(rating_matrix[count])
        rated_index = np.where(array > 0)
        unrated_movies = np.array(items)
        unrated_movies = np.delete(unrated_movies, rated_index)
        matrix = []
        similarity = []
        for similar in similar_users[user]:
            index = users_dict[similar[0]]
            similarity.append(similar[1])
            matrix.append(np.copy(rating_matrix[index]))
        matrix = np.array(matrix)
        similarity = np.array(similarity, dtype=float)

        ratings = getRating(similarity, matrix, rated_index)
        index = np.argsort(ratings)[-15:]
        index = index[::-1]

        for num in index:
            recom.write(user + "\t" + str(unrated_movies[num]) + "\t" + str(ratings[num]) + "\n")

    recom.close()






similarityMatrix = skl.cosine_similarity(matrix)
np.fill_diagonal(similarityMatrix, 0)

for knn in knns:

    print("Processing", knn, "...")

    print("\tClustering...")

    clusters = dict()

    for i, user in enumerate(users):
        array = similarityMatrix[i]
        js = np.argsort(array)[-knn:]
        js = js[::-1]
        clusters[user] = []
        for j in js:
            clusters[user].append((users[j], array[j]))

    print("\tComputing recommendation...")


    getRecs(rating_mtx, users, clusters, items, save + str(knn))
    # getRecs_negative(rating_mtx, users, clusters, user_item_pairs, save + str(knn))

#############################################################################################################

print("Done.")
