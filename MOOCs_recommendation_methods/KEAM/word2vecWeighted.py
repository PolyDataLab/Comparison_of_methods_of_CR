#! /usr/bin/python3 -u

import sys
from os.path import exists
from os import makedirs

from gensim.models import Word2Vec as word2vec
import numpy as np

from time import time

from sklearn import preprocessing

from multiprocessing import cpu_count, Pool as mpPool

import configparser as cfg





senteces_file = 'edm&ipsj/sentences_EXT.tsv'

features_dict_file = 'edm&ipsj/feature_map_EXT.dict'

usersDir = 'edm&ipsj/SP_EXT/'

newDir = 'edm&ipsj/w2vSP_EXT/'

users_sentences = 'edm&ipsj/stu_sentences_EXT.tsv'

epochs = 50

size = 100

window = 500

min_count = 1


if not exists(newDir):
    makedirs(newDir)

#######################################################################################

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

#######################################################################################

# key-value: index-feature

file = open(features_dict_file, "r", encoding="utf-8")
features = file.read().splitlines()
file.close()

hashmap = dict()
reversemap = dict()
for line in features:
    index, feature = line.split("\t")
    hashmap[feature] = index
    reversemap[index] = feature

#######################################################################################


# 加载 sentence 每行是用户的feature

file = open(senteces_file, "r", encoding="utf-8")
full_sentences = file.read().splitlines()
file.close()

weighted_feats = []

sentences = []

#sentence 每行

for item in full_sentences:
    
    sentence = item.split('\t')
    
    sentences.append(sentence)
    
    weighted_feats += sentence
    # print(sentences)
    # print(weighted_feats)
    # sys.exit()

weighted_feats = set(weighted_feats)

topn = len(weighted_feats)

#######################################################################################

print("Starting training...")

tic = time()

model = word2vec(sentences, min_count=min_count, vector_size=size, epochs=epochs, window=window)

print("\tTime: ", time()-tic)

#######################################################################################

print("Starting prediction...")

file = open(users_sentences, "r", encoding="utf-8")
users = file.read().splitlines()
file.close()

dicto = dict()

# index-userID 反存
for count, user in enumerate(users):
    dicto[user] = count

def getUP(user):

    file = open(usersDir + user + ".tsv", "r", encoding="utf-8")
    lines = file.read().splitlines()
    file.close()

    i = dicto[user]

    all = []
    for line in lines:
        uri, w = line.split("\t")
        all.append((uri, float(w)))

    sentence = sentences[i]

    predictions = model.predict_output_word(sentence, topn=topn)  # sorted by softmax

    words = []
    for word in sentence:
        w = word.split("_")
        words.append(int(w[0]))

    uris = []
    values = []
    for prediction in predictions:
        feat = prediction[0]
        index, value = feat.split("_")
        index = int(index)

        if index not in words and index not in uris:
            uris.append(index)
            values.append(float(value))

    uris = [reversemap[str(x)] for x in uris]

    if (values != []):
        values = np.array(values).reshape(-1, 1)
        values = scaler.fit_transform(values)
        values = values.reshape(values.shape[0],)
        values = values.tolist()
    
        for i, value in enumerate(values):
            all.append((uris[i], float(value)))

    all = sorted(all, key=lambda x:x[1], reverse=True)

    file = open(newDir + user, "w", encoding="utf-8")
    for f in all:
        file.write(f[0] + "\t" + str(f[1]) + "\n")
    file.close()

#######################################################################################

# 多线程不加 会执行多次！！！
if __name__ == '__main__':

    #######################################################################################

    print("Starting training...")

    tic = time()

    model = word2vec(sentences, min_count=min_count, vector_size=size, epochs=epochs, window=window)

    print("\tTime: ", time()-tic)

    #######################################################################################

    print("Starting prediction...")

    file = open(users_sentences, "r", encoding="utf-8")
    users = file.read().splitlines()
    file.close()

    dicto = dict()

    # index-userID 反存
    for count, user in enumerate(users):
        dicto[user] = count

    for user in users:
        getUP(user)

    #######################################################################################

    print("Done.")
