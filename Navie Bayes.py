#!/usr/bin/env python
# coding: utf-8

# In[14]:


import json
import numpy 
import collections
import random
from scipy.special import logsumexp


# In[3]:


def get_vocabulary(D):
    """
    Given a list of documents, where each document is represented as
    a list of tokens, return the resulting vocabulary. The vocabulary
    should be a set of tokens which appear more than once in the entire
    document collection plus the "<unk>" token.
    """
    # TODO
#   raise NotImplementedError
    result =  set() 
    word_dict = set() 
    for i in D:
        for a in i:
            if a in result:
                word_dict.add(a)
            else:
                result.add(a)
    word_dict.add('<unk>')
    return word_dict
                
        


# In[4]:


class BBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the binary bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
        # raise NotImplementedError
        word_dict = {}
        for i in doc:
            if i in vocab:
                word_dict[i] = 1
            else:
                word_dict['<unk>'] = 1
                
        return word_dict
            


# In[5]:


class CBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the count bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
#         raise NotImplementedError
        word_dict = {}
        for i in doc:
            if i in vocab:
                if i not in word_dict:
                    word_dict[i] = 1
                else:
                    word_dict[i] += 1
            else:
                if '<unk>' not in word_dict:
                    word_dict['<unk>'] = 1 
                else:
                    word_dict['<unk>'] += 1     
        return word_dict


# In[6]:


def compute_idf(D, vocab):
    """
    Given a list of documents D and the vocabulary as a set of tokens,
    where each document is represented as a list of tokens, return the IDF scores
    for every token in the vocab. The IDFs should be represented as a dictionary that
    maps from the token to the IDF value. If a token is not present in the
    vocab, it should be mapped to "<unk>".
    """
    # TODO
#     raise NotImplementedError
    D_value = len(D)
    list_d = []
    for doc in D:
        list_d.append({i:1 for i in doc})

    idf = {}
    idf['<unk>'] = 0
    for j in range(len(list_d)):
        for a, v in list_d[j].items():
            if a in idf.keys() and a in vocab:
                idf[a] += v
            elif a not in idf.keys() and a in vocab:
                idf[a] = v

    for j in range(len(list_d)):
        for a, v in list_d[j].items():
            if a not in vocab:
                idf['<unk>'] += v
                break
    for i in idf:
        idf[i] = numpy.log(D_value / idf[i])

    #     for word1 in vocab:
    #         if word1 != '<unk>':
    #             for word2 in D:
    #                 if word1 in word2:
    #                     if word1 not in idf_dict:
    #                         idf_dict[word1] = 1
    #                     else:
    #                         idf_dict[word1] += 1
    #             idf_dict[word1] = numpy.log(D_value / idf_dict[word1])
    #     for word3 in D:
    #         for word4 in word3:
    #             if word4 not in vocab:
    #                 if '<unk>' not in idf_dict:
    #                     idf_dict['<unk>'] = 1
    #                     break
    #                 else:
    #                     idf_dict['<unk>'] += 1
    #                     break

    #     idf_dict['<unk>'] = numpy.log(D_value / idf_dict['<unk>'])

    return idf

            
           
    
class TFIDFFeaturizer(object):
    def __init__(self, idf):
        """The idf scores computed via `compute_idf`."""
        self.idf = idf
    
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and
        the vocabulary as a set of tokens, compute
        the TF-IDF feature representation. This function
        should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
        # raise NotImplementedError
        word_dict = {}
        word_dict_count = {}
        for i in doc:
            if i in vocab:
                if i not in word_dict_count:
                    word_dict_count[i] = 1
                else:
                    word_dict_count[i] += 1
            else:
                if '<unk>' not in word_dict_count:
                    word_dict_count['<unk>'] = 1
                else:
                    word_dict_count['<unk>'] += 1

        for i in doc:
            if i in word_dict_count:
                word_dict[i] = word_dict_count[i] * self.idf[i]
            if i not in word_dict_count:
                word_dict['<unk>'] = word_dict_count['<unk>'] * self.idf['<unk>']
        return word_dict


# In[7]:


# You should not need to edit this cell
def load_dataset(file_path):
    D = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            instance = json.loads(line)
            D.append(instance['document'])
            y.append(instance['label'])
    return D, y

def convert_to_features(D, featurizer, vocab):
    X = []
    for doc in D:
        X.append(featurizer.convert_document_to_feature_dictionary(doc, vocab))
    return X


# In[8]:


def train_naive_bayes(X, y, k, vocab):
    """
    Computes the statistics for the Naive Bayes classifier.
    X is a list of feature representations, where each representation
    is a dictionary that maps from the feature name to the value.
    y is a list of integers that represent the labels.
    k is a float which is the smoothing parameters.
    vocab is the set of vocabulary tokens.
    
    Returns two values:
        p_y: A dictionary from the label to the corresponding p(y) score
        p_v_y: A nested dictionary where the outer dictionary's key is
            the label and the innner dictionary maps from a feature
            to the probability p(v|y). For example, `p_v_y[1]["hello"]`
            should be p(v="hello"|y=1).
    """
    
    
    y_size = len(X)
    p_y = {}
    py = collections.Counter(y)
  
    for i in py:
        p_y[i] = py[i] / y_size
        
    p_v_y ={}
    index_list = {}
    

#     denominator1 = 0
#     denominator2 = 0
#     sum1 = 0
#     sum2 = 0

#     for word in vocab:
#         for index in range(len(y)):
#             if y[index] == 0:
#                 if word in X[index]:
#                     sum1 += X[index][word]
#             if y[index] == 1:
#                 if word in X[index]:
#                     sum2 += X[index][word]

#     denominator1 = k * len(vocab) + sum1
#     denominator2 = k * len(vocab) + sum2

    #     for word in vocab:
    #         sum0 = 0
    #         sum1 = 0
    #         for index in range(len(y)):
    #             if y[index]==0:
    #                 if word in X[index]:
    #                     sum0 += X[index][word]
    #             if y[index]==1:
    #                 if word in X[index]:
    #                     sum1 += X[index][word]
    #         y_v_p[0][word] = (sum0+k) / denominator1
    #         y_v_p[1][word] = (sum1+k) / denominator2
    #     return py_dict , y_v_p
    
    
    for i in p_y:
        index = [j for j, l in enumerate(y) if l == i]
        index_list[i] = index
        p_v_y[i] = {}
  
    
    for i in index_list:
        for j in index_list[i]:
#             p_v_y[i] = dict(collections.Counter(p_v_y[i]) + collections.Counter(X[j]))
            for a, v in X[j].items():
                if a in p_v_y[i].keys():
                    p_v_y[i][a] += v
                else:
                    p_v_y[i][a] = v

        
 
    for word in vocab:
        if word not in p_v_y[0]:
            p_v_y[0][word] = 0 
        if word not in p_v_y[1]:
            p_v_y[1][word] = 0     

    
    for i in p_v_y:
        sum1 = 0
        for word in p_v_y[i]:
            sum1 += p_v_y[i][word]
        for word in p_v_y[i]:
            p_v_y[i][word] = (k + p_v_y[i][word]) / (sum1 + k*len(vocab))
            
    return p_y, p_v_y
    # TODO
    # raise NotImplementedError


# In[20]:


def predict_naive_bayes(D, p_y, p_v_y):
    """
    Runs the prediction rule for Naive Bayes. D is a list of documents,
    where each document is a list of tokens.
    p_y and p_v_y are output from `train_naive_bayes`.
    
    Note that any token which is not in p_v_y should be mapped to
    "<unk>". Further, the input dictionaries are probabilities. You
    should convert them to log-probabilities while you compute
    the Naive Bayes prediction rule to prevent underflow errors.
    
    Returns two values:
        predictions: A list of integer labels, one for each document,
            that is the predicted label for each instance.
        confidences: A list of floats, one for each document, that is
            p(y|d) for the corresponding label that is returned.
    """
    # TODO
#     raise  
    y_pred = []
    y_con = []
    for documents in D:
        result = []
        positive = 0
        negative = 0
        con_pos = 1
        con_neg = 1
        for word in documents:
            if word not in p_v_y[1]:
                positive += numpy.log(p_v_y[1]['<unk>'])
                negative += numpy.log(p_v_y[0]['<unk>'])
#                 con_pos *= p_v_y[1]['<unk>']
#                 con_neg *= p_v_y[0]['<unk>']
            else:
                positive += numpy.log(p_v_y[1][word])
                negative += numpy.log(p_v_y[0][word])
#                 con_pos *= p_v_y[1][word]
#                 con_neg *= p_v_y[0][word]
        positive =  positive+numpy.log(p_y[1])
        negative =  negative+numpy.log(p_y[0])
        count = 0
        confidence = 0
        pd = []
        pd.append(positive)
        pd.append(negative)
        pd = numpy.array(pd)
        pd = logsumexp(pd)
        if positive > negative:
            count = 1
            y_pred.append(1)
            confidence = numpy.exp(positive - pd) 
            y_con.append(confidence)
        else:
            count = 0
            y_pred.append(0)
            confidence = numpy.exp(negative - pd) 
            y_con.append(confidence)
            
#         confidence = 1
#         for word in documents:
#             if word not in p_v_y[count]:
#                 confidence *= p_v_y[count]['<unk>']
#             else:
#                 confidence *= p_v_y[count][word]
#         confidence *= p_y[count]
#         pd = (con_neg*p_y[0] + con_pos*p_y[1])
#         if pd ==0:
#             confidence = 1
#         else:
#             confidence /= pd  
        
    return y_pred, y_con       
                   


# In[10]:


def train_semi_supervised(X_sup, y_sup, D_unsup, X_unsup, D_valid, y_valid, k, vocab, mode):
    """
    Trains the Naive Bayes classifier using the semi-supervised algorithm.
    
    X_sup: A list of the featurized supervised documents.
    y_sup: A list of the corresponding supervised labels.
    D_unsup: The unsupervised documents.
    X_unsup: The unsupervised document representations.
    D_valid: The validation documents.
    y_valid: The validation labels.
    k: The smoothing parameter for Naive Bayes.
    vocab: The vocabulary as a set of tokens.
    mode: either "threshold" or "top-k", depending on which selection
        algorithm should be used.
    
    Returns the final p_y and p_v_y (see `train_naive_bayes`) after the
    algorithm terminates.    
    """
    # TODO
#     raise NotImplementedError
    count1 = 0
    while(True):
        print(len(X_sup))
        
        p_y , p_v_y = train_naive_bayes(X_sup, y_sup, k, vocab)
        acc_first = 0
        if count1 == 0:
            y_pred, y_con = predict_naive_bayes(D_valid, p_y, p_v_y)
            for i in range(len(y_pred)):
                if y_pred[i] == y_valid[i]:
                    acc_first += 1
            acc_first = acc_first / len(y_pred)
            print(acc_first)
        count1 += 1
        y_pred, y_con = predict_naive_bayes(D_unsup, p_y, p_v_y)
        if (mode =='threshold'):
            new_List =[]
            for i in range(len(y_pred)):
                new_List.append((y_pred[i],y_con[i],X_unsup[i],D_unsup[i]))   
            count = 0
            for i in range(len(y_pred)):
                if new_List[i][1] >=0.98:
                    if new_List[i][2] not in X_sup:
                        X_sup.append(new_List[i][2])
                        y_sup.append(new_List[i][0])
                        X_unsup.remove(new_List[i][2])
                        D_unsup.remove(new_List[i][3])
                        count +=1
            if count == 0:
                return p_y, p_v_y
        if (mode == 'top-k'):
            if len(X_unsup) == 0:
                return p_y, p_v_y
            new_List =[]
            count = 0
            for i in range(len(y_pred)):
                new_List.append((y_pred[count],y_con[count],X_unsup[count],D_unsup[i]))
                count += 1
            new_List = sorted(new_List, key = lambda p: p[1], reverse = True)
            if len(y_pred) <=10000:
                for i in range(len(y_pred)):
                    X_sup.append(new_List[i][2])
                    y_sup.append(new_List[i][0])
                    X_unsup.remove(new_List[i][2])
                    D_unsup.remove(new_List[i][3])
            else:
                for i in range(10000):
                        X_sup.append(new_List[i][2])
                        y_sup.append(new_List[i][0])
                        X_unsup.remove(new_List[i][2])
                        D_unsup.remove(new_List[i][3])
            
            
        
            
        
        
        


# In[11]:


# Variables that are named D_* are lists of documents where each
# document is a list of tokens. y_* is a list of integer class labels.
# X_* is a list of the feature dictionaries for each document.
D_train, y_train = load_dataset('data/train.jsonl')
D_valid, y_valid = load_dataset('data/valid.jsonl')
D_test, y_test = load_dataset('data/test.jsonl')

vocab = get_vocabulary(D_train)


# In[12]:


# Compute the features, for example, using the BBowFeaturizer.
# You actually only need to conver the training instances to their
# feature-based representations.
# 
# This is just starter code for the experiment. You need to fill in
# the rest.
featurizer = BBoWFeaturizer()
X_train = convert_to_features(D_train, featurizer, vocab)


# In[13]:


# for each representation to choose best k 
k_list = [0.001,0.01,0.1,1.0,10]
# for BBO:
featurizer = BBoWFeaturizer()
X_train = convert_to_features(D_train, featurizer, vocab)
acc_list1 =[]
for i in k_list:
    p_y, p_v_y = train_naive_bayes(X_train, y_train, i, vocab)
    y_pred, ycon = predict_naive_bayes(D_valid, p_y, p_v_y)
    acc = 0
    for a in range(len(y_pred)):
        if y_pred[a] == y_valid[a]:
            acc += 1
    acc_list1.append(acc/len(y_pred))
print(acc_list1)
#predict:
p_y, p_v_y = train_naive_bayes(X_train, y_train, 0.1, vocab)
y_pred, ycon = predict_naive_bayes(D_test, p_y, p_v_y)
acc = 0
for a in range(len(y_pred)):
    if y_pred[a] == y_test[a]:
        acc += 1
accuracy = acc/len(y_test)
print(accuracy)


# In[264]:


# for each representation to choose best k 
k_list = [0.001,0.01,0.1,1.0,10]
# for CBO:
featurizer = CBoWFeaturizer()
X_train = convert_to_features(D_train, featurizer, vocab)
acc_list1 =[]
for i in k_list:
    p_y, p_v_y = train_naive_bayes(X_train, y_train, i, vocab)
    y_pred, ycon = predict_naive_bayes(D_valid, p_y, p_v_y)
    acc = 0
    for a in range(len(y_pred)):
        if y_pred[a] == y_valid[a]:
            acc += 1
    acc_list1.append(acc/len(y_pred))
print(acc_list1)
# predict:
p_y, p_v_y = train_naive_bayes(X_train, y_train, 0.1, vocab)
y_pred, ycon = predict_naive_bayes(D_test, p_y, p_v_y)
acc = 0
for a in range(len(y_pred)):
    if y_pred[a] == y_test[a]:
        acc += 1
accuracy = acc/len(y_test)
print(accuracy)


# In[302]:


# for each representation to choose best k 
k_list = [0.001,0.01,0.1,1.0,10]
# for CBO:
idf = compute_idf(D_train, vocab)
featurizer = TFIDFFeaturizer(idf)
X_train = convert_to_features(D_train, featurizer, vocab)
acc_list1 =[]
for i in k_list:
    p_y, p_v_y = train_naive_bayes(X_train, y_train, i, vocab)
    y_pred, ycon = predict_naive_bayes(D_valid, p_y, p_v_y)
    acc = 0
    for a in range(len(y_pred)):
        if y_pred[a] == y_valid[a]:
            acc += 1
    acc_list1.append(acc/len(y_pred))
print(acc_list1)
# predict:
p_y, p_v_y = train_naive_bayes(X_train, y_train, 1.0, vocab)
y_pred, ycon = predict_naive_bayes(D_test, p_y, p_v_y)
acc = 0
for a in range(len(y_pred)):
    if y_pred[a] == y_test[a]:
        acc += 1
accuracy = acc/len(y_test)
print(accuracy)


# In[27]:


#For semi-supervised classifier:
# BBO:
featurizer = BBoWFeaturizer()
X_train = convert_to_features(D_train, featurizer, vocab)

new_X_sup = []
new_y_sup = []
new_D_unsup = D_train.copy()
new_X_unsup = X_train.copy()
List_500=random.sample(range(0,len(D_train)),50)
for i in List_500:
    new_X_sup.append(X_train[i])
    new_y_sup.append(y_train[i])
    new_D_unsup.remove(D_train[i])
    new_X_unsup.remove(X_train[i])
mode = 'top-k'
k = 0.1
p_y, p_v_y = train_semi_supervised(new_X_sup, new_y_sup, new_D_unsup, new_X_unsup, D_valid, y_valid, k, vocab, mode)
y_pred, y_con = predict_naive_bayes(D_valid, p_y, p_v_y)
acc_end = 0
for i in range(len(y_pred)):
        if y_pred[i] == y_valid[i]:
            acc_end += 1
acc_end = acc_end / len(y_pred)
print(acc_end)
y_pred, y_con = predict_naive_bayes(D_test, p_y, p_v_y)
acc_test = 0
for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            acc_test += 1
acc_test = acc_test / len(y_pred)
print(acc_test)


# In[34]:


#For semi-supervised classifier:
# CBO:
featurizer = CBoWFeaturizer()
X_train = convert_to_features(D_train, featurizer, vocab)
new_X_sup = []
new_y_sup = []
new_D_unsup = D_train.copy()
new_X_unsup = X_train.copy()
List_500=random.sample(range(0,len(D_train)),50)
for i in List_500:
    new_X_sup.append(X_train[i])
    new_y_sup.append(y_train[i])
    new_D_unsup.remove(D_train[i])
    new_X_unsup.remove(X_train[i])
mode = 'top-k'
k = 0.1
p_y, p_v_y = train_semi_supervised(new_X_sup, new_y_sup, new_D_unsup, new_X_unsup, D_valid, y_valid, k, vocab, mode)
y_pred, y_con = predict_naive_bayes(D_valid, p_y, p_v_y)
acc_end = 0
for i in range(len(y_pred)):
        if y_pred[i] == y_valid[i]:
            acc_end += 1
acc_end = acc_end / len(y_pred)
print(acc_end)
y_pred, y_con = predict_naive_bayes(D_test, p_y, p_v_y)
acc_test = 0
for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            acc_test += 1
acc_test = acc_test / len(y_pred)
print(acc_test)


# In[41]:


idf = compute_idf(D_train, vocab)
featurizer = TFIDFFeaturizer(idf)
X_train = convert_to_features(D_train, featurizer, vocab)
new_X_sup = []
new_y_sup = []
new_D_unsup = D_train.copy()
new_X_unsup = X_train.copy()
List_500=random.sample(range(0,len(D_train)),500)
for i in List_500:
    new_X_sup.append(X_train[i])
    new_y_sup.append(y_train[i])
    new_D_unsup.remove(D_train[i])
    new_X_unsup.remove(X_train[i])
mode = 'top-k'
k = 0.1
p_y, p_v_y = train_semi_supervised(new_X_sup, new_y_sup, new_D_unsup, new_X_unsup, D_valid, y_valid, k, vocab, mode)
y_pred, y_con = predict_naive_bayes(D_valid, p_y, p_v_y)
acc_end = 0
for i in range(len(y_pred)):
        if y_pred[i] == y_valid[i]:
            acc_end += 1
acc_end = acc_end / len(y_pred)
print(acc_end)
y_pred, y_con = predict_naive_bayes(D_test, p_y, p_v_y)
acc_test = 0
for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            acc_test += 1
acc_test = acc_test / len(y_pred)
print(acc_test)


# In[339]:





# In[ ]:




