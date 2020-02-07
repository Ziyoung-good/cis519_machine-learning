#!/usr/bin/env python
# coding: utf-8

# In[723]:


import json
import matplotlib.pylab as plt
import os
import math
import random
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score


# In[724]:


def calculate_f1(y_gold, y_model):
    """
    Computes the F1 of the model predictions using the
    gold labels. Each of y_gold and y_model are lists with
    labels 1 or -1. The function should return the F1
    score as a number between 0 and 1.
    """
    ac = sum([1 for i in range(len(y_gold)) if y_gold[i] == 1])
    mp = sum([1 for i in range(len(y_gold)) if y_model[i] == 1])
    ap = sum([1 for i in range(len(y_gold)) if y_model[i] == 1 and y_model[i] == y_gold[i]])
    precision = ap / mp
    recall = ap / ac 
    f1 = 2 * precision * recall / (precision + recall)
    
    print('')

    return f1


# In[725]:


class Classifier(object):
    """
    The Classifier class is the base class for all of the Perceptron-based
    algorithms. Your class should override the "process_example" and
    "predict_single" functions. Further, the averaged models should
    override the "finalize" method, where the final parameter values
    should be calculated. You should not need to edit this class any further.
    """
    def train(self, X, y):
        iterations = 10
        for iteration in range(iterations):
            for x_i, y_i in zip(X, y):
                self.process_example(x_i, y_i)
        self.finalize()

    def process_example(self, x, y):
        
        """
        Makes a predicting using the current parameter values for
        the features x and potentially updates the parameters based
        on the gradient. "x" is a dictionary which maps from the feature
        name to the feature value and y is either 1 or -1.
        """
        
#         raise NotImplementedError

    def finalize(self):
        """Calculates the final parameter values for the averaged models."""
        pass

    def predict(self, X):
        """
        Predicts labels for all of the input examples. You should not need
        to override this method.
        """
        y = []
        for x in X:
            y.append(self.predict_single(x))
        return y

    def predict_single(self, x):
        """
        Predicts a label, 1 or -1, for the input example. "x" is a dictionary
        which maps from the feature name to the feature value.
        """
#         raise NotImplementedError


# In[726]:


class Perceptron(Classifier):
    def __init__(self, features):
        """
        Initializes the parameters for the Perceptron model. "features"
        is a list of all of the features of the model where each is
        represented by a string.
        """
        # Do not change the names of these 3 data members because
        # they are used in the unit tests
        self.eta = 1
        self.w = {feature: 0.0 for feature in features}
        self.theta = 0
        
        
         
    def process_example(self, x, y):
        y_pred = self.predict_single(x)
        if y != y_pred:
            for feature, value in x.items():
                self.w[feature] += self.eta * y * value
            self.theta += self.eta * y

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1


# For the rest of the Perceptron-based algorithms, you will have to implement the corresponding class like we have done for "Perceptron".
# Use the "Perceptron" class as a guide for how to implement the functions.

# In[727]:


class AveragedPerceptron(Classifier):
    def __init__(self, features):
        self.eta = 1
        self.w = {feature: 0.0 for feature in features}
        self.theta = 0
        # You will need to add data members here
        
        self.w_accumu = {feature: 0.0 for feature in features}
        self.w_accumu['theta'] = 0
        self.fix = 0
        self.total = 0
        
        
    def process_example(self, x, y):
        y_pred = self.predict_single(x)
        if y != y_pred:
            for feature, value in x.items():
                    self.w[feature] = self.w[feature] + y * self.eta * value
                    self.w_accumu[feature] = self.w_accumu[feature] + self.total * (y * self.eta * value)
            self.w_accumu['theta'] = self.w_accumu['theta'] + self.total * (y * self.eta)
            self.theta = self.theta + y * self.eta
            self.total += self.fix
            self.fix = 1
        else:
            self.fix += 1 
#         raise NotImplementedError

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1
#         raise NotImplementedError
        
    def finalize(self):
        for feature in self.w:
            self.w[feature] = self.w[feature]-self.w_accumu[feature] / self.total
        self.theta = self.theta - self.w_accumu['theta'] / self.total
#         raise NotImplementedError


# In[728]:


class Winnow(Classifier):
    def __init__(self, alpha, features):
        # Do not change the names of these 3 data members because
        # they are used in the unit tests
        self.alpha = alpha
        self.w = {feature: 1.0 for feature in features}
        self.theta = -len(features)
        
    def process_example(self, x, y):
        y_pred = self.predict_single(x)
        if y != y_pred:
            for feature, value in x.items():
                self.w[feature] = self.w[feature] * pow(self.alpha, y * value)
#         raise NotImplementedError

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1
#         raise NotImplementedError


# In[729]:


class AveragedWinnow(Classifier):
    def __init__(self, alpha, features):
        self.alpha = alpha
        self.w = {feature: 1.0 for feature in features}
        self.theta = -len(features)
        # You will need to add data members here
        
        self.w_accumu, self.w_accumu['theta'] = {feature: 1.0 for feature in features}, -len(features)
        self.fix = 0
        self.total = 0
    def process_example(self, x, y):
        if y != self.predict_single(x):
            for feature in self.w:
                self.w_accumu[feature] = self.w_accumu[feature] + self.fix * self.w[feature]
            for feature, value in x.items():
                self.w[feature] = self.w[feature] * pow(self.alpha, y * value)
            self.total += self.fix
            self.fix = 1
        else:
            self.fix += 1
                
#         raise NotImplementedError

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1
#         raise NotImplementedError
        
    def finalize(self):
        self.total += self.fix
        for feature in self.w:
            self.w_accumu[feature] = self.w_accumu[feature] + self.fix * self.w[feature]
            self.w_accumu[feature] /= self.total
        self.w = self.w_accumu

#         raise NotImplementedError


# In[730]:


class AdaGrad(Classifier):
    def __init__(self, eta, features):
        # Do not change the names of these 3 data members because
        # they are used in the unit tests
        self.eta = eta
        self.w = {feature: 0.0 for feature in features}
        self.theta = 0
        self.G = {feature: 1e-5 for feature in features}  # 1e-5 prevents divide by 0 problems
        self.H = 0
        
    def process_example(self, x, y):
            
            L = y * (sum([self.w[feature]*value for feature, value in x.items()]) + self.theta)
            if L <= 1:
                for feature, value in x.items():
                    l1 = (-y * value)
                    self.G[feature] = self.G[feature] + (l1 ** 2)
                    self.w[feature] = self.w[feature] + self.eta * y * value / math.sqrt(self.G[feature])
                self.H = self.H + pow(-y, 2)
                self.theta = self.theta + self.eta * y / math.sqrt(self.H)
#         raise NotImplementedError

    def predict_single(self, x): 
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1
#          raise NotImplementedError


# In[731]:


class AveragedAdaGrad(Classifier):
    def __init__(self, eta, features):
        self.eta = eta
        self.w = {feature: 0.0 for feature in features}
        self.theta = 0
        self.G = {feature: 1e-5 for feature in features}
        self.H = 0
        # You will need to add data members here
        self.w_accumu = {feature: 0.0 for feature in features}
        self.w_accumu['theta'] = 0
        self.fix = 0
        self.total = 0
    def process_example(self, x, y):
            L = y * (sum([self.w[feature]*value for feature, value in x.items()]) + self.theta)
            if L <=1:
                for feature in self.w:
                        self.w_accumu[feature] = self.w_accumu[feature] + self.fix * self.w[feature]
                for feature, value in x.items():
                        g = -y * value 
                        self.G[feature] = self.G[feature] + pow(g, 2)
                        self.w[feature] = self.w[feature] - self.eta* g / math.sqrt(self.G[feature])
                self.H = self.H + pow(-y, 2)
                self.w_accumu['theta'] = self.w_accumu['theta'] + self.fix * self.theta
                self.theta = self.theta + self.eta * y / math.sqrt(self.H)
                self.total += self.fix
                self.fix = 1
            else:
                self.fix += 1
#         raise NotImplementedError

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1
#       raise NotImplementedError
        
    def finalize(self):
        self.total += self.fix
        for feature in self.w:
                self.w_accumu[feature] = self.w_accumu[feature] + self.fix * self.w[feature]
                self.w_accumu[feature] /= self.total
        self.w_accumu['theta'] = self.w_accumu['theta'] + self.fix * self.theta
        self.w_accumu['theta'] /= self.total
        self.w = self.w_accumu
        self.theta = self.w_accumu['theta']

#         raise NotImplementedError


# In[732]:


def plot_learning_curves(perceptron_accs,
                         winnow_accs,
                         adagrad_accs,
                         avg_perceptron_accs,
                         avg_winnow_accs,
                         avg_adagrad_accs,
                         svm_accs):
    """
    This function will plot the learning curve for the 7 different models.
    Pass the accuracies as lists of length 11 where each item corresponds
    to a point on the learning curve.
    """
    assert len(perceptron_accs) == 11
    assert len(winnow_accs) == 11
    assert len(adagrad_accs) == 11
    assert len(avg_perceptron_accs) == 11
    assert len(avg_winnow_accs) == 11
    assert len(avg_adagrad_accs) == 11
    assert len(svm_accs) == 11

    x = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
   
    plt.figure()
    f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')
    ax.plot(x, perceptron_accs, label='perceptron')
    ax2.plot(x, perceptron_accs, label='perceptron')
    ax.plot(x, winnow_accs, label='winnow')
    ax2.plot(x, winnow_accs, label='winnow')
    ax.plot(x, adagrad_accs, label='adagrad')
    ax2.plot(x, adagrad_accs, label='adagrad')
    ax.plot(x, avg_perceptron_accs, label='avg-perceptron')
    ax2.plot(x, avg_perceptron_accs, label='avg-perceptron')
    ax.plot(x, avg_winnow_accs, label='avg-winnow')
    ax2.plot(x, avg_winnow_accs, label='avg-winnow')
    ax.plot(x, avg_adagrad_accs, label='avg-adagrad')
    ax2.plot(x, avg_adagrad_accs, label='avg-adagrad')
    ax.plot(x, svm_accs, label='svm')
    ax2.plot(x, svm_accs, label='svm')
    ax.set_xlim(0, 5500)
    ax2.set_xlim(49500, 50000)
    ax2.set_xticks([50000])
    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()
    ax2.legend()


# In[733]:


def load_synthetic_data(directory_path):
    """
    Loads a synthetic dataset from the dataset root (e.g. "synthetic/sparse").
    You should not need to edit this method.
    """
    def load_jsonl(file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def load_txt(file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(int(line.strip()))
        return data

    def convert_to_sparse(X):
        sparse = []
        for x in X:
            data = {}
            for i, value in enumerate(x):
                if value != 0:
                    data[str(i)] = value
            sparse.append(data)
        return sparse

    X_train = load_jsonl(directory_path + '/train.X')
    X_dev = load_jsonl(directory_path + '/dev.X')
    X_test = load_jsonl(directory_path + '/test.X')

    num_features = len(X_train[0])
    features = [str(i) for i in range(num_features)]

    X_train = convert_to_sparse(X_train)
    X_dev = convert_to_sparse(X_dev)
    X_test = convert_to_sparse(X_test)

    y_train = load_txt(directory_path + '/train.y')
    y_dev = load_txt(directory_path + '/dev.y')
    y_test = load_txt(directory_path +  '/test.y')

    return X_train, y_train, X_dev, y_dev, X_test, y_test, features


# In[734]:


def run_synthetic_experiment(data_path):
    """
    Runs the synthetic experiment on either the sparse or dense data
    depending on the data path (e.g. "data/sparse" or "data/dense").
    
    We have provided how to train the Perceptron on the training and
    test on the testing data (the last part of the experiment). You need
    to implement the hyperparameter sweep, the learning curves, and
    predicting on the test dataset for the other models.
    """
    X_train, y_train, X_dev, y_dev, X_test, y_test, features         = load_synthetic_data(data_path)
    
    # TODO: Hyperparameter sweeps
    alpha_values = [1.1, 1.01, 1.005, 1.0005, 1.0001]
    winnow_para = {}
    for val in alpha_values:
        classifier = Winnow(val,features)
        classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_dev)
        acc = accuracy_score(y_dev, y_pred)
        winnow_para[val] = acc
        
    print("winnow_para for Winnow:",winnow_para)
    
    alpha_values = [1.1, 1.01, 1.005, 1.0005, 1.0001]
    Ave_winnow_para = {}
    for val in alpha_values:
        classifier = AveragedWinnow(val,features)
        classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_dev)
        acc = accuracy_score(y_dev, y_pred)
        Ave_winnow_para[val] = acc
        
    print("Ave_winnow_para for AveragedWinnow:",Ave_winnow_para)
    
    eta_values = [1.5, 0.25, 0.03, 0.005, 0.001]
    Ada_para = {}
    for val in eta_values:
        classifier = AdaGrad(val,features)
        classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_dev)
        acc = accuracy_score(y_dev, y_pred)
        Ada_para[val] = acc
        
    print("Ada_para for AdaGrad:",Ada_para)
    
    Ave_Ada_para = {}
    for val in eta_values:
        classifier = AveragedAdaGrad(val,features)
        classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_dev)
        acc = accuracy_score(y_dev, y_pred)
        Ave_Ada_para[val] = acc
    
    print("Ave_Ada_para for AveragedAdaGrad:",Ave_Ada_para)
    winnow_best = max(zip(winnow_para.values(),winnow_para.keys()))[1]
    # TODO: Placeholder data for the learning curves. You should write
    # the logic to downsample the dataset to the number of desired training
    # instances (e.g. 500, 1000), then train all of the models on the
    # sampled dataset. Compute the accuracy and add the accuraices to
    # the corresponding list.

    perceptron_accs = [0.1] * 11
    winnow_accs = [0.2] * 11
    adagrad_accs = [0.3] * 11
    avg_perceptron_accs = [0.4] * 11
    avg_winnow_accs = [0.5] * 11
    avg_adagrad_accs = [0.6] * 11
    svm_accs = [0.7] * 11
    
    
    train_sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
    count = 0 
    random.seed(5)
    for j in train_sizes:
        randomX = []
        randomY = []
        for i in random.sample(range(0, 50000), j):
            randomX.append(X_train[i])
            randomY.append(y_train[i])
        classifier = Perceptron(features)
        classifier.train(randomX, randomY)
        y_pred = classifier.predict(X_dev)
        acc = accuracy_score(y_dev, y_pred)
        perceptron_accs[count] = acc
        count += 1
        
    count =0
    
    for j in train_sizes:
        randomX = []
        randomY = []
        for i in random.sample(range(0, 50000), j):
            randomX.append(X_train[i])
            randomY.append(y_train[i])
        classifier = Winnow(winnow_best,features)
        classifier.train(randomX, randomY)
        y_pred = classifier.predict(X_dev)
        acc = accuracy_score(y_dev, y_pred)
        winnow_accs[count] = acc
        count+=1
        
    count =0
   
    for j in train_sizes:
        randomX = []
        randomY = []
        for i in random.sample(range(0, 50000), j):
            randomX.append(X_train[i])
            randomY.append(y_train[i])
        classifier = AveragedWinnow(1.1,features)
        classifier.train(randomX, randomY)
        y_pred = classifier.predict(X_dev)
        acc = accuracy_score(y_dev, y_pred)
        avg_winnow_accs[count] = acc
        count +=1 
    
    count = 0
  
    for j in train_sizes:
        randomX = []
        randomY = []
        for i in random.sample(range(0, 50000), j):
            randomX.append(X_train[i])
            randomY.append(y_train[i])
        classifier = AveragedPerceptron(features)
        classifier.train(randomX, randomY)
        y_pred = classifier.predict(X_dev)
        acc = accuracy_score(y_dev, y_pred)
        avg_perceptron_accs[count] = acc
        count += 1
        
    count =0
   
    for j in train_sizes:
        randomX = []
        randomY = []
        for i in random.sample(range(0, 50000), j):
            randomX.append(X_train[i])
            randomY.append(y_train[i])
        classifier = AdaGrad(1.5,features)
        classifier.train(randomX, randomY)
        y_pred = classifier.predict(X_dev)
        acc = accuracy_score(y_dev, y_pred)
        adagrad_accs[count] = acc
        count += 1
    
        
    count = 0
    
    for j in train_sizes:
        randomX = []
        randomY = []
        for i in random.sample(range(0, 50000), j):
            randomX.append(X_train[i])
            randomY.append(y_train[i])
        classifier = AveragedAdaGrad(1.5,features)
        classifier.train(randomX, randomY)
        y_pred = classifier.predict(X_dev)
        acc = accuracy_score(y_dev, y_pred)
        avg_adagrad_accs[count] = acc
        count += 1
        
    
    
    count = 0 
   
    for j in train_sizes:
        randomX = []
        randomY = []
        for i in random.sample(range(0, 50000), j):
            randomX.append(X_train[i])
            randomY.append(y_train[i])
        vectorizer = DictVectorizer()
        classifier = LinearSVC(loss= 'hinge')
        X_train_dict = vectorizer.fit_transform(randomX)
        classifier.fit(X_train_dict ,randomY)
        X_dev_dict = vectorizer.transform(X_dev)
        y_pred = classifier.predict(X_dev_dict)
        acc = accuracy_score(y_dev, y_pred)
        svm_accs[count] = acc
        count += 1
        
        
    
    plot_learning_curves(perceptron_accs, winnow_accs, adagrad_accs, avg_perceptron_accs, avg_winnow_accs, avg_adagrad_accs, svm_accs)
    
    # TODO: Train all 7 models on the training data and test on the test data
    classifier = Perceptron(features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Perceptron ', acc)
    
    classifier = Winnow(winnow_best,features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Winnow ', acc)
    
    classifier = AveragedWinnow(1.1, features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('AveragedWinnow ', acc)
    
    classifier = AveragedPerceptron(features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('AveragedPerceptron ', acc)
    
    classifier = AdaGrad(1.5,features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('AdaGrad ', acc)
    
    classifier = AveragedAdaGrad(1.5,features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('AveragedAdaGrad ', acc)
    
    vectorizer = DictVectorizer()
    X_train_dict = vectorizer.fit_transform(X_train)
    classifier = LinearSVC(loss= 'hinge')
    classifier.fit(X_train_dict,y_train)
    X_test = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('SVM: ', acc)
    
     
    
    
    


# In[735]:


def load_ner_data(path):
    """
    Loads the NER data from a path (e.g. "ner/conll/train"). You should
    not need to edit this method.
    """
    # List of tuples for each sentence
    data = []
    for filename in os.listdir(path):
        with open(path + '/' + filename, 'r') as file:
            sentence = []
            for line in file:
                if line == '\n':
                    data.append(sentence)
                    sentence = []
                else:
                    sentence.append(tuple(line.split()))
    return data


# In[736]:


def extract_ner_features_train(train):
    """
    Extracts feature dictionaries and labels from the data in "train"
    Additionally creates a list of all of the features which were created.
    We have implemented the w-1 and w+1 features for you to show you how
    to create them.
    
    TODO: You should add your additional featurization code here.
    """
    y = []
    X = []
    features = set([])
    for sentence in train:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        for i in range(3, len(padded) - 3):
            y.append(1 if padded[i][1] == 'I' else -1)
            feat3 = 'w-1=' + str(padded[i - 1][0])
            feat4 = 'w+1=' + str(padded[i + 1][0])
            feat2 = 'w-2=' + str(padded[i - 2][0])
            feat1 = 'w-3=' + str(padded[i - 3][0])
            feat5 = 'w+2=' + str(padded[i + 2][0])
            feat6 = 'w+3=' + str(padded[i + 3][0])
            feat7 = 'w-1'+str(padded[i-1][0])+'&w-2='+ str(padded[i-2][0])
            feat8 = 'w+1'+str(padded[i+1][0])+  '&w+2=' +  str(padded[i+2][0])
            feat9 = 'w-1'+str(padded[i-1][0])+  '&w+1=' +  str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
            features.update(feats)
            feats = {feature: 1 for feature in feats}
            X.append(feats)
    return features, X, y


# In[737]:


def extract_features_dev_or_test(data, features):
    """
    Extracts feature dictionaries and labels from "data". The only
    features which should be computed are those in "features". You
    should add your additional featurization code here.
    
    TODO: You should add your additional featurization code here.
    """
    y = []
    X = []
    
    for sentence in data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        for i in range(3, len(padded) - 3):
            y.append(1 if padded[i][1] == 'I' else -1)
            feat3 = 'w-1=' + str(padded[i - 1][0])
            feat4 = 'w+1=' + str(padded[i + 1][0])
            feat2 = 'w-2=' + str(padded[i - 2][0])
            feat1 = 'w-3=' + str(padded[i - 3][0])
            feat5 = 'w+2=' + str(padded[i + 2][0])
            feat6 = 'w+3=' + str(padded[i + 3][0])
            feat7 = 'w-1'+str(padded[i-1][0])+  '&w-2=' + str(padded[i-2][0])
            feat8 = 'w+1'+str(padded[i+1][0])+  '&w+2=' +  str(padded[i+2][0])
            feat9 = 'w-1'+str(padded[i-1][0])+  '&w+1=' +  str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
            feats = {feature: 1 for feature in feats if feature in features}
            X.append(feats)
    return X, y


# In[738]:


def run_ner_experiment(data_path):
    """
    Runs the NER experiment using the path to the ner data
    (e.g. "ner" from the released resources). We have implemented
    the standard Perceptron below. You should do the same for
    the averaged version and the SVM.
    
    The SVM requires transforming the features into a different
    format. See the end of this function for how to do that.
    """
    train = load_ner_data(data_path + '/conll/train')
    conll_test = load_ner_data(data_path + '/conll/test')
    enron_test = load_ner_data(data_path + '/enron/test')

    features, X_train, y_train = extract_ner_features_train(train)
    X_conll_test, y_conll_test = extract_features_dev_or_test(conll_test, features)
    X_enron_test, y_enron_test = extract_features_dev_or_test(enron_test, features)
                 
    # You should do this for the Averaged Perceptron and SVM
    classifier = AveragedPerceptron(features)
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_conll_test)
    conll_f1 = calculate_f1(y_conll_test, y_pred)
    y_pred = classifier.predict(X_enron_test)
    enron_f1 = calculate_f1(y_enron_test, y_pred)
    print('Averaged Perceptron')
    print('  CoNLL', conll_f1)
    print('  Enron', enron_f1)
    
    # This is how you convert from the way we represent features in the
    # Perceptron code to how you need to represent features for the SVM.
    # You can then train with (X_train_dict, y_train) and test with
    # (X_conll_test_dict, y_conll_test) and (X_enron_test_dict, y_enron_test)
    vectorizer = DictVectorizer()
    classifier = LinearSVC(loss= 'hinge')
    X_train_dict = vectorizer.fit_transform(X_train)
    classifier.fit(X_train_dict,y_train)
    X_conll_test_dict = vectorizer.transform(X_conll_test)
    y_pred1 = classifier.predict(X_conll_test_dict)
    conll_f1 = calculate_f1(y_conll_test, y_pred1) 
    X_enron_test_dict = vectorizer.transform(X_enron_test)
    y_pred2 = classifier.predict(X_enron_test_dict)
    enron_f1 = calculate_f1(y_enron_test, y_pred2)
    print('SVM')
    print('  CoNLL', conll_f1)
    print('  Enron', enron_f1)


# In[739]:


# Run the synthetic experiment on the sparse dataset. "synthetic/sparse"
# is the path to where the data is located.
run_synthetic_experiment('synthetic/sparse')


# In[740]:


# Run the synthetic experiment on the sparse dataset. "synthetic/dense"
# is the path to where the data is located.
run_synthetic_experiment('synthetic/dense')


# In[741]:


# Run the NER experiment. "ner" is the path to where the data is located.
run_ner_experiment('ner')


# In[ ]:





# In[ ]:




