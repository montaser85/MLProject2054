
# coding: utf-8

# In[13]:


# This file create and Vectorize and out result with training

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score 
import numpy as np
import time
start_time = time.time()


# In[14]:


def TrainX(vec):
    fin = open("datasetML.txt", encoding="utf-8")
    corpus = []
    text = fin.readline()

    while text:
        corpus.append(text.strip())
        text = fin.readline()
    fin.close()
    print(len(corpus))
    return vec.fit_transform(corpus)


# In[15]:


def TrainY():
    trainY = []
    fin = open("resultML.txt")

    yval = fin.readline()
    while yval:
        trainY.append(int(yval))
        yval = fin.readline()

    return np.asarray(trainY)


# In[16]:


def SVM(trainX,trainY,testX): #Support Vector Machine
    model = svm.SVC(C=1.0,kernel="linear")
    model.fit(trainX, trainY)
    result = []
    for i in testX:
        result.append(model.predict(i))

    return np.asarray(result)


# In[17]:


def LR(trainX, trainY, testX, testY):#Logistic Regression
    clf = LogisticRegression(fit_intercept=True, C = 1e15)
    clf.fit(trainX, trainY)

    print ('Accuracy from logistic regression: {0}'.format(clf.score(testX, testY)))

    print (clf.intercept_, clf.coef_)
    # print (weights)


# In[18]:


def KNC(trainX, trainY, testX, testY): #K Neighbors Classifier
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=50, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=3, p=2,
           weights='distance')
    knn.fit(trainX, trainY)
    # print(knn.predict(testX))
    print(knn.predict_proba(testX))
#    print(knn.predict_proba(data)[:, 1])
    print('accuracy for KNN:{0}'.format(knn.score(testX, testY)))


# In[19]:


def MLP(trainX, trainY, testX, testY): #Multi Layer Perceptron
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, ), random_state=1)
    clf.fit(trainX, trainY)
    clf.predict(testX)
    print(clf.predict_proba(testX))
    print('accuracy for MLP:{0}'.format(clf.score(testX, testY)))


# def GNB(trainX, trainY, testX, testY):
#     clf=GaussianNB()
#     clf.fit(trainX, trainY)
#     clf.predict(testX)
#     accuracy=accuracy_score(pred,testY)
#     print('accuracy for GNB:",accuracy)

# In[20]:


if __name__ == '__main__':
    vec = CountVectorizer(tokenizer=lambda x: x.split(),ngram_range=(1,4) )
    dataX = TrainX(vec)
    dataY = TrainY()

   # print(dataX.shape)
    #x = dataX[8797:]
    #y = dataY[8797:]
    #tx = dataX[:3797]
    #ty = dataY[:3797]
    
    x = dataX[14500:16000]
    y = dataY[14500:16000]
    tx = dataX[16000:16500]
    ty = dataY[16000:16500]

    predictedResult = SVM(x,y,tx)

    #print(f1_score(ty,predictedResult))

    cm = confusion_matrix(ty,predictedResult)
    accu = accuracy_score(ty,predictedResult)

   # print(cm)
    print("Accuracy from SVM = ", accu)
    print("--- %s seconds ---" % (time.time() - start_time))

    predict2= LR(x, y, tx, ty)

    print("--- %s seconds ---" % (time.time() - start_time))
    
    KNC(x, y, tx, ty)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    MLP(x, y, tx, ty)
    print("--- %s seconds ---" % (time.time() - start_time))

