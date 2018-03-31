import numpy as np

import glob
import csv
import operator
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.linear_model import Lasso
#from keras.models import Sequential
#from keras.layers import Dense

path ='E:\2012' # use your path
allFiles = glob.glob(path + "/*.csv")
mlist=[]
dt=[]
cp=[]
dat=0
for f in allFiles:
    with open(f, 'r') as f1:
        reader = csv.reader(f1)
        dat=dat+1
       # print(dat)
        for o in reader:
            o[1]=dat
            mlist.append(o)
path ='2013' # use your path
allFiles = glob.glob(path + "/*.csv")
for f in allFiles:
    with open(f, 'r') as f1:
        reader = csv.reader(f1)
        dat=dat+1
       # print(dat)
        for o in reader:
            o[1]=dat
            mlist.append(o)
path ='2014' # use your path
allFiles = glob.glob(path + "/*.csv")
for f in allFiles:
    with open(f, 'r') as f1:
        reader = csv.reader(f1)
        dat=dat+1
       # print(dat)
        for o in reader:
            o[1]=dat            
            mlist.append(o)
path ='2015' # use your path
allFiles = glob.glob(path + "/*.csv")
for f in allFiles:
    with open(f, 'r') as f1:
        reader = csv.reader(f1)
        dat=dat+1
      #  print(dat)
        for o in reader:
            o[1]=dat 
            mlist.append(o)
sorlist=sorted(mlist, key=operator.itemgetter(0,1), reverse=False)
na=np.array(sorlist)


trainx=[]
trainy=[]

x=0
y=na[0][0]
list1=[]
list2=[]
for i in na:    
    if(y!=i[0]):
       # print(list2)
        y=i[0]
        trainx.append(list1)
        trainy.append(list2)
        list1=[]
        list2=[]
    try: 
        kx=[]
        kx.append(float(i[1]))
        kx.append(float(i[5]))
        kx.append(float(i[6]))
        list1.append(kx)
        list2.append(float(i[5]))
    except:
        print("")


for k in range(0,1):
    """
    print(len(trainx[k]),len(trainy[k]))
    model = LinearRegression()
    print(trainx)
    model.fit(trainx[k][0:400],trainy[k][1:401])

    mmm=model.predict(np.array(trainx[k]))
    print(mmm)
    

    model = Sequential()
    model.add(Dense(4, input_dim=3,kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(np.array(trainx[k][0:400]),np.array(trainy[k][1:401]), epochs=150, batch_size=10)
    predictions = model.predict(np.array(trainx[k]))
    #print(predictions)
    """
    
    
    
    def accFind(result):
        acc = 0
        for i in range(0,len(result)):
            acc=acc+(result[i]-trainy[k][i])**2
        acc=acc/len(result)
        acc=acc**(1/2.0)
        print(1-acc)
        
    LS = Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
    normalize=False, positive=False, precompute=False, random_state=None,
    selection='cyclic', tol=0.0001, warm_start=False)
    LS.fit(np.array(trainx[k][0:400]),np.array(trainy[k][1:401]))
    print(LS.predict(np.array(trainx[k])))
    print(trainy[k])
    result = LS.predict(np.array(trainx[k][0:400]))
    print("Lasso")
    accFind(result)

    
    from sklearn.linear_model import LassoCV
    LSC = LassoCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
    max_iter=1000, n_alphas=100, n_jobs=1, normalize=False, positive=False,
    precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
    verbose=False)
    LSC.fit(np.array(trainx[k][0:400]),np.array(trainy[k][1:401]))
    print(LSC.predict(np.array(trainx[k])))
    print(trainy[k])
    result = LSC.predict(np.array(trainx[k][0:400]))
    print("LassoCV")
    accFind(result)

    from sklearn.linear_model import ElasticNet

    EN = ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=1,
      max_iter=1000, normalize=False, positive=False, precompute=False,
      random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
    EN.fit(np.array(trainx[k][0:400]),np.array(trainy[k][1:401]))
    print(EN.predict(np.array(trainx[k])))
    print(trainy[k])
    result = EN.predict(np.array(trainx[k][0:400]))
    print("ElasticNet")
    accFind(result)

    from sklearn.linear_model import LassoLarsCV
    LL = LassoLarsCV(copy_X=True, cv=None, eps=2.2204460492503131e-16,
      fit_intercept=True, max_iter=500, max_n_alphas=1000, n_jobs=1,
      normalize=True, positive=False, precompute='auto', verbose=False)
    LL.fit(np.array(trainx[k][0:400]),np.array(trainy[k][1:401]))
    print(LL.predict(np.array(trainx[k])))
    print(trainy[k])
    result = LL.predict(np.array(trainx[k][0:400]))
    print("LassolarsCV")
    accFind(result)


    from sklearn.linear_model import RidgeCV
    RG = RidgeCV(alphas=(0.2, 2.0, 20.0), cv=None, fit_intercept=True, gcv_mode=None,
    normalize=False, scoring=None, store_cv_values=False)
    RG.fit(np.array(trainx[k][0:400]),np.array(trainy[k][1:401]))
    print(RG.predict(np.array(trainx[k])))
    print(trainy[k])
    result = RG.predict(np.array(trainx[k][0:400]))
    print("Ridge")
    accFind(result)
        
    
    
 
    
    
    
