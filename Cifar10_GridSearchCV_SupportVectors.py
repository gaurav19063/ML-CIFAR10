import pandas as pd
import pickle as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


from sklearn.model_selection import GridSearchCV
from sklearn import svm
import time

def un_pickle(path):             #given in dataset site


    with open(path, 'rb') as fo:
        data = pl.load(fo, encoding='bytes')
    return data



def read_CIFAR_data(path):
    y_train = []

    for i in range(1, 6):
        cifar_train_data_dict = un_pickle(path + "/data_batch_%d" %(i,))

        if i == 1:
            x_train = cifar_train_data_dict[b'data']
        else:
            x_train = np.vstack((x_train, cifar_train_data_dict[b'data']))

        y_train += cifar_train_data_dict[b'labels']

    y_train = np.array(y_train)

    cifar_test_data_dict = un_pickle(path + "/test_batch")
    x_test = cifar_test_data_dict[b'data']
    y_test = cifar_test_data_dict[b'labels']
    y_test = np.array(y_test)


    return x_train,y_train,x_test,y_test



#---------------------------------------------------Data Loading---------------------------------------------------------------------
path='/home/gaurav/Desktop/IIITD/ML/Assignment2/cifar-10-python/cifar-10-batches-py'

x_train,y_train,x_test,y_test=read_CIFAR_data(path)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0
pca=PCA(n_components=42)
print("yes")
pca.fit(x_train)
x_train=pca.transform(x_train)
pca=PCA(n_components=42)
pca.fit(x_test)
print("yes")
x_test=pca.transform(x_test)
print("Train data: ", x_train.shape)
print("Train labels: ", y_train.shape)
print("Test data: ", x_test.shape)
print("Test labels: ", y_test.shape)
#-------------------------------------------------GridSearchCV----------------------------------------------


parameters = {'kernel': ( 'rbf','linear'), 'C': [1.0, 10, 100]}
svc = svm.SVC(gamma='scale')
print("hii")
gsc=GridSearchCV(svc, parameters, cv=5)
print("hey")
start_time = time.time()
gsc.fit(x_train, y_train)
end_time=time.time()
print("hello")
print('Best Kernel:',gsc.best_estimator_.kernel)
print('Best C:',gsc.best_estimator_.C)
print('Best Gamma:',gsc.best_estimator_.gamma)
svc = svm.SVC(kernel=gsc.best_estimator_.kernel,C=gsc.best_estimator_.C,gamma=gsc.best_estimator_.gamma)
start_time1=time.time()
svc.fit(x_train,y_train)
end_time1=time.time()
pred=svc.predict(x_test)
print("yup")
print(accuracy_score(y_test,pred))
print("Time Taken to Tune best Parameters using GridSearchCV in seconds ------ %s" % (end_time- start_time))
print("Time Taken to fit data in seconds ------ %s" % (end_time1- start_time1))

# -----------------------------------------------------------------------------------------------------Q.1.2------------------
x=svc.support_vectors_
y=[]
x_support=pd.DataFrame(x)
data=pd.DataFrame(x_train)
y_train=pd.DataFrame(y_train)
data['lable']=y_train
for d in x_support:
    i=np.where(data.iloc[:,:-1]==d)[0][0]
    y.append(data.iloc[i,len(data.cloumns)-1])


svc = svm.SVC(kernel=gsc.best_estimator_.kernel,C=gsc.best_estimator_.C,gamma=gsc.best_estimator_.gamma)
start_time=time.time()
svc.fit(x_support,y)
end_time1=time.time()
pred_train=svc.predict(x_train)

print("Train Accuracy is:",accuracy_score(y_train,pred_train))
pred_test=pred_train=svc.predict(x_test)
print("Train Accuracy is:",accuracy_score(y_test,pred_train))
print("Time Taken to fit data in seconds ------ %s" % (end_time1- start_time1))

