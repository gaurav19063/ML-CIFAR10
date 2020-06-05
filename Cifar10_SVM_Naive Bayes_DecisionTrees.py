from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import scikitplot as skplt
from sklearn.model_selection import GridSearchCV
import time


# ------------------------------------------Pairwise Relationshis plot-----------
def PairwiseRelationPlot():
    wine = load_wine()
    dataframe = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names'] + ['target'])
    sb.pairplot(dataframe, vars=dataframe.columns[:-1], hue='target')
    plt.show()


# ---------------------------------------SVM--------------------------------------------------
def revisedData(data,i):
    for j in range (len(data)):
        # print(data[j],"hii", i)
        if(data[j]==i):
            data[j]=1
        else:
            data[j]=0
    return data

def predict_probs(model,x):
    wtx= x.dot(model.coef_.transpose())
    y = wtx+ model.intercept_
    # print(y)
    return y



def Predict(x_train, y_train,models):
    predicts=[]
    a=[]
    b=[]
    c=[]
    pos=-1
    for i in x_train:

        for j in range(3):
            p = predict_probs(models[j],i.reshape(1, -1))

            if j==0:
                a.append(p[0])
            if j==1:
                b.append(p[0])
            if j==2:
                c.append(p[0])
            # print(p)
            if p>0:
                pos=j

        predicts.append(pos)

    return predicts,a,b,c
def classAccuracy(prediction, y_test, lable):
    sum = 0
    n=len(y_test)
    for i in range(n):
        if (prediction[i] == lable and y_test[i] == lable) or (prediction[i] != lable and y_test[i] != lable):
            sum = sum + 1
    return sum * 100 / n

def classwiseAccuracy(y_test,prediction):
    for i in range (3):
        Test_Accuracy = classAccuracy(prediction, y_test, i)
        print("Test Accuracy of class", i, ' ', Test_Accuracy)
def classwiseAccuracy1(y_test,prediction):
    for i in range (3):
        Test_Accuracy = classAccuracy(prediction, y_test, i)
        print("Train Accuracy of class", i, ' ', Test_Accuracy)


def plotRoconeVsRest(y1,a,b,c):
    for i in range(3):
        z=np.copy(y1)
        y=revisedData(z,i)

        if i==0:
            k=a
        if i==1:
            k=b
        if i==2:
            k=c

        fpr, tpr, threshold = metrics.roc_curve(y, k)
        roc_auc = metrics.auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.8f' % roc_auc)
        plt.legend(loc='lower right')

        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()






def SVMOneVsRest(x_train, x_test, y_train, y_test):
    models = []
    s_time = time.time()
    for i in range (3):
        model = svm.SVC(kernel='linear', probability=True)
        n=np.copy(y_train)
        m=np.copy(y_test)
        n=revisedData(n,i)

        models.append(model.fit(x_train,n))
    e_time = time.time()

    prediction,a,b,c = Predict(x_train, y_train, models)
    print("F-1 Score of oneVsAll using svm for train data:", f1_score(y_train, prediction, average='macro'))
    print("Accuracy of oneVsAll using svm for train data:", accuracy_score(y_train, prediction))


    classwiseAccuracy1(y_train, prediction)
    y = np.copy(y_train)
    plotRoconeVsRest(y, a, b, c)
    prediction,a,b,c = Predict(x_test, y_test, models)
    print("F-1 Score of oneVsAll using svm for test data:", f1_score(y_test, prediction, average='macro'))
    print("Accuracy of oneVsAll using svm for test data:", accuracy_score(y_test, prediction))

    classwiseAccuracy(y_test,prediction)
    y = np.copy(y_test)
    plotRoconeVsRest(y, a, b, c)

    print("Time taken to fit SVM One-vs-Rest in seconds :%f" % (e_time - s_time))




def predict_probs1(model,x):
    wtx= x.dot(model.coef_.transpose())
    y = wtx+ model.intercept_
    return y



def Predict1(x,model,l):
    predicts=[]
    score=[]
    if l==0 or l==2:
        k=1
        m=1
    else:
        k=2
        m=3
    for i in x:

        p=predict_probs(model,i)
        if p<0:
            predicts.append((l+k)%3)
        else:
            predicts.append((l+m+1)%3)
        score.append(p[0])

    return predicts,score



def revisedData_ovo(dataframe,l):
    df=pd.DataFrame.copy(dataframe)
    df=df[df['target']!=l]
    y = df.pop('target')
    x = df
    return  train_test_split(x, y, test_size=0.3)
def overallPred(x_train,y_train,models):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    a,s1= Predict1(x_train, models[0], 0)
    b,s2 = Predict1(x_train, models[1], 1)
    c,s3 = Predict1(x_train, models[2], 2)
    overall_pred = []
    overall_score = []
    for i in range(len(y_train)):
        list = [a[i], b[i], c[i]]
        list1 = [s1[i], s2[i], s3[i]]
        val = max(set(list), key=list.count)
        score = max(set(list), key=list1.count)
        overall_pred.append(val)
        overall_score.append(score)

    return overall_pred

def OverallAccuracy(models,dataframe):
    df = pd.DataFrame.copy(dataframe)
    y = df.pop('target')
    x = df
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3)
    overall_pred=overallPred(x_train,y_train,models)

    print("F-1 Score of OneVsOne using svm for train data:", f1_score(y_train, overall_pred, average='macro'))
    print("Accuracy of OneVsOne using svm for train data:", accuracy_score(y_train,overall_pred))
    overall_pred = overallPred(x_test, y_test, models)
    print("F-1 Score of OneVsOne using svm for test data:", f1_score(y_test, overall_pred, average='macro'))
    print("Accuracy of oneVsone using svm for test data:", accuracy_score(y_test, overall_pred))





def binerisation(y,l):

    if l==0:
        for i in range(len(y)):
            if y[i]==1:
                y[i]=0
            else:
                y[i]=1
    if l==1:
        for i in range(len(y)):
            if y[i] == 0:
                y[i] = 0
            else:
                y[i] = 1
    if l==2:
        for i in range(len(y)):
            if y[i] == 0:
                y[i] = 1
            else:
                y[i] = 0


    return y
def plotRoc(y,score,l):#--------------Ref to Roc curve plot
    y=binerisation(y,l)
    fpr, tpr, threshold = metrics.roc_curve(y, score)
    roc_auc = metrics.auc(fpr, tpr)


    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.8f' % roc_auc)
    plt.legend(loc='lower right')


    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def classWiseAccuracy_ovo(dataframe,models):
    OverallAccuracy(models, dataframe)
    x_train, x_test, y_train, y_test = revisedData_ovo(dataframe, 0)
    x = np.array(x_train)
    y = np.array(y_train)
    predictions_class0,Score_class0 =Predict1(x,models[0],0)


    print("Class 0 train data accuracy",accuracy_score(y,predictions_class0))

    plotRoc(y, Score_class0, 0)
    x_train, x_test, y_train, y_test = revisedData_ovo(dataframe, 1)
    x = np.array(x_train)
    y = np.array(y_train)
    predictions_class1,Score_class1  = Predict1(x, models[1],1)
    print("Class 1 train data accuracy",accuracy_score(y, predictions_class1))


    plotRoc(y, Score_class1, 1)
    x_train, x_test, y_train, y_test = revisedData_ovo(dataframe, 2)
    x = np.array(x_train)
    y = np.array(y_train)
    predictions_class2,Score_class2  = Predict1(x, models[2],2)
    print("Class 2 train data accuracy",accuracy_score(y, predictions_class2))

    plotRoc(y, Score_class2, 1)

    x_train, x_test, y_train, y_test = revisedData_ovo(dataframe, 0)

    x = np.array(x_test)
    y = np.array(y_test)
    predictions_class0,Score_class0 = Predict1(x, models[0], 0)
    print("Class 0 test data accuracy", accuracy_score(y, predictions_class0))
    y1 = np.copy(y)
    plotRoc(y1, Score_class0, 0)

    x_train, x_test, y_train, y_test = revisedData_ovo(dataframe, 1)
    x = np.array(x_test)
    y = np.array(y_test)
    predictions_class1,Score_class1 = Predict1(x, models[1], 1)
    print("Class 1 test data accuracy", accuracy_score(y, predictions_class1))
    y1 = np.copy(y)
    plotRoc(y1, Score_class1, 1)
    x_train, x_test, y_train, y_test = revisedData_ovo(dataframe, 2)
    x = np.array(x_test)
    y = np.array(y_test)
    predictions_class2,Score_class2 = Predict1(x, models[2], 2)
    print("Class 2 test data accuracy", accuracy_score(y, predictions_class2))
    y1=np.copy(y)
    plotRoc(y1, Score_class2, 1)





def SVMOneVsOne():
    wine = load_wine()
    dataframe = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names'] + ['target'])

    models=[]
    s_time = time.time()
    for i in range(3):
        model = svm.SVC(kernel='linear', probability=True)

        x_train,x_test,y_train,y_test = revisedData_ovo(dataframe,i)

        models.append(model.fit(x_train, y_train))

    e_time = time.time()

    classWiseAccuracy_ovo(dataframe,models)
    print("Time taken to fit SVM One-vs-One in seconds :%f" % (e_time - s_time))


# --------------------------------------------NiaveBayes--------------------------


def GaussianNiaveBayes(x_train, x_test, y_train, y_test):
    GNB=GaussianNB()
    s_time = time.time()
    GNB.fit(x_train,y_train)
    e_time = time.time()
    pred_train = GNB.predict(x_train)
    pred_test = GNB.predict(x_test)
    print("Train data Accuracy of Gaussian Niave Bayes:", metrics.accuracy_score(y_train, pred_train))
    print("Train data F-1 Score of Gaussian Niave Bayes:", f1_score(y_train, pred_train, average='macro'))
    print("Test data Accuracy of Gaussian Niave Bayes:", metrics.accuracy_score(y_test, pred_test))
    print("Test data F-1 Score of Gaussian Niave Bayes:",f1_score(y_test, pred_test, average='macro'))
    cms = confusion_matrix(y_train, pred_train)
    cm = cms.astype('float') / cms.sum(axis=1)[:, np.newaxis]
    val = cm.diagonal()
    print("Train data Accuracy of Class 0 of Gaussian Niave Bayes:%f", val[0])
    print("Train data Accuracy of Class 1 of Gaussian Niave Bayes:%f", val[1])
    print("Train data Accuracy of Class 2 of Gaussian Niave Bayes:%f", val[2])
    cms = confusion_matrix(y_test, pred_test)
    cm = cms.astype('float') / cms.sum(axis=1)[:, np.newaxis]
    val=cm.diagonal()
    print("Test data Accuracy of Class 0 of Gaussian Niave Bayes:%f",val[0])
    print("Test data Accuracy of Class 1 of Gaussian Niave Bayes:%f", val[1])
    print("Test data Accuracy of Class 2 of Gaussian Niave Bayes:%f", val[2])
    print("Time taken to fit GaussianNiaveBayes model in seconds :%f" % (e_time - s_time))
    y_score = GNB.predict_proba(x_train)
    skplt.metrics.plot_roc(y_train, y_score, title="ROC Curve For GaussianNiaveBayes")
    plt.show()
    y_score = GNB.predict_proba(x_test)
    skplt.metrics.plot_roc(y_test, y_score,title="ROC Curve For GaussianNiaveBayes")
    plt.show()




# -------------------------------------------DecisionTree and tune the hyper-parameters--------------
def bestParameter(x_train,y_train):
    param_grid = {'max_depth': [30,40,50, 70, 80, 100],'max_features': [1,2, 3],'min_samples_leaf': [2, 4, 6],'min_samples_split': [7, 11, 13],}
    rf = tree.DecisionTreeClassifier()
    grid = GridSearchCV(estimator=rf, param_grid=param_grid,cv=3, n_jobs=-1, verbose=2)
    grid.fit(x_train, y_train)

    return grid.best_params_

def DecisionTree(x_train, x_test, y_train, y_test):
    bestParams=bestParameter(x_train,y_train)
    print(bestParams)
    s_time=time.time()
    model=tree.DecisionTreeClassifier(max_depth=bestParams['max_depth'],max_features=bestParams['max_features'],min_samples_leaf=bestParams['min_samples_leaf'],min_samples_split=bestParams['min_samples_split'])
    e_time=time.time()
    model.fit(x_train,y_train)
    pred_train = model.predict(x_train)
    print("Train data Accuracy of Decision Tree Classifier:", metrics.accuracy_score(y_train, pred_train))
    print("Train data F-1 Score of Decision Tree Classifier:", f1_score(y_train, pred_train, average='macro'))
    pred_test=model.predict(x_test)
    print("Test data Accuracy of Decision Tree Classifier:", metrics.accuracy_score(y_test, pred_test))
    print("Test data F-1 Score of Decision Tree Classifier:", f1_score(y_test, pred_test, average='macro'))
    cms = confusion_matrix(y_train, pred_train)
    cm = cms.astype('float') / cms.sum(axis=1)[:, np.newaxis]
    val = cm.diagonal()
    print("Train data Accuracy of Class 0 of DecisionTree:", val[0])
    print("Train data Accuracy of Class 1 of DecisionTree:", val[1])
    print("Train data Accuracy of Class 2 of DecisionTree:", val[2])
    cms = confusion_matrix(y_test, pred_test)
    cm = cms.astype('float') / cms.sum(axis=1)[:, np.newaxis]
    val = cm.diagonal()
    print("Test data Accuracy of Class 0 of DecisionTree:", val[0])
    print("Test data Accuracy of Class 1 of DecisionTree:", val[1])
    print("Test data Accuracy of Class 2 of DecisionTree:", val[2])
    print("Time taken to fit DecisionTree model in seconds :%f"%(e_time-s_time))
    y_score = model.predict_proba(x_train)
    skplt.metrics.plot_roc(y_train, y_score, title="ROC Curve For DecisionTree")
    plt.show()
    y_score=model.predict_proba(x_test)
    skplt.metrics.plot_roc(y_test, y_score,title="ROC Curve For DecisionTree")
    plt.show()





# ----------------------------------------------------------------------------------------------------------------------

features, target = load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(features, target,test_size=0.30,random_state=42)

print("Shape of x_train: ",x_train.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of x_test: ",x_test.shape)
print("Shape of y_test: ",y_test.shape)

PairwiseRelationPlot()
SVMOneVsRest(x_train, x_test, y_train, y_test)
SVMOneVsOne()
GaussianNiaveBayes(x_train, x_test, y_train, y_test)
DecisionTree(x_train, x_test, y_train, y_test)



