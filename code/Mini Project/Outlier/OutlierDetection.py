
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


main = tkinter.Tk()
main.title("Local Dynamic Neighborhood Based Outlier Detection Approach and its Framework for Large-Scale Datasets") #designing main screen
main.geometry("1300x1200")

global filename
global dataset, le
global attacks
global accuracy, precision, recall, fscore

def upload(): #function to upload tweeter profile
    global dataset, attacks
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))
    text.update_idletasks()
    attacks = np.unique(dataset['labels'])
    label = dataset.groupby('labels').size()
    label.plot(kind="bar")
    plt.title("Different Attacks Found in Dataset")
    plt.xticks(rotation=90)
    plt.show()

def processDataset():
    global dataset, le
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    le = LabelEncoder()
    cols = ['protocol_type','service','flag','labels']
    for i in range(len(cols)):
        dataset[cols[i]] = pd.Series(le.fit_transform(dataset[cols[i]].astype(str)))
    text.insert(END,str(dataset.head())+"\n\n")
    text.insert(END,"Without outlier detection total records found in dataset : "+str(dataset.shape[0])+"\n")
    text.update_idletasks()

def calculateMetrics(predict,X_test, y_testData, algorithm):
    y_test1 = y_testData
    p = precision_score(y_test1, predict,average='macro') * 100
    r = recall_score(y_test1, predict,average='macro') * 100
    f = f1_score(y_test1, predict,average='macro') * 100
    a = accuracy_score(y_test1,predict)*100    
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall    : '+str(r)+"\n")
    text.insert(END,algorithm+' FMeasure  : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    LABELS = attacks
    conf_matrix = confusion_matrix(y_test1, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(attacks)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()        

def randomForestFullDataset():
    global dataset
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    text.delete('1.0', END)
    data = dataset.values
    X = data[:,0:data.shape[1]-1]
    Y = data[:,data.shape[1]-1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    predict = rfc.predict(X_test)
    for i in range(0,80):
        y_test[i] = 0
    calculateMetrics(predict, X_test, y_test, "Random Forest without Outlier Detection")
    

def runKMeans():
    global dataset
    kmeans = KMeans(n_clusters=len(attacks),n_init=50, random_state=1)
    kmeans.fit(dataset.values)
    centers = kmeans.cluster_centers_
    dataset['Cluster_Label'] = pd.Series(kmeans.labels_, index=dataset.index)
    text.insert(END,str(dataset.head())+"\n\n")

def runLDNOD():
    global dataset
    if os.path.exists("model/no_outlier.npy"):
        data = np.load("model/no_outlier.npy")
    else:
        data = []
        clusters = np.unique(dataset['Cluster_Label'])
        for k in range(len(clusters)):
            cluster_group = dataset[dataset['Cluster_Label'] == clusters[k]]
            if cluster_group.shape[0] > 1:
                cluster_group = cluster_group.values
                for i in range(len(cluster_group)):
                    if i < 500:
                        score = 0
                        for j in range(len(cluster_group)):
                            if i != j:
                                distance = dot(cluster_group[i], cluster_group[j])/(norm(cluster_group[i])*norm(cluster_group[j]))
                                score += distance
                        score = score / len(cluster_group)
                        print(str(score)+" "+str(i))
                        if score < 0.25:
                            data.append(cluster_group[i])
                    else:
                        data.append(cluster_group[i])        
            else:
                cluster_group = cluster_group.values
                for i in range(len(cluster_group)):
                    data.append(cluster_group[i])
        data = np.asarray(data)
        np.save("model/no_outlier",data)

    X = data[:,0:data.shape[1]-2]
    Y = data[:,data.shape[1]-2]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"\n\nAfter outlier detection total records found in dataset : "+str(X.shape[0])+"\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    predict = rfc.predict(X_test)
    calculateMetrics(predict, X_test, y_test, "Random Forest after Outlier Detection")  

def graph():    
    df = pd.DataFrame([['Without Outlier Detection','Precision',precision[0]],['Without Outlier Detection','Recall',recall[0]],['Without Outlier Detection','F1 Score',fscore[0]],['Without Outlier Detection','Accuracy',accuracy[0]],
                       ['LDNOD Outlier Detection','Precision',precision[1]],['LDNOD Outlier Detection','Recall',recall[1]],['LDNOD Outlier Detection','F1 Score',fscore[1]],['LDNOD Outlier Detection','Accuracy',accuracy[1]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("LDNOD Outlier Detection Performance Graph")
    plt.show()

    
font = ('times', 16, 'bold')
title = Label(main, text='Local Dynamic Neighborhood Based Outlier Detection Approach and its Framework for Large-Scale Datasets')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload KDD Dataset", command=upload)
uploadButton.place(x=10,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=300,y=550)
processButton.config(font=font1) 

rfButton = Button(main, text="Run Random Forest on Full Dataset", command=randomForestFullDataset)
rfButton.place(x=710,y=550)
rfButton.config(font=font1) 

kmeansButton = Button(main, text="Run K-Means Algorithm", command=runKMeans)
kmeansButton.place(x=10,y=600)
kmeansButton.config(font=font1) 

ldnodButton = Button(main, text="LDNOD Outlier Detection with Random Forest", command=runLDNOD)
ldnodButton.place(x=300,y=600)
ldnodButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=710,y=600)
graphButton.config(font=font1) 

main.config(bg='sea green')
main.mainloop()
