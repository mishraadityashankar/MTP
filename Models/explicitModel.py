import numpy as np
from collections import Counter
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans, splitting_type

def Explicit_ME(X_train,y_train,X_test,y_test,experts,gate_classifier):

    initial_centers = kmeans_plusplus_initializer(X_train, 1).initialize();
    xmeans_instance = xmeans(X_train, initial_centers);
    xmeans_instance.process();
    clusters = xmeans_instance.get_clusters();
    clustX = []
    clusty = []
    final_clustX = []
    final_clustY = []

    arrX= np.array(X_train)
    arrY= np.array(y_train)
    
    for i in range(len(clusters)):
        tempx = []
        tempy = []
        for j in range(len(clusters[i])):
            tempx.append(arrX[clusters[i][j]])
            tempy.append(arrY[clusters[i][j]])
        clustX.append(tempx)
        clusty.append(tempy)
    
    for i in range(len(clustX)):
        counter1 = Counter(clusty[i])
        if(len(counter1)>1):
            final_clustX.append(clustX[i])
            final_clustY.append(clusty[i])
            
    bestLocalExperts = []
    pred_y_train_clust = []
    pred_y_test_clust = []
    X_train_new = X_train
    X_test_new = X_test
    count = 0
    clustSize = len(final_clustX)
    
    for i in range(clustSize):
#         clust_X_train, clust_X_test, clust_y_train, clust_y_test = train_test_split(arr_X_train[i],arr_y_train[i], test_size=0.2,random_state=1)
        clust_X = final_clustX[i]
        clust_y = final_clustY[i]
        #index of best classifer by default 0
        best_expert = 0
        best_score = -2
        for j in range(len(experts)):
            local_result = experts[j](clust_X,clust_y,clust_X,clust_y)
            if(local_result["MCC"] > best_score):
                best_expert = j
                best_score = local_result["MCC"]

        bestLocalExperts.append(best_expert)
      
    #taking output of train and test data from local expert 
    for i in range(clustSize):
        train_result = experts[bestLocalExperts[i]](final_clustX[i],final_clustY[i],X_train,y_train)
        test_result =  experts[bestLocalExperts[i]](final_clustX[i],final_clustY[i],X_test,y_test)
        pred_y_train_clust.append(train_result["Prediction"])
        pred_y_test_clust.append(test_result["Prediction"])


    #adding outputs of local as features to gate
    for data in X_train_new:
        for i in range(clustSize):
            np.append(data,pred_y_train_clust[i][count])
        count = count + 1
    count = 0
    for data in X_test_new:
        for i in range(clustSize):
            np.append(data,pred_y_test_clust[i][count])
        count = count + 1
    
    
    #training gate
    res =  gate_classifier(X_train_new,y_train,X_test_new,y_test)
    return res


