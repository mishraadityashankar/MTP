import numpy as np
from collections import Counter
def Implicit_ME(X_train,y_train,X_test,y_test,clustSize,experts,gate_classifier):
 
    arr_X_train = np.array_split(X_train, clustSize)
    arr_y_train = np.array_split(y_train, clustSize)
    bestLocalExperts = []
    pred_y_train_clust = []
    pred_y_test_clust = []
    X_train_new = X_train
    X_test_new = X_test
    count = 0
    
    for i in range(clustSize):
#         clust_X_train, clust_X_test, clust_y_train, clust_y_test = train_test_split(arr_X_train[i],arr_y_train[i], test_size=0.2,random_state=1)
        clust_X = arr_X_train[i]
        clust_y = arr_y_train[i]
        #index of best classifer by default 0
        best_expert = 0
        best_score = -2
        counter1 = Counter(clust_y)
        if(len(counter1)>1):
            for j in range(len(experts)):
                local_result = experts[j](clust_X,clust_y,clust_X,clust_y)
                if(local_result["MCC"] > best_score):
                    best_expert = j
                    best_score = local_result["MCC"]

        bestLocalExperts.append(best_expert)
            
    #taking output of train and test data from local expert 
    for i in range(clustSize):
        train_result = experts[bestLocalExperts[i]](arr_X_train[i],arr_y_train[i],X_train,y_train)
        test_result =  experts[bestLocalExperts[i]](arr_X_train[i],arr_y_train[i],X_test,y_test)
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
    
    
    res =  gate_classifier(X_train_new,y_train,X_test_new,y_test)
    return res
    
    