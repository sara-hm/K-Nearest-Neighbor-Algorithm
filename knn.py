import pandas as pd
import numpy as np
from scipy import stats as st
import time
import matplotlib.pyplot as plt

def loadData(filename):
    # Load data from file into X
    X = []
    count = 0
    
    text_file = open(filename, "r")
    lines = text_file.readlines()
        
    for line in lines:
        X.append([])
        words = line.split(",")
        # Convert values of the first attribute into float
        for word in words:
            if (word=='M'):
                word = 0.333
            if (word=='F'):
                word = 0.666
            if (word=='I'):
                word = 1
            X[count].append(float(word))
        count += 1
    
    return np.asarray(X)

# Basic check to see if any pre-processing steps are needed
X = loadData('abalone.data')
# creating a list of column names
column_values = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

# creating the dataframe
df = pd.DataFrame(data = X, columns = column_values)
  
# displaying the info and basic statistics of the dataframe for basic checks like missing values
print(df.describe())
print(df.info())
# no missing data and datatype in float


def testNorm(X_norm):
    # To verify the mean and sum of each attribute
    xMerged = np.copy(X_norm[0])
    # Merge datasets
    for i in range(len(X_norm)-1):
        xMerged = np.concatenate((xMerged,X_norm[i+1]))
    print(np.mean(xMerged,axis=0))
    print(np.sum(xMerged,axis=0))


def dataNorm(X):
    # Normalize the dataset
    X_norm = np.array(X)    # initializes array X_norm with the same values as array x
    for i in range(X.shape[1]-1):   # updates X_norm with the 8 normalized input attributes
        X_norm[:,i] = (X_norm[:,i]-np.min(X[:,i]))/(np.max(X[:,i])-np.min(X[:,i]))
    return X_norm

# To verify the mean and sum of each attribute after normalization
# Un-comment below 3 lines of code to check
# X = loadData('abalone.data')
# X_norm = dataNorm(X)
# testNorm([X_norm])


def splitTT(X_norm, percentTrain):
    # Split the dataset into training and testing using the train-and-test split method
    percentTrain = float(percentTrain)
    if percentTrain > 1 or percentTrain < 0:    #checks that expected portion of dataset for training the domain is (0..1)
        print("Invalid training split size")
    else:
        np.random.shuffle(X_norm)
        X_train = X_norm[:round(percentTrain * X_norm.shape[0]), :]
        X_test = X_norm[round(percentTrain * X_norm.shape[0]):, ]
        X_split = [X_train, X_test]
        return X_split

# To verify the mean and sum of each attribute after train-and-test split
# Un-comment below 4 lines of code to check
# X = loadData('abalone.data')
# X_norm = dataNorm(X)
# X_split = splitTT(X_norm, 0.6)
# testNorm(X_split)


def splitCV(X_norm, k):
    # Split the dataset into training and testing using the k-fold cross-validation method
    np.random.shuffle(X_norm)
    X_split = []
    data_per_ele = X_norm.shape[0]//k   # no. of data per k element
    for i in range(k):
        if i < (k-1):
            X_split.append(X_norm[(i*data_per_ele):((i+1)*data_per_ele), :])
        else:
            X_split.append(X_norm[((k-1)*data_per_ele):, :])   
    return X_split

# To verify the mean and sum of each attribute after k-fold cross-validation split
# Un-comment below 4 lines of code to check
# X = loadData('abalone.data')
# X_norm = dataNorm(X)
# X_split = splitCV(X_norm, 5)  # test with 5-fold
# testNorm(X_split)


def knn(X_train, X_test,k):
    # KNN algorithm using Euclidean distance as the similarity measure for any two samples
    correct_val = 0     # initializes the count of correctly predicted values

    for i in range(X_test.shape[0]):
        Eu_dist = np.zeros((X_train.shape[0],2))    # initializing empty array to store Euclidean distance between test data pt and training data pt
        Eu_dist[:,0] = np.sum((X_train[:,:-1]-X_test[i,:-1])**2,axis=1)**0.5    # calculating Euclidean distance for each training data pt per test data pt and storing it
        Eu_dist[:,-1] = X_train[:,-1]   # storing output of each training data pt
        Eu_dist = Eu_dist[Eu_dist[:, 0].argsort()]  # sorting the Euclidean distance in ascending order
        mode, counts = st.mode(Eu_dist[:k,-1])  # finding the mode of nearest k neighbors
        prediction = mode[0]
        if X_test[i,-1] == prediction:
            correct_val = correct_val + 1   # finding the number of accurate predictions per test data pt

    accuracy = correct_val/X_test.shape[0]*100  # returns the accuracy as %

    return accuracy


# The main KNN function with train-and-test + Euclidean
def knnMain(filename,percentTrain,k):
    
    # Data load
    X = loadData(filename)
    # Normalization
    X_norm = dataNorm(X)
    # Data split: train-and-test
    X_split = splitTT(X_norm,percentTrain)
    # KNN: Euclidean
    accuracy = knn(X_split[0],X_split[1],k)
    
    return accuracy


# Below codes are run to compare the classification performance of the KNN() algorithm for different values of K using train-and-test + Euclidean
K = [1,5,10,15,20]  # initializing different values of K
# For Train-and-Test 0.7 - 0.3
ac_y_tt_7 = []  # stores the accuracy score
ct_y_tt_7 = []  # stores the computational time
for i in K:
    start_time = time.perf_counter()
    accuracy = knnMain('abalone.data',0.7,i)
    stop_time = time.perf_counter()
    time_elapsed = stop_time - start_time
    ac_y_tt_7.append(accuracy)
    ct_y_tt_7.append(time_elapsed)
    print(f"Accuracy score for Train-and-Test 0.7 - 0.3 and K = {i} is {accuracy:0.1f}%")
    print(f"Computational time for Train-and-Test 0.7 - 0.3 and K = {i} is {time_elapsed:0.3f} seconds")

# For Train-and-Test 0.6 – 0.4
ac_y_tt_6 = []  # stores the accuracy score
ct_y_tt_6 = []  # stores the computational time
for i in K:
    start_time = time.perf_counter()
    accuracy = knnMain('abalone.data',0.6,i)
    stop_time = time.perf_counter()
    time_elapsed = stop_time - start_time
    ac_y_tt_6.append(accuracy)
    ct_y_tt_6.append(time_elapsed)
    print(f"Accuracy score for Train-and-Test 0.6 – 0.4 and K = {i} is {accuracy:0.1f}%")
    print(f"Computational time for Train-and-Test 0.6 – 0.4 and K = {i} is {time_elapsed:0.3f} seconds")

# For Train-and-Test 0.5 - 0.5
ac_y_tt_5 = []  # stores the accuracy score
ct_y_tt_5 = []  # stores the computational time
for i in K:
    start_time = time.perf_counter()
    accuracy = knnMain('abalone.data',0.5,i)
    stop_time = time.perf_counter()
    time_elapsed = stop_time - start_time
    ac_y_tt_5.append(accuracy)
    ct_y_tt_5.append(time_elapsed)
    print(f"Accuracy score for Train-and-Test 0.5 - 0.5 and K = {i} is {accuracy:0.1f}%")
    print(f"Computational time for Train-and-Test 0.5 - 0.5 and K = {i} is {time_elapsed:0.3f} seconds")


# The main KNN function with k-fold cross-validation + Euclidean
def knnMain(filename,k_fold,k): 
    # Data load
    X = loadData(filename)
    # Normalization
    X_norm = dataNorm(X)
    # Data split: k-fold cross-validation
    X_split = splitCV(X_norm,k_fold)
    # KNN: Euclidean
    accuracy = []
    for i in range(k_fold):
        X_split_new = X_split[:]
        X_test = X_split_new.pop(i)
        X_train = np.vstack(X_split_new)
        accuracy.append(knn(X_train,X_test,k))
    
    accuracy = sum(accuracy) / k_fold   # getting the average accuracy from the k-fold runs
    
    return accuracy


# Below codes are run to compare the classification performance of the KNN() algorithm for different values of K using k-fold cross-validation + Euclidean
K = [1,5,10,15,20]  # initializing different values of K
# For 5-fold Cross-Validation
ac_y_5fold = []  # stores the accuracy score
ct_y_5fold = []  # stores the computational time
for i in K:
    start_time = time.perf_counter()
    accuracy = knnMain('abalone.data',5,i)
    stop_time = time.perf_counter()
    time_elapsed = stop_time - start_time
    ac_y_5fold.append(accuracy)
    ct_y_5fold.append(time_elapsed)
    print(f"Accuracy score for 5-fold Cross-Validation and K = {i} is {accuracy:0.1f}%")
    print(f"Computational time for 5-fold Cross-Validation and K = {i} is {time_elapsed:0.3f} seconds")

# For 10-fold Cross-Validation
ac_y_10fold = []  # stores the accuracy score
ct_y_10fold = []  # stores the computational time
for i in K:
    start_time = time.perf_counter()
    accuracy = knnMain('abalone.data',10,i)
    stop_time = time.perf_counter()
    time_elapsed = stop_time - start_time
    ac_y_10fold.append(accuracy)
    ct_y_10fold.append(time_elapsed)
    print(f"Accuracy score for 10-fold Cross-Validation and K = {i} is {accuracy:0.1f}%")
    print(f"Computational time for 10-fold Cross-Validation and K = {i} is {time_elapsed:0.3f} seconds")

# For 15-fold Cross-Validation
ac_y_15fold = []  # stores the accuracy score
ct_y_15fold = []  # stores the computational time
for i in K:
    start_time = time.perf_counter()
    accuracy = knnMain('abalone.data',15,i)
    stop_time = time.perf_counter()
    time_elapsed = stop_time - start_time
    ac_y_15fold.append(accuracy)
    ct_y_15fold.append(time_elapsed)
    print(f"Accuracy score for 15-fold Cross-Validation and K = {i} is {accuracy:0.1f}%")
    print(f"Computational time for 15-fold Cross-Validation and K = {i} is {time_elapsed:0.3f} seconds")


# To plot the accuracy score for each of the experiments
# Please un-comment to run plots

# K_axis = [1, 5, 10, 15, 20]
# plt.figure(figsize=(10, 6))  
# plt.plot(K_axis, ac_y_tt_7, color = 'blue', label = 'Train-and-Test 0.7 - 0.3')
# plt.plot(K_axis, ac_y_tt_6, color = '#F77538', label = 'Train-and-Test 0.6 – 0.4')
# plt.plot(K_axis, ac_y_tt_5, color = 'green', label = 'Train-and-Test 0.5 - 0.5')
# plt.plot(K_axis, ac_y_5fold, color = 'blue', linestyle = '--', label = '5-fold Cross-Validation')
# plt.plot(K_axis, ac_y_10fold, color = '#F77538', linestyle = '--', label = '10-fold Cross-Validation')
# plt.plot(K_axis, ac_y_15fold, color = 'green', linestyle = '--', label = '15-fold Cross-Validation')

# plt.xticks(np.arange(0, 21, 5.0))
# plt.ylim(0,40)
# plt.legend()
# plt.title("Accuracy score comparison")
# plt.xlabel("K")
# plt.ylabel("Accuracy score (%)")
# plt.show()

# To plot the computational time for each of the experiments
# Please un-comment to run plots

# K_axis = [1, 5, 10, 15, 20]
# plt.figure(figsize=(10, 6))  
# plt.plot(K_axis, ct_y_tt_7, color = 'blue', label = 'Train-and-Test 0.7 - 0.3')
# plt.plot(K_axis, ct_y_tt_6, color = '#F77538', label = 'Train-and-Test 0.6 – 0.4')
# plt.plot(K_axis, ct_y_tt_5, color = 'green', label = 'Train-and-Test 0.5 - 0.5')
# plt.plot(K_axis, ct_y_5fold, color = 'blue', linestyle = '--', label = '5-fold Cross-Validation')
# plt.plot(K_axis, ct_y_10fold, color = '#F77538', linestyle = '--', label = '10-fold Cross-Validation')
# plt.plot(K_axis, ct_y_15fold, color = 'green', linestyle = '--', label = '15-fold Cross-Validation')

# plt.xticks(np.arange(0, 21, 5.0))
# plt.ylim(0,3)
# plt.legend()
# plt.title("Computational time comparison")
# plt.xlabel("K")
# plt.ylabel("Computational time (s)")
# plt.show()


# Part e. 
# Using the classification_report() function provided by the scikit-learn library 
# to construct a classification report for the 5-fold cross validation with K = 15

# Method 1: 5-fold cross validation split using previously defined splitCV
from sklearn.metrics import classification_report

def knn_parte(X_train, X_test,k):
    # Modified KNN algorithm using Euclidean distance as the similarity measure for any two samples
    # return prediction and y_test as output
    prediction = []
    for i in range(X_test.shape[0]):
        Eu_dist = np.zeros((X_train.shape[0],2))    # initializing empty array to store Euclidean distance between test data pt and training data pt
        Eu_dist[:,0] = np.sum((X_train[:,:-1]-X_test[i,:-1])**2,axis=1)**0.5    # calculating Euclidean distance for each training data pt per test data pt and storing it
        Eu_dist[:,-1] = X_train[:,-1]   # storing output of each training data pt
        Eu_dist = Eu_dist[Eu_dist[:, 0].argsort()]  # sorting the Euclidean distance in ascending order
        mode, counts = st.mode(Eu_dist[:k,-1])  # finding the mode of nearest k neighbors
        prediction.append(mode[0])
        
    prediction = np.array(prediction)
    y_test = X_test[:,-1]

    return prediction, y_test

X = loadData('abalone.data')
X_norm = dataNorm(X)
X_split = splitCV(X_norm,5) # 5-fold cross validation

y_test_combined = []    # storing the actual test results for each fold of test samples
pre_combined = []   # storing the predicted test results for each fold of test samples

for i in range(5):  # 5-fold cross validation
    X_split_new = X_split[:]
    X_test = X_split_new.pop(i)
    X_train = np.vstack(X_split_new)
    prediction, y_test = knn_parte(X_train,X_test,15)   # K = 15
    pre_combined.append(prediction)
    y_test_combined.append(y_test)

y_test_combined = np.concatenate(y_test_combined, axis = 0)
pre_combined = np.concatenate(pre_combined, axis = 0)

print("Classification_report using splitCV:")
print(classification_report(y_test_combined, pre_combined))


# Method 2: Using KFold to split the data
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

X = loadData('abalone.data')
X_norm = dataNorm(X)
X_cl_data = X_norm[:,:-1]
y_cl_data = X_norm[:,-1]
kf = KFold(n_splits=5)  # 5-fold cross validation
knn = KNeighborsClassifier(n_neighbors=15)  # K = 15

y_test_combined = []    # storing the actual test results for each fold of test samples
pre_combined = []   # storing the predicted test results for each fold of test samples

for train_index, test_index in kf.split(X_cl_data):
    X_train, X_test = X_cl_data[train_index], X_cl_data[test_index]
    y_train, y_test = y_cl_data[train_index], y_cl_data[test_index]
    knn.fit(X_train, y_train)
    pre = knn.predict(X_test)
    pre_combined.append(pre)
    y_test_combined.append(y_test)

y_test_combined = np.concatenate(y_test_combined, axis = 0)
pre_combined = np.concatenate(pre_combined, axis = 0)

print("Classification_report using KFold:")
print(classification_report(y_test_combined, pre_combined))
