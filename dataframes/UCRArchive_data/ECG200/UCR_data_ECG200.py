import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

 
# Read the data from the CSV file
ECG200_test = pd.read_csv('/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/ECG200/ECG200_TEST.tsv', sep='\t', header=None)
print(ECG200_test.head())
print(ECG200_test.tail())
print(f"ECG200_test data size: {ECG200_test.shape}")
print(ECG200_test.info())
length = ECG200_test.shape[1]
print(f"length: {length -1}")

# labels in the first column
# print test first 10 rows, 0 column
print(ECG200_test.iloc[0:10, 0])

# plot the first row of the dataframe wihtout the first column
# use the first column to label the plot
plt.figure(figsize=(10, 5));
x = np.linspace(0, 1, length -1)
plt.plot(x, ECG200_test.iloc[0, 1:])
plt.title(f"ECG200_test label: {ECG200_test.iloc[0, 0]}")
plt.show()


plt.figure(figsize=(10, 5));
x = np.linspace(0, 1, length -1)
plt.plot(x, ECG200_test.iloc[4, 1:])
plt.title(f"ECG200_test label: {ECG200_test.iloc[4, 0]}")
plt.show()



# how many classes are there
print(ECG200_test[0].unique())
# how many classes are there in total
print(ECG200_test[0].value_counts())

# what are the rows for class 1
print(ECG200_test[ECG200_test[0] == 1].index)

list_class_1 = list(ECG200_test[ECG200_test[0] == 1].index)
print(list_class_1)
list_class_1 = list_class_1[0:4]
print(list_class_1)

list_class_2 = list(ECG200_test[ECG200_test[0] == -1].index)
print(list_class_2)
list_class_2 = list_class_2[0:4]
print(list_class_2)


# create plots, one for each label and plot
# the first 4 signals for each label
indices = [list_class_1, list_class_2]
print(indices)
print(len(indices))

# Create subplots (2 rows, 2 columns) 
fig, axes = plt.subplots(1, 2, figsize=(13, 8))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

# Flatten axes array
axes = axes.flatten()

# Loop through indices and plot data
for i, idx_list in enumerate(indices):
    for idx in idx_list:
        axes[i].plot(x, ECG200_test.iloc[idx, 1:])

    # Set title
    axes[i].set_title(f"ECG200_test labels: {', '.join(str(ECG200_test.iloc[idx, 0]) for idx in idx_list)}")
    #axes[i].legend()


plt.show();





# find index for each Beef_test[0].unique()
for label in ECG200_test[0].unique():
    print(f" For class {label}: {ECG200_test[ECG200_test[0] == label].index}")


# find the count of each class in the 0th column, i.e. how many signals per class are there
for label in ECG200_test[0].value_counts().index:
    print(f"Class {label} has {ECG200_test[0].value_counts()[label]} signals")

# change -1 to 2
ECG200_test[0] = ECG200_test[0].replace(-1, 2)
print(ECG200_test[0].value_counts())



####### Beef_test csv matrix for each test

# make csv's using labels and remove columne 0
for label in ECG200_test[0].unique():
    df =ECG200_test[ECG200_test[0] == label]
    #if label == -1:
    #    label = '2'
    #df = df.drop(df.columns[0], axis = 1)
    df = df.transpose()
    df.to_csv(f'ECG200{label}_test_matrix.csv', index=False, header=False) # keeping the labels in test files
    print(df.shape)




ECG2001_test_matrix = pd.read_csv('ECG2001_test_matrix.csv')
print(ECG2001_test_matrix.shape)
print(ECG2001_test_matrix.head())

ECG2002_test_matrix = pd.read_csv('ECG2002_test_matrix.csv')
print(ECG2002_test_matrix.shape)
print(ECG2002_test_matrix.head())



######## train csv matrix   


ECG200_train = pd.read_csv('/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/ECG200/ECG200_TRAIN.tsv', sep='\t', header=None)
print(ECG200_train.head())
print(f"ECG200_train data size: {ECG200_train.shape}")

# find index for each Bee_train[0].unique()
for label in ECG200_train[0].unique():
    print(f" For class {label}: {ECG200_train[ECG200_train[0] == label].index}")
list_class_1 = list(ECG200_train[ECG200_train[0] == 1].index)
print(list_class_1)

list_class_2 = list(ECG200_train[ECG200_train[0] == -1].index)
print(list_class_2)

# Plot all signals for class 1 on the same plot
plt.figure(figsize=(10, 5))
for idx in list_class_1:
    plt.plot(ECG200_train.iloc[idx, 1:])
plt.title("ECG200_train class 1 signals")
plt.show()



plt.figure(figsize=(10, 5))
for idx in list_class_2:
    plt.plot(ECG200_train.iloc[idx, 1:])
plt.title("ECG200_train class -1 signals")
plt.show()


# find the count of each class in the 0th column, i.e. how many 16's are there
for label in ECG200_train[0].value_counts().index:
    print(f"Class {label} has {ECG200_train[0].value_counts()[label]} signals")

# change -1 to 2
ECG200_train[0] = ECG200_train[0].replace(-1, 2)
print(ECG200_train[0].value_counts())


# sort the dataframe by the first column
ECG200_train_sorted = ECG200_train.sort_values(0)
print(ECG200_train_sorted.head(10))    
plt.figure(figsize=(10, 5))
for j in range(5):
    plt.plot(ECG200_train_sorted.iloc[j, 1:])
plt.title("CinCECGTorso_train class 1 signals")
plt.show()


path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_data/ECG200"
# transpose beef train tsv file
# can't make labels into integers in first row.
# the columns keep the same dtype 
# will deal with only for accruacy results
ECG200_train_transpose = ECG200_train_sorted.transpose() # this is the same as Beef_train.T 
print(ECG200_train_transpose) 
print(ECG200_train_transpose.shape)
print(ECG200_train_transpose.iloc[1:, 0:5]) # first 5 signals, excluding the first row and last row
print(ECG200_train_transpose.head(10))



# plot the first signal in CinCECGTorso_train_transpose
plt.figure(figsize=(10, 5))
plt.plot(ECG200_train_transpose.iloc[1:, 0:5])
plt.title("ECG200_train first signal")
plt.show()




# Save to CSV
ECG200_train_transpose.to_csv(os.path.join(path, 'ECG200_train_matrix.csv'), index=False, header=False) # index for row, header for column index
print(ECG200_train_transpose.shape)
print(ECG200_train_transpose.head())

ECG200_train_matrix = pd.read_csv('/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_data/ECG200/ECG200_train_matrix.csv')
print(ECG200_train_matrix.shape)
print(ECG200_train_matrix.head())
plt.figure(figsize=(10, 5))
plt.plot(ECG200_train_matrix.iloc[:, 0:5])
plt.title("ECG200_train first signal")
plt.show()





#####
"""

#OLDer way.  NEW way keeps data like the one in EKG.
# make one dataframe for train, remove column 0, make a csv
Beef_train_df = Beef_train.drop(Beef_train.columns[0], axis = 1)
Beef_train_df = Beef_train_df.transpose()
Beef_train_df.to_csv('Beef_train_matrix.csv', index=False)
print(Beef_train_df.shape)
print(Beef_train_df.head())
"""
###########