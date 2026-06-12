import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

 
# Read the data from the CSV file
ECG5000_test = pd.read_csv('/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/ECG5000/ECG5000_TEST.tsv', sep='\t', header=None)
print(ECG5000_test.head())
print(ECG5000_test.tail())
print(f"ECGFiveDaystest data size: {ECG5000_test.shape}")
print(ECG5000_test.info())
length = ECG5000_test.shape[1]
print(f"length: {length -1}")

# labels in the first column
# print test first 10 rows, 0 column
print(ECG5000_test.iloc[0:10, 0])

# plot the first row of the dataframe wihtout the first column
# use the first column to label the plot
plt.figure(figsize=(10, 5));
x = np.linspace(0, 1, length -1)
plt.plot(x, ECG5000_test.iloc[0, 1:])
plt.title(f"ECG5000_test label: {ECG5000_test.iloc[0, 0]}")
plt.show()


plt.figure(figsize=(10, 5));
x = np.linspace(0, 1, length -1)
plt.plot(x, ECG5000_test.iloc[4, 1:])
plt.title(f"ECG5000_test label: {ECG5000_test.iloc[4, 0]}")
plt.show()



# how many classes are there
print(ECG5000_test[0].unique())
# how many classes are there in total
print(ECG5000_test[0].value_counts())

# what are the rows for class 1
print(ECG5000_test[ECG5000_test[0] == 1].index)

list_class_1 = list(ECG5000_test[ECG5000_test[0] == 1].index)
print(list_class_1)
list_class_1 = list_class_1[0:4]
print(list_class_1)

list_class_2 = list(ECG5000_test[ECG5000_test[0] == 2].index)
print(list_class_2)
list_class_2 = list_class_2[0:4]
print(list_class_2)

list_class_3 = list(ECG5000_test[ECG5000_test[0] == 3].index)
print(list_class_3)
list_class_3 = list_class_3[0:4]
print(list_class_3)
list_class_4 = list(ECG5000_test[ECG5000_test[0] == 4].index)
print(list_class_4) 
list_class_4 = list_class_4[0:4]
print(list_class_4)
list_class_5 = list(ECG5000_test[ECG5000_test[0] == 5].index)
print(list_class_5)
list_class_5 = list_class_5[0:4]
print(list_class_5)


# create plots, one for each label and plot
# the first 4 signals for each label
indices = [list_class_1, list_class_2, list_class_3, list_class_4, list_class_5]
print(indices)
print(len(indices))

# Create subplots (2 rows, 2 columns) 
fig, axes = plt.subplots(3, 2, figsize=(13, 8))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

# Flatten axes array
axes = axes.flatten()

# Loop through indices and plot data
for i, idx_list in enumerate(indices):
    for idx in idx_list:
        axes[i].plot(x,ECG5000_test.iloc[idx, 1:])

    # Set title
    axes[i].set_title(f"ECGFiveDays_test labels: {', '.join(str(ECG5000_test.iloc[idx, 0]) for idx in idx_list)}")
    #axes[i].legend()

# remove the last figure as there are only 5 figures
fig.delaxes(axes[-1])
plt.show();





# find index for each test[0].unique()
for label in ECG5000_test[0].unique():
    print(f" For class {label}: {ECG5000_test[ECG5000_test[0] == label].index}")


# find the count of each class in the 0th column, i.e. how many signals per class are there
for label in ECG5000_test[0].value_counts().index:
    print(f"Class {label} has {ECG5000_test[0].value_counts()[label]} signals")



####### Beef_test csv matrix for each test

# make csv's using labels and remove columne 0
for label in ECG5000_test[0].unique():
    df =ECG5000_test[ECG5000_test[0] == label]
    #df = df.drop(df.columns[0], axis = 1)
    df = df.transpose()
    df.to_csv(f'ECG5000{label}_test_matrix.csv', index=False, header=False) # keeping the labels in test files
    print(df.shape)




ECG50001_test_matrix = pd.read_csv('ECG50001_test_matrix.csv')
print(ECG50001_test_matrix.shape)
print(ECG50001_test_matrix.head())

ECG50002_test_matrix = pd.read_csv('ECG50002_test_matrix.csv')
print(ECG50002_test_matrix.shape)
print(ECG50002_test_matrix.head())

ECG50003_test_matrix = pd.read_csv('ECG50003_test_matrix.csv')
print(ECG50003_test_matrix.shape)
print(ECG50003_test_matrix.head())


ECG50004_test_matrix = pd.read_csv('ECG50004_test_matrix.csv')
print(ECG50004_test_matrix.shape)
print(ECG50004_test_matrix.head())

ECG50005_test_matrix = pd.read_csv('ECG50005_test_matrix.csv')
print(ECG50005_test_matrix.shape)
print(ECG50005_test_matrix.head())




######## train csv matrix   


ECG5000_train = pd.read_csv('/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/ECG5000/ECG5000_TRAIN.tsv', sep='\t', header=None)
print(ECG5000_train.head())
print(f"ECG5000_train data size: {ECG5000_train.shape}")

# find index for each Bee_train[0].unique()
for label in ECG5000_train[0].unique():
    print(f" For class {label}: {ECG5000_train[ECG5000_train[0] == label].index}")
list_class_1 = list(ECG5000_train[ECG5000_train[0] == 1].index)
print(list_class_1)
print(len(list_class_1))

list_class_2 = list(ECG5000_train[ECG5000_train[0] == 2].index)
print(list_class_2)
print(len(list_class_2))

list_class_3 = list(ECG5000_train[ECG5000_train[0] == 3].index)
print(list_class_3)
print(len(list_class_3))

list_class_4 = list(ECG5000_train[ECG5000_train[0] == 4].index)
print(list_class_4)
print(len(list_class_4))

list_class_5 = list(ECG5000_train[ECG5000_train[0] == 5].index)
print(list_class_5)
print(len(list_class_5))

# Plot all signals for class 1 on the same plot
plt.figure(figsize=(10, 5))
for idx in list_class_1:
    plt.plot(ECG5000_train.iloc[idx, 1:])
plt.title("ECG5000_train class 1 signals")
plt.show()



plt.figure(figsize=(10, 5))
for idx in list_class_2:
    plt.plot(ECG5000_train.iloc[idx, 1:])
plt.title("ECG5000_train class 2 signals")
plt.show()

plt.figure(figsize=(10, 5))
for idx in list_class_3:
    plt.plot(ECG5000_train.iloc[idx, 1:])
plt.title("ECG5000_train class 3 signals")
plt.show()

plt.figure(figsize=(10, 5))
for idx in list_class_4:
    plt.plot(ECG5000_train.iloc[idx, 1:])
plt.title("ECG5000_train class 4 signals")
plt.show()

plt.figure(figsize=(10, 5))
for idx in list_class_5:
    plt.plot(ECG5000_train.iloc[idx, 1:])
plt.title("ECG5000_train class 5 signals")
plt.show()


# find the count of each class in the 0th column, i.e. how many 16's are there
for label in ECG5000_train[0].value_counts().index:
    print(f"Class {label} has {ECG5000_train[0].value_counts()[label]} signals")




# sort the dataframe by the first column
ECG5000_train_sorted = ECG5000_train.sort_values(0)
print(ECG5000_train_sorted.head(10))    
plt.figure(figsize=(10, 5))
for j in range(5):
    plt.plot(ECG5000_train_sorted.iloc[j, 1:])
plt.title("ECG5000_train class 1 signals")
plt.show()

#find 5 in first column
print(ECG5000_train_sorted[ECG5000_train_sorted[0] == 5].index)


path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_data/ECG5000"
# transpose beef train tsv file
# can't make labels into integers in first row.
# the columns keep the same dtype 
# will deal with only for accruacy results
ECG5000_train_transpose = ECG5000_train_sorted.transpose() # this is the same as Beef_train.T 
print(ECG5000_train_transpose) 
print(ECG5000_train_transpose.shape)
print(ECG5000_train_transpose.iloc[1:, 0:5]) # first 5 signals, excluding the first row and last row
print(ECG5000_train_transpose.head(10))
print(ECG5000_train_transpose.iloc[:, -1]) 
# print the last two columns
print(ECG5000_train_transpose.iloc[:, -2:]) 


# plot the first signal in CinCECGTorso_train_transpose
plt.figure(figsize=(10, 5))
plt.plot(ECG5000_train_transpose.iloc[1:, 0:5])
plt.title("ECG5000_train first signals")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(ECG5000_train_transpose.iloc[1:, -2:])
plt.title("ECG5000_train last signals")
plt.show()




# Save to CSV
ECG5000_train_transpose.to_csv(os.path.join(path, 'ECG5000_train_matrix.csv'), index=False, header=False) # index for row, header for column index
print(ECG5000_train_transpose.shape)
print(ECG5000_train_transpose.head())

ECG5000_train_matrix = pd.read_csv('/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_data/ECG5000/ECG5000_train_matrix.csv')
print(ECG5000_train_matrix.shape)
print(ECG5000_train_matrix.head())
plt.figure(figsize=(10, 5))
plt.plot(ECG5000_train_matrix.iloc[:, -2:])
plt.title("ECG5000_train last two signals")
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