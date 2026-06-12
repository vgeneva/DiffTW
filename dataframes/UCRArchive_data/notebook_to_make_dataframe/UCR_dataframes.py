import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


Dataset = 'ACSFone'  # "ECG5000" "ECG200"  "ECGFiveDays" "TwoLeadECG" "Beef" "GunPoint" "GunPointMaleVersusFemale"
save_path = f'/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_data/{Dataset}'


# Read the data from the CSV file
test = pd.read_csv(f'/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/{Dataset}/{Dataset}_TEST.tsv', sep='\t', header=None)
print(test.head())
print(test.tail())
print(f"test data size: {test.shape}")
print(test.info())
length = test.shape[1]
print(f"length: {length -1}")

# labels in the first column
# print test first 10 rows, 0 column
print(test.iloc[0:10, 0])

# plot the first row of the dataframe wihtout the first column
# use the first column to label the plot
plt.figure(figsize=(10, 5));
x = np.linspace(0, 1, length -1)
plt.plot(x, test.iloc[0, 1:])
plt.title(f"test label: {test.iloc[0, 0]}")
plt.show()


plt.figure(figsize=(10, 5));
x = np.linspace(0, 1, length -1)
plt.plot(x, test.iloc[4, 1:])
plt.title(f"test label: {test.iloc[4, 0]}")
plt.show()



# how many classes are there
print(test[0].unique())
# how many classes are there in total
print(test[0].value_counts())
list_class_0 = list(test[test[0] == 0].index)

for i in list_class_0:
    plt.figure(figsize=(10, 5));
    plt.plot(x, test.iloc[i, 1:])
    plt.title(f"test label: {test.iloc[i, 0]}")
    plt.show()


# what are the rows for class 1
print(test[test[0] == 1].index)

list_class_1 = list(test[test[0] == 1].index)
print(list_class_1)
list_class_1 = list_class_1[0:4]
print(list_class_1)

for i in list_class_1:
    plt.figure(figsize=(10, 5));
    plt.plot(x, test.iloc[i, 1:])
    plt.title(f"test label: {test.iloc[i, 0]}")
    plt.show()      


list_class_2 = list(test[test[0] == 2].index)
print(list_class_2)
list_class_2 = list_class_2[0:4]
print(list_class_2)



"""
list_class_3 = list(test[test[0] == 3].index)
print(list_class_3)
list_class_3 = list_class_3[0:4]
print(list_class_3)


list_class_4 = list(test[test[0] == 4].index)
print(list_class_4) 
list_class_4 = list_class_4[0:4]
print(list_class_4)
list_class_5 = list(test[test[0] == 5].index)
print(list_class_5)
list_class_5 = list_class_5[0:4]
print(list_class_5)
"""


# create plots, one for each label and plot
# the first 4 signals for each label
indices = [list_class_1, list_class_2]#, list_class_3]#, list_class_4, list_class_5]
print(indices)
print(len(indices))

# Create subplots (2 rows, 2 columns) 
fig, axes = plt.subplots(2, 1, figsize=(13, 8))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

# Flatten axes array
axes = axes.flatten()

# Loop through indices and plot data
for i, idx_list in enumerate(indices):
    for idx in idx_list:
        axes[i].plot(x,test.iloc[idx, 1:])

    # Set title
    axes[i].set_title(f"{Dataset} test labels: {', '.join(str(test.iloc[idx, 0]) for idx in idx_list)}")
    #axes[i].legend()

# remove the last figure as there are only a certain amount of figures
fig.delaxes(axes[-1])
plt.show();





# find index for each test[0].unique()
for label in test[0].unique():
    print(f" For class {label}: {test[test[0] == label].index}")


# find the count of each class in the 0th column, i.e. how many signals per class are there
for label in test[0].value_counts().index:
    print(f"{Dataset} Class {label} has {test[0].value_counts()[label]} test signals")



####### test csv matrix for each test
# ensure the save_path exists before saving files
os.makedirs(save_path, exist_ok=True)

# make csv's using labels and remove columne 0
for label in test[0].unique():
    df = test[test[0] == label]
    #df = df.drop(df.columns[0], axis = 1)
    df = df.transpose()
    # save in the specified path
    os.makedirs(save_path, exist_ok=True)

    file_path = os.path.join(save_path, f'{Dataset}{label}_test_matrix.csv')

    df.to_csv(file_path, index=False, header=False) # keeping the labels in test files

    print(f'Saved {file_path} with shape {df.shape}')



file_path_1 = os.path.join(save_path, f'{Dataset}1_test_matrix.csv')
print(f'Checking saved file at: {file_path_1}')
globals()[f"{Dataset}1_test_matrix"] = pd.read_csv(file_path_1)
print(globals()[f"{Dataset}1_test_matrix"].shape)
print(globals()[f"{Dataset}1_test_matrix"].head())


plt.figure(figsize=(10, 5));
#file_path_2 = os.path.join(save_path, f'{Dataset}2_test_matrix.csv')
#globals()[f"{Dataset}2_test_matrix"] = pd.read_csv(file_path_2)
#print(globals()[f"{Dataset}2_test_matrix"].shape)
#print(globals()[f"{Dataset}2_test_matrix"].head())

"""
file_path_3 = os.path.join(save_path, f'{Dataset}3_test_matrix.csv')
globals()[f"{Dataset}3_test_matrix"] = pd.read_csv(file_path_3)
print(globals()[f"{Dataset}3_test_matrix"].shape)
print(globals()[f"{Dataset}3_test_matrix"].head())


ECG50004_test_matrix = pd.read_csv('ECG50004_test_matrix.csv')
print(ECG50004_test_matrix.shape)
print(ECG50004_test_matrix.head())

ECG50005_test_matrix = pd.read_csv('ECG50005_test_matrix.csv')
print(ECG50005_test_matrix.shape)
print(ECG50005_test_matrix.head())
"""


######## train csv matrix   


train = pd.read_csv(f'/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/{Dataset}/{Dataset}_TRAIN.tsv', sep='\t', header=None)
print(train.head())
print(f"{Dataset}train data size: {train.shape}")

# find index for each Bee_train[0].unique()
for label in train[0].unique():
    print(f" For class {label}: {train[train[0] == label].index}")
list_class__train_1 = list(train[train[0] == 1].index)
print(list_class_1)
print(len(list_class_1))

list_class_train_2 = list(train[train[0] == 2].index)
print(list_class_2)
print(len(list_class_2))

list_class_train_0 = list(train[train[0] == 0].index)
 
for idx in list_class_train_0:
    plt.figure(figsize=(10, 5));
    plt.plot(x, train.iloc[idx, 1:])
    plt.title(f"train label: {train.iloc[idx, 0]}")
    plt.show()

"""
list_class_3 = list(train[train[0] == 3].index)
print(list_class_3)
print(len(list_class_3))


list_class_4 = list(ECG5000_train[ECG5000_train[0] == 4].index)
print(list_class_4)
print(len(list_class_4))

list_class_5 = list(ECG5000_train[ECG5000_train[0] == 5].index)
print(list_class_5)
print(len(list_class_5))
"""

# Plot all signals for class 1 on the same plot
plt.figure(figsize=(10, 5))
for idx in list_class_1:
    plt.plot(train.iloc[idx, 1:])
    plt.title(f"{Dataset}_train class 1 signals")
    plt.show()



plt.figure(figsize=(10, 5))
for idx in list_class_2:
    plt.plot(train.iloc[idx, 1:])
plt.title(f"{Dataset}_train class 2 signals")
plt.show()

"""
plt.figure(figsize=(10, 5))
for idx in list_class_3:
    plt.plot(train.iloc[idx, 1:])
plt.title(f"{Dataset}]_train class 3 signals")
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
"""

# find the count of each class in the 0th column, i.e. how many 16's are there
for label in train[0].value_counts().index:
    print(f"{Dataset} Class {label} has {train[0].value_counts()[label]} train signals")




# sort the dataframe by the first column
globals()[f"{Dataset}_train_sorted"] = train.sort_values(0)
print(globals()[f"{Dataset}_train_sorted"].head(10))    
plt.figure(figsize=(10, 5))
for j in range(5):
    plt.plot(globals()[f"{Dataset}_train_sorted"].iloc[j, 1:])
plt.title(f"{Dataset}_train class 1 signals")
plt.show()

#find 2 in first column
print(globals()[f"{Dataset}_train_sorted"][globals()[f"{Dataset}_train_sorted"][0] == 2].index)


#path = "/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_data/ECG5000"
# transpose train tsv file
# can't make labels into integers in first row.
# the columns keep the same dtype 
# will deal with only for accruacy results
globals()[f"{Dataset}_train_transpose"] = globals()[f"{Dataset}_train_sorted"].transpose() # this is the same as DATAset_train.T 
print(globals()[f"{Dataset}_train_transpose"]) 
print(globals()[f"{Dataset}_train_transpose"].shape)
print(globals()[f"{Dataset}_train_transpose"].iloc[1:, 0:5]) # first 5 signals, excluding the first row and last row
print(globals()[f"{Dataset}_train_transpose"].head(10))
print(globals()[f"{Dataset}_train_transpose"].iloc[:, -1]) 
# print the last two columns
print(globals()[f"{Dataset}_train_transpose"].iloc[:, -2:]) 


# plot the first signal in dataset_train_transpose
plt.figure(figsize=(10, 5))
plt.plot(globals()[f"{Dataset}_train_transpose"].iloc[1:, 0:5])
plt.title(f"{Dataset}_train first signals")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(globals()[f"{Dataset}_train_transpose"].iloc[1:, -2:])
plt.title(f"{Dataset}_train last signals")
plt.show()




# Save to CSV
globals()[f"{Dataset}_train_transpose"].to_csv(os.path.join(save_path, f'{Dataset}_train_matrix.csv'), index=False, header=False) # index for row, header for column index
print(globals()[f"{Dataset}_train_transpose"].shape)
print(globals()[f"{Dataset}_train_transpose"].head())

globals()[f"{Dataset}_train_matrix"] = pd.read_csv(os.path.join(save_path, f'{Dataset}_train_matrix.csv'))
print(globals()[f"{Dataset}_train_matrix"].shape)
print(globals()[f"{Dataset}_train_matrix"].head())
plt.figure(figsize=(10, 5))
plt.plot(globals()[f"{Dataset}_train_matrix"].iloc[:, -2:])
plt.title(f"{Dataset}_train last two signals")
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

#for label in test[0].value_counts().index:
    # sort label in order then print the count
    

#    print(f"{Dataset} Class {label} has {test[0].value_counts()[label]} test signals")

#for label in train[0].value_counts().index:
#    print(f"{Dataset} Class {label} has {train[0].value_counts()[label]} train signals")


# For test set
for label in sorted(test[0].value_counts().index):
    print(f"{Dataset} Class {label} has {test[0].value_counts()[label]} test signals")

# For train set
for label in sorted(train[0].value_counts().index):
    print(f"{Dataset} Class {label} has {train[0].value_counts()[label]} train signals")
