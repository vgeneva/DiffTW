import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

 
# Read the data from the CSV file
ECGThorax1_test = pd.read_csv('/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/NonInvasiveFetalECGThorax1/NonInvasiveFetalECGThorax1_TEST.tsv', sep='\t', header=None)
print(ECGThorax1_test.head())
print(ECGThorax1_test.tail())
print(f"ECGThorax1 test data size: {ECGThorax1_test.shape}")
print(ECGThorax1_test.info())
length = ECGThorax1_test.shape[1]
print(f"length: {length -1}")

ECGThorax1_train = pd.read_csv('/Users/vickyhaney/Documents/GAship/DrBruno/EKG/UCRArchive_2018/NonInvasiveFetalECGThorax1/NonInvasiveFetalECGThorax1_TRAIN.tsv', sep='\t', header=None)
print(ECGThorax1_train.head())
print(f"ECGThorax1_train data size: {ECGThorax1_train.shape}")

# find index for each Bee_train[0].unique()
for label in ECGThorax1_train[0].unique():
    print(f" For class {label}: {len(ECGThorax1_train[ECGThorax1_train[0] == label].index)}")
list_class_1 = list(ECGThorax1_train[ECGThorax1_train[0] == 1].index)
print(list_class_1)
print(len(list_class_1))