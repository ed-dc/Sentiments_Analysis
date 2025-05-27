import pandas as pd
import matplotlib .pyplot as plt
import torch 


#Load the data and visualisation 

df = pd.read_csv("data/archive/Combined_Data.csv")
df["statement"] = df["statement"].str.lower()
df = df.drop(columns=["Unnamed: 0"], axis=1)

# print(df.head())
list_count = list(df["status"].value_counts())
plt.bar(df["status"].value_counts().index, list_count)
plt.xlabel("Different Pathology")
plt.ylabel("Number of samples")
plt.title('Mental Health Status Counts')
plt.show()
