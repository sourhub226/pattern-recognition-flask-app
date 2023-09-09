import pandas as pd
import numpy as np

# Load the CSV dataset into a pandas DataFrame
df = pd.read_csv("datasets/iris.csv")

# Assuming the first column contains class labels and the rest are features
class_column = df.columns[0]
feature_columns = df.columns[1:]

# Initialize an empty list to store the statistics data
stats_data = []

# Calculate statistics for each class and feature
for class_label in df[class_column].unique():
    class_data = df[df[class_column] == class_label]

    for feature in feature_columns:
        mean = class_data[feature].mean()
        std = class_data[feature].std()
        min_val = class_data[feature].min()
        max_val = class_data[feature].max()

        stats_data.append([class_label, feature, mean, std, min_val, max_val])

# Create the statistics DataFrame
columns = ["Class", "Feature", "Mean", "Std", "Min", "Max"]
stats_df = pd.DataFrame(stats_data, columns=columns)

# Pivot the DataFrame to have features as rows and statistics as columns
pivot_df = stats_df.pivot(index="Feature", columns="Class")

# Display the statistics table
print(pivot_df.T)
