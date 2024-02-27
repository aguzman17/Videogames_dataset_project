import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the local file
file_path = "vgsales.csv"
df = pd.read_csv(file_path)

# Step 5: Cleaning Your Data
# Add your data cleaning steps here

# Step 6: Single Variable Distribution Plots
# Choose four columns
columns = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales']

for column in columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Step 7: Multiple Variable Plots
# Choose two relationships
relationships = [('Year', 'Global_Sales'), ('EU_Sales', 'JP_Sales')]

for relationship in relationships:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=relationship[0], y=relationship[1], data=df)
    plt.title(f'Relationship between {relationship[0]} and {relationship[1]}')
    plt.xlabel(relationship[0])
    plt.ylabel(relationship[1])
    plt.show()

