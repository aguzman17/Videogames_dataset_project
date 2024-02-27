import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the local file
file_path = "vgsales.csv"
df = pd.read_csv(file_path)

#Step 5
# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)

# Fill missing values with mean
df.fillna(df.mean(), inplace=True)

# Rremove duplicates
df.drop_duplicates(inplace=True)

# Convert 'Year' column to integer
df['Year'] = df['Year'].astype(int)

# Display the cleaned dataset
print("\nCleaned Dataset:")
print(df.head())

# Step 6
# Choose four columns
columns = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales']

for column in columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Step
# Choose two relationships
relationships = [('Year', 'Global_Sales'), ('EU_Sales', 'JP_Sales')]

for relationship in relationships:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=relationship[0], y=relationship[1], data=df)
    plt.title(f'Relationship between {relationship[0]} and {relationship[1]}')
    plt.xlabel(relationship[0])
    plt.ylabel(relationship[1])
    plt.show()

