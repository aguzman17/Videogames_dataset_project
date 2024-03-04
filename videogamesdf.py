import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "https://raw.githubusercontent.com/aguzman17/Videogames_dataset_project/main/vgsales.csv"
df = pd.read_csv(file_path)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)

# Fill missing values with mean for numeric columns
numeric_columns = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert Year column to integer
df['Year'] = df['Year'].astype(int)

print("\nCleaned Dataset:")
print(df.head())

columns = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales']

for column in columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

relationships = [('Year', 'Global_Sales'), ('EU_Sales', 'JP_Sales')]

for relationship in relationships:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=relationship[0], y=relationship[1], data=df)
    plt.title(f'Relationship between {relationship[0]} and {relationship[1]}')
    plt.xlabel(relationship[0])
    plt.ylabel(relationship[1])
    plt.show()
