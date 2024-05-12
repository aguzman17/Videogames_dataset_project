### Milestone 1 ###

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

### Milestone 2 ###

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Define features (X) and target variable (y)
X = df.drop(['Global_Sales', 'Name', 'Platform', 'Genre', 'Publisher'], axis=1)  # Excluding non-numeric columns and target variable
y = df['Global_Sales']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#K-Nearest Neighbors
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Predictions on training set & testing set
y_train_pred = knn_model.predict(X_train_scaled)
y_test_pred = knn_model.predict(X_test_scaled)

# Evaluate the model on training data & testing data
train_mse = mean_squared_error(y_train, y_train_pred)
print("Training Mean Squared Error:", train_mse)

test_mse = mean_squared_error(y_test, y_test_pred)
print("Testing Mean Squared Error:", test_mse)

k_values = [3, 7, 10]

for k in k_values:
    # Train the K-Nearest Neighbors model with different k values
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train_scaled, y_train)

    # Predictions on training set & testing set
    y_train_pred = knn_model.predict(X_train_scaled)
    y_test_pred = knn_model.predict(X_test_scaled)

    # Evaluate the model on training data & testing data
    train_mse = mean_squared_error(y_train, y_train_pred)
    print(f"Training Mean Squared Error (k={k}):", train_mse)

    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"Testing Mean Squared Error (k={k}):", test_mse)
    print()

# Define the Decision Tree model
decision_tree_model = DecisionTreeRegressor(random_state=42)

# Fit the model to the training data
decision_tree_model.fit(X_train_scaled, y_train)

# Predictions on training set  & testing set
y_train_pred_dt = decision_tree_model.predict(X_train_scaled)
y_test_pred_dt = decision_tree_model.predict(X_test_scaled)

# Evaluate the model on training data & testing data
train_mse_dt = mean_squared_error(y_train, y_train_pred_dt)
print("Decision Tree - Training Mean Squared Error:", train_mse_dt)

test_mse_dt = mean_squared_error(y_test, y_test_pred_dt)
print("Decision Tree - Testing Mean Squared Error:", test_mse_dt)

# Define different hyperparameters
max_depth_values = [None, 5, 10]
min_samples_split_values = [2, 5, 10]

for max_depth in max_depth_values:
    for min_samples_split in min_samples_split_values:
        # Define the Decision Tree model with different hyperparameters
        decision_tree_model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)

        # Fit the model to the training data
        decision_tree_model.fit(X_train_scaled, y_train)

        # Predictions on training set & testing set
        y_train_pred_dt = decision_tree_model.predict(X_train_scaled)
        y_test_pred_dt = decision_tree_model.predict(X_test_scaled)

        # Evaluate the model on training data & testing data
        train_mse_dt = mean_squared_error(y_train, y_train_pred_dt)
        test_mse_dt = mean_squared_error(y_test, y_test_pred_dt)

        print(f"Decision Tree (max_depth={max_depth}, min_samples_split={min_samples_split}) - Training MSE: {train_mse_dt}, Testing MSE: {test_mse_dt}")
        
### Milestone 3 ###

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#Step 10
# Group the dataset by release year and calculate the total global sales for each year
yearly_sales = df.groupby('Year')['Global_Sales'].sum().reset_index()

# Drop rows with missing values in the 'Year' column
df_cleaned = df.dropna(subset=['Year'])

# Select numeric columns for PCA
numeric_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
df_numeric = df_cleaned[numeric_columns]

# Standardize the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Perform PCA
pca = PCA()
pca.fit(df_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='-')
plt.title('Explained Variance Ratio by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.grid(True)
plt.show()

# Cumulative explained variance ratio
cumulative_explained_variance_ratio = explained_variance_ratio.cumsum()

# Plot cumulative explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), cumulative_explained_variance_ratio, marker='o', linestyle='-')
plt.title('Cumulative Explained Variance Ratio by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.xticks(range(1, len(cumulative_explained_variance_ratio) + 1))
plt.grid(True)
plt.show()

#Step 11
# Select numeric columns for clustering analysis
sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
X_sales = df[sales_columns]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sales)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Based on the Elbow Method the optimal number of clusters
optimal_k = 4

# Perform KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)

# Assign cluster labels to the dataset
df['Cluster'] = kmeans.labels_

# Visualize the clusters in 2D space
plt.figure(figsize=(10, 6))
sns.scatterplot(x='NA_Sales', y='EU_Sales', hue='Cluster', data=df, palette='viridis', legend='full')
plt.title('Clustering of Video Games Based on Sales in NA and EU')
plt.xlabel('NA Sales')
plt.ylabel('EU Sales')
plt.grid(True)
plt.show()
