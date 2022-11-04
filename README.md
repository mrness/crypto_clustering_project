# Crypto Clustering 
import pandas as pd
import hvplot.pandas
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    Path("Resources/crypto_market_data.csv"),
    index_col="coin_id")

# Display sample data
df_market_data.head(2)
# Generate summary statistics
df_market_data.describe()
# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)
---
### Prepare the Data

This section prepares the data before running the K-Means algorithm. It follows these steps:

1. Use the `StandardScaler` module from scikit-learn to normalize the CSV file data. This will require you to utilize the `fit_transform` function.

2. Create a DataFrame that contains the scaled data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.

# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled_data = StandardScaler().fit_transform(df_market_data)
# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(
    scaled_data,
    columns=df_market_data.columns
)

# Copy the crypto names from the original data
df_market_data_scaled["coin_id"] = df_market_data.index

# Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coin_id")

# Display sample data
df_market_data_scaled.head()
---
### Find the Best Value for k Using the Original Data

In this section, you will use the elbow method to find the best value for `k`.

1. Code the elbow method algorithm to find the best value for `k`. Use a range from 1 to 11. 

2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.

3. Answer the following question: What is the best value for `k`?
# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))

# Create an empy list to store the inertia values
inert = []
# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
for i in k:
    km = KMeans(n_clusters=i, random_state=1)
    km.fit(df_market_data_scaled)
    inert.append(km.inertia_)
# Create a dictionary with the data to plot the Elbow curve
data = {"k":k, "inertia":inert}

# Create a DataFrame with the data to plot the Elbow curve
elbow_df = pd.DataFrame(data)
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
kmeans_elbow = elbow_df.hvplot.line(x = "k", y = "inertia", xlabel = "k", ylabel = "Inertia", title="KMeans Chart")
kmeans_elbow
#### Answer the following question: What is the best value for k?
**Question:** What is the best value for `k`?

**Answer:** k = 6 is the best value for k
---
### Cluster Cryptocurrencies with K-means Using the Original Data

In this section, you will use the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the price changes of cryptocurrencies provided.

1. Initialize the K-Means model with four clusters using the best value for `k`. 

2. Fit the K-Means model using the original data.

3. Predict the clusters to group the cryptocurrencies using the original data. View the resulting array of cluster values.

4. Create a copy of the original data and add a new column with the predicted clusters.

5. Create a scatter plot using hvPlot by setting `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.
# Initialize the K-Means model using the best value for k
kmeans = KMeans(n_clusters=6)
# Fit the K-Means model using the scaled data
kmeans.fit(df_market_data_scaled)
# Predict the clusters to group the cryptocurrencies using the scaled data
clusters = kmeans.predict(df_market_data_scaled)

# View the resulting array of cluster values.
clusters
# Create a copy of the DataFrame
df_market_data_scaled_clusters = df_market_data_scaled.copy()
# Add a new column to the DataFrame with the predicted clusters
df_market_data_scaled_clusters['cluster'] = clusters
df_market_data_scaled_clusters.head(5)

# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
kmeans_scatterplot = df_market_data_scaled_clusters.hvplot.scatter(x='price_change_percentage_24h', y='price_change_percentage_7d', by='cluster', hover_cols=['coin_id'], title='KMeans Cluster scatterplot')
kmeans_scatterplot
---
### Optimize Clusters with Principal Component Analysis

In this section, you will perform a principal component analysis (PCA) and reduce the features to three principal components.

1. Create a PCA model instance and set `n_components=3`.

2. Use the PCA model to reduce to three principal components. View the first five rows of the DataFrame. 

3. Retrieve the explained variance to determine how much information can be attributed to each principal component.

4. Answer the following question: What is the total explained variance of the three principal components?

5. Create a new DataFrame with the PCA data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.
# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)
# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
pca_components = pca.fit_transform(df_market_data_scaled)

# View the first five rows of the DataFrame. 
pd.DataFrame(pca_components).head(5)
# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
print("Total explained variance:", sum(pca.explained_variance_))
print("Total explained variance percentage:", sum(pca.explained_variance_ratio_))
#### Answer the following question: What is the total explained variance of the three principal components?

**Question:** What is the total explained variance of the three principal components?

**Answer:** # YOUR ANSWER HERE!
# Create a new DataFrame with the PCA data.
# Note: The code for this step is provided for you
pca_df = pd.DataFrame(pca_components, columns=['pca_0', 'pca_1', 'pca_2'])

# Copy the crypto names from the original data
pca_df['coin_id'] = df_market_data.index

# Set the coinid column as index
pca_df.set_index('coin_id', inplace=True)

# Display sample data
pca_df
---
### Find the Best Value for k Using the PCA Data

In this section, you will use the elbow method to find the best value for `k` using the PCA data.

1. Code the elbow method algorithm and use the PCA data to find the best value for `k`. Use a range from 1 to 11. 

2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.

3. Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?
# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))
# Create an empy list to store the inertia values
inert2 = []
# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    km = KMeans(n_clusters=i, random_state=1)
    km.fit(pca_df)
    inert2.append(km.inertia_)
# Create a dictionary with the data to plot the Elbow curve
data2 = {"k":k, "inertia":inert2}

# Create a DataFrame with the data to plot the Elbow curve
elbow_df = pd.DataFrame(data2)
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
kmeans_pca_elbow = elbow_df.hvplot.line(x = "k", y = "inertia", xlabel = "k", ylabel = "inertia", title="PCA KMeans Chart")
kmeans_pca_elbow
#### Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?
* **Question:** What is the best value for `k` when using the PCA data?

  * **Answer:** k = 6 is still the best value


* **Question:** Does it differ from the best k value found using the original data?

  * **Answer:** It does not differ.
---
### Cluster Cryptocurrencies with K-means Using the PCA Data

In this section, you will use the PCA data and the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the principal components.

1. Initialize the K-Means model with four clusters using the best value for `k`. 

2. Fit the K-Means model using the PCA data.

3. Predict the clusters to group the cryptocurrencies using the PCA data. View the resulting array of cluster values.

4. Add a new column to the DataFrame with the PCA data to store the predicted clusters.

5. Create a scatter plot using hvPlot by setting `x="PC1"` and `y="PC2"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.
# Initialize the K-Means model using the best value for k
kmeans = KMeans(n_clusters=6)
# Fit the K-Means model using the PCA data
kmeans.fit(pca_df)
# Predict the clusters to group the cryptocurrencies using the PCA data
kmeans_clusters = kmeans.predict(pca_df)

# View the resulting array of cluster values.
kmeans_clusters

# Create a copy of the DataFrame with the PCA data
pca_clusters_df = df_market_data_scaled.copy()

# Add a new column to the DataFrame with the predicted clusters
pca_clusters_df['cluster'] = kmeans_clusters

# Display sample data
pca_clusters_df.head(2)
# Create a scatter plot using hvPlot by setting 
# `x="PC1"` and `y="PC2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
kmeans_pca_scatterplot = pca_clusters_df.hvplot.scatter(x='price_change_percentage_24h', y='price_change_percentage_7d', by='cluster', hover_cols=['coin_id'], title='PCA KMeans Cluster Scatterplot')
kmeans_pca_scatterplot
---
### Visualize and Compare the Results

In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

1. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the Elbow Curve that you created to find the best value for `k` with the original and the PCA data.

2. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the cryptocurrencies clusters using the original and the PCA data.

3. Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

> **Rewind:** Back in Lesson 3 of Module 6, you learned how to create composite plots. You can look at that lesson to review how to make these plots; also, you can check [the hvPlot documentation](https://holoviz.org/tutorial/Composing_Plots.html).
# Composite plot to contrast the Elbow curves
kmeans_elbow + kmeans_pca_elbow
# Compoosite plot to contrast the clusters
kmeans_scatterplot + kmeans_pca_scatterplot
#### Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

  * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

  * **Answer:** # YOUR ANSWER HERE!
