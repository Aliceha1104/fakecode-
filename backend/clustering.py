
# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler,LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans


# In[2]:

file_path = '../dataset/Clean_dataset.csv'
df = pd.read_csv(file_path)


df = df.drop(['Address', 'Listing ID', 'PropType'], axis=1)
df.columns


df.describe()

suburb_encoder = LabelEncoder()
status_encoder = LabelEncoder()
re_agency_encoder = LabelEncoder()



df['Suburb'] = suburb_encoder.fit_transform(df['Suburb'])
df['Status'] = status_encoder.fit_transform(df['Status'])
df['RE Agency'] = re_agency_encoder.fit_transform(df['RE Agency'])

df['Price'] = np.log(df['Price Verify'])
df['Property Age'] = 2024 - df['Built Year Verify']


scaler = StandardScaler()

df[['CBD Distance', 'Bedroom', 'Bathroom', 'Car-Garage', 'Landsize', 'Building Area', 'Property Age', 'Suburb', 'RE Agency', 'Status']] = scaler.fit_transform(df[['CBD Distance', 'Bedroom', 'Bathroom', 'Car-Garage', 'Landsize', 'Building Area', 'Property Age', 'Suburb', 'RE Agency', 'Status']])

df

plt.scatter(x=df['Price'], y=df['CBD Distance'])
plt.show()


df = df.drop(['Bathroom', 'Bedroom', 'Car-Garage'], axis=1)

dist_df = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df[['Price', 'CBD Distance']])
    dist_df.append([k, kmeans.inertia_])
dist_df = pd.DataFrame(dist_df, columns=['Number of Cluster','Inertia'])

dist_df

def optimise_k_means(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)

    fig =plt.subplots(figsize=(10,5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Cluster')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

optimise_k_means(df[['Price', 'CBD Distance']], 10)

kmeans = KMeans(n_clusters=3)
kmeans.fit(df[['Price', 'CBD Distance']])
df['kmeans_3'] = kmeans.labels_
df


plt.figure(figsize=(6, 3))
plt.scatter(x=df['Price'], y=df['CBD Distance'], c=df['kmeans_3'], cmap='viridis')
plt.ylabel('Price')
plt.xlabel('CBD Distance')
plt.show()


df_cbd = df[['Price', 'CBD Distance', 'kmeans_3']]
df_landsize = df[['Price', 'Landsize', 'kmeans_3']]

# Take a random sample of 1000 rows
sampled_cbd = df_cbd.sample(n=1000, random_state=1)
sampled_landsize = df_landsize.sample(n=1000, random_state=1)

# Load the original clean dataset
clean_df = pd.read_csv(file_path)

# Compute the mean and standard deviation of the original columns
mean_price = clean_df['Price Verify'].mean()
std_price = clean_df['Price Verify'].std()
mean_cbd_distance = clean_df['CBD Distance'].mean()
std_cbd_distance = clean_df['CBD Distance'].std()
mean_landsize = clean_df['Landsize'].mean()
std_landsize = clean_df['Landsize'].std()

# Denormalize the sampled data
sampled_cbd['Price'] = sampled_cbd['Price'] * std_price + mean_price
sampled_cbd['CBD Distance'] = sampled_cbd['CBD Distance'] * std_cbd_distance + mean_cbd_distance
sampled_landsize['Price'] = sampled_landsize['Price'] * std_price + mean_price
sampled_landsize['Landsize'] = sampled_landsize['Landsize'] * std_landsize + mean_landsize

# Save the denormalized data to final JSON files
sampled_cbd.to_json('denormalized_price_cbd_sample.json', orient='records')
sampled_landsize.to_json('denormalized_price_landsize_sample.json', orient='records')







