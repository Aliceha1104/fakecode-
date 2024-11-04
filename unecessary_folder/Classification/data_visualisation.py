from random import random
from random import randint
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions
from statsmodels.graphics.gofplots import qqplot

def plot_data(data):
    features = ['PropType', 'Price', 'CBD Distance', 'Bedroom', 'Bathroom', 'Car-Garage', 'Landsize', 'Building Area', 'Property Age', 'Median Price', 'Median Rental', 'PropertyCount']
    fig, axs = plt.subplots(4, 3, figsize=(18, 16))  # Adjust the grid size based on the number of features

    for i, feature in enumerate(features):
        row = i // 3
        col = i % 3
        sns.kdeplot(data[feature], shade=True, ax=axs[row, col])
        sns.distplot(a=data[feature], hist=True, kde=True, rug=False, ax=axs[row, col])
        axs[row, col].set_title(feature)
        
    plt.tight_layout()
    plt.show()
    
def qq_plot_data(data):
    features = ['PropType', 'Price', 'CBD Distance', 'Bedroom', 'Bathroom', 'Car-Garage', 'Landsize', 'Building Area', 'Property Age', 'Median Price', 'Median Rental', 'PropertyCount']  # Add all the features you want to visualize
    num_features = len(features)
    fig, axs = plt.subplots(1, num_features, figsize=(5 * num_features, 5))

    for i, feature in enumerate(features):
        qqplot(data[feature], line='s', ax=axs[i])
        axs[i].set_title(feature)

    plt.tight_layout()
    plt.show()
    
def plot_heatmap(data):
    plt.figure(figsize=(10, 8))

    matrix = data.corr()
    title = "Feature Correlation Heatmap"
    
    sns.heatmap(matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title(title)
    plt.show()
    

    