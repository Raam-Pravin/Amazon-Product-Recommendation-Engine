# Amazon-Product-Recommendation-Engine
This repository contains an implementation of a product recommendation system using various collaborative filtering techniques. The primary focus is on building a system that can recommend products to users based on their preferences, using both item-based and model-based collaborative filtering approaches. The models have been fine-tuned using grid search cross-validation to optimize performance.

## Contents
- Amazon_Recommendation_Systems2.ipynb: Jupyter Notebook containing full code and output
- README.md: The document you are currently reading


## Overview
The project implements two main types of recommendation algorithms:

1. Item-Based Collaborative Filtering (KNN):
- Uses similarity between items to recommend similar products
- Hyperparameters like k (number of neighbors) and min_k (minimum number of neighbors) were optimized using grid search

2. Model-Based Collaborative Filtering (SVD):
- Uses matrix factorization techniques to recommend products based on latent factors extracted from the user-item interaction matrix
- Hyperparameters like n_epochs (number of iterations), lr_all (learning rate), and reg_all (regularization term) were fine-tuned to improve model accuracy


## Requirements:
To run this project, you must have the following libraries installed

- scikit-learn: Used for machine learning tasks, including the `mean_squared_error` function to evaluate model performance.
- Surprise: Library for building and evaluating collaborative filtering recommendation models.
- Pandas: Used for data manipulation and creating dataframes.
- NumPy: For numerical operations and handling arrays.
- Matplotlib/Seaborn: Used for visualizing model performance and metrics.
- collections.defaultdict: Simplifies handling missing keys in dictionaries during data processing and evaluation.


## Data
The dataset used in this project is a user-product interaction dataset, containing ratings for various products from different users. The dataset is preprocessed to create a user-item matrix, which serves as the input for the collaborative filtering algorithms. 

Link to dataset used can be found here: https://drive.google.com/file/d/1XahZcR287ke7j48I7-oj0KzmmwSSvA3Y/view?usp=drive_link 


## Results
- Item-Based Collaborative Filtering (KNN) achieved a F1-score of 0.816 after hyperparameter tuning, which shows that predicting ratings using items similar to the target item produces reasonable results
- SVD-based Collaborative Filtering performed the best with an F1-score of 0.828, suggesting that model-based approaches using matrix factorization provide better predictions


## Conclusion and Future Work
Both item-based and model-based collaborative filtering algorithms were implemented and evaluated.
The dataset's imbalance, with many ratings being 5, influenced the model's performance, often resulting in predicted ratings around 4.29, close to the mean rating.
Future work can focus on improving model performance through techniques like bootstrapping or generating synthetic datasets to better balance the rating distribution.
