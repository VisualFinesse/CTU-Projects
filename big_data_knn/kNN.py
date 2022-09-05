from sklearn import preprocessing
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# We're going to see if there's any correlation between an app rating and size to predict the cost of the application.

# Reading dataset from Kaggle https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps
df = pd.read_csv("google_playstore_data.csv")
# print(df)

# creating labelEncoder
le = preprocessing.LabelEncoder()

# First Feature
size = df["Size"]
# print(size)
size_encoding = le.fit_transform(size)
# print(size_encoding)

# Second Feature
rating = df["Rating"]
rating_encoding = le.fit_transform(rating)
# print(rating)

price = df["Price"]
# print(price.head(20))
price_encoding = le.fit_transform(price)
# print(price_encoding)

# Combine size into a single set of data using "zip" function - creating a single list of tuples
features = list(zip(size_encoding, rating_encoding))
# print(features.isna().sum())

# Split dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(
    features, price_encoding, test_size=0.3)  # 70% training and 30% test

# using sklearn neighbors KNeiborsClassifier we can start whipping our model into shape

model = KNeighborsClassifier(n_neighbors=3)
model.fit(features, price_encoding)

# Let's see what we did!
plt.title("ThisTitle")  # Create a title for our Graph

# converting types to be the same.
Y_train = list(Y_train)
Y_test = list(Y_test)

# print(type(X_train), type(y_train))


plt.title("Price and Features(Rating and Size)")  # Pl
plt.xlabel("Price in Encoded USD")
plt.ylabel("Rating and Size")

plt.plot(price[0:500], features[0:500])

plt.grid(True)  # Turns on the grid
plt.show()  # Shows the graph
