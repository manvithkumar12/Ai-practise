import pandas as pd
import numpy as np

import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "yasserh/titanic-dataset",
  path="Titanic-Dataset.csv"
)
cleaned_df = df

for i in df:
  if(df[i].dtype=="int"):
    print(i)

cleaned_df = pd.get_dummies(cleaned_df,columns=["Sex"])

cleaned_df["Embarked"] = cleaned_df["Embarked"].fillna(cleaned_df["Embarked"].mode()[0])
cleaned_df = pd.get_dummies(cleaned_df, columns=["Embarked"])

cleaned_df.isnull().sum()

cleaned_df["Age"] = cleaned_df["Age"].fillna(cleaned_df["Age"].median())

cleaned_df["Age"].isnull().sum()

import matplotlib.pyplot as plt

cleaned_df["Age"].plot.box()

plt.show()

cleaned_df["Pclass"].plot.box()

cleaned_df["SibSp"].plot.box()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numerical_cols = ["Age", "Fare", "SibSp", "Parch"]

cleaned_df[numerical_cols] = scaler.fit_transform(
    cleaned_df[numerical_cols]
)

cleaned_df

cleaned_df["Deck"] = cleaned_df["Cabin"].str[0]
cleaned_df["Deck"] = cleaned_df["Deck"].fillna(cleaned_df["Deck"].mode())
cleaned_df = pd.get_dummies(cleaned_df, columns=["Deck"])
cleaned_df = cleaned_df.drop(columns=["Cabin"])
