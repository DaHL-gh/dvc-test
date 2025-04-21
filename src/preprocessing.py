import pandas as pd
import numpy as np


def preprocess_dataset():
    df = pd.read_csv("./data/housing.csv")

    # one extra engeneered feature
    df["rooms_per_household"] = df["total_rooms"] / df["households"]

    # ohe categorical column
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

    # drop highly correlated columns
    df = drop_highly_correlated(df)

    df.to_csv("./data/housing_processed.csv")


def drop_highly_correlated(df: pd.DataFrame, threshold=0.90) -> pd.DataFrame: 
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    return df.drop(columns=to_drop)

if __name__ == "__main__":
    preprocess_dataset()
