import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train(root):
    path = os.path.join('..', root, 'train.csv')
    file = pd.read_csv(path)
    columns = file.columns
    lines = file.values
    train, val = train_test_split(lines, train_size=0.75, random_state=42)
    train = pd.DataFrame(train, columns=columns)
    train.to_csv('../train.csv', index=False)
    val = pd.DataFrame(val, columns=columns)
    val.to_csv('../val.csv', index=False)
    return True


if __name__ == '__main__':
    split_train('dataset')