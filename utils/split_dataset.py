import os
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train(root):
    path = os.path.join(root, 'train.csv')
    file = pd.read_csv(path)
    columns = list(file.columns.values)
    lines = file.values
    train, val = train_test_split(lines, train_size=0.75, random_state=42)
    # train = pd.DataFrame(train, columns=columns)
    train_with_parts = []
    for element in train:
        for i in range(1,3):
            tmp = copy.deepcopy(element)
            element = np.append(element, i)
            element[-1], element[-2] = element[-2], element[-1]
            train_with_parts.append(element)
            element = tmp

    val_with_parts = []
    for element in val:
        for i in range(1,3):
            tmp = copy.deepcopy(element)
            element = np.append(element, i)
            element[-1], element[-2] = element[-2], element[-1]
            val_with_parts.append(element)
            element = tmp

    columns.append('part')
    columns[-1], columns[-2] = columns[-2], columns[-1]

    train = pd.DataFrame(train_with_parts, columns=columns).sort_values(['id_code', 'part'])
    val = pd.DataFrame(val_with_parts, columns=columns).sort_values(['id_code', 'part'])

    train.to_csv('../train.csv', index=False, columns=columns)
    val.to_csv('../val.csv', index=False, columns=columns)
    return True


if __name__ == '__main__':
    split_train('/media/data/Projects/course_paper_2020/dataset')
