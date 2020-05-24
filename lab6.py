import csv
import itertools

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def teach(indexes):
    X = df.values[:, indexes]
    X_train, test_X, y_train, test_Y = train_test_split(X, Y, test_size=0.2)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    result = knn.predict(test_X)

    # проверка результатов
    n = len(test_Y)
    good = 0

    for i in range(n):
        if result[i] == test_Y[i]:
            good += 1
    print(good / n)
    test_result = good / n

    result = knn.predict(X_train)
    good = 0
    n = len(y_train)
    for i in range(n):
        if result[i] == y_train[i]:
            good += 1
    print(good / n)
    return test_result

file_name = "features_train.csv"
with open(file_name,'r') as f:
    reader=csv.reader(f,delimiter=',')
    n=len(next(reader))

# обучение
df1 = pd.read_csv(file_name)

df1.columns = ['Y'] + ['X'+str(i) for i in range(n-1)]
df1.head()
df = df1.copy()
for numerical_columns in ['X'+str(i) for i in range(n-1)]:
    data_numerical = df[numerical_columns]
    data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()
    data_numerical.describe()


Y = df.values[:, 0]
indexes = itertools.combinations(range(1,n-1),8)
max = 0
for i in indexes:
    toch = teach(i)
    if toch > max:
        print(toch, i)
        max = toch

# the best features (1, 3, 7, 8, 9, 10, 11, 19)

