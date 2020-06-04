import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def get_eigenvalues(table):
    ''' Получение собственных значений numpy матрицы '''
    # в таблице есть строковые записи и числа типа "0,32", т.е. через запятую - они не чикаются
    a = table.tolist()
    z = []
    for array in a:
        z.append([i if isinstance(i, float) or isinstance(i, int) else float(i.replace(',', '.')) for i in array])
    table = np.array(z)

    transp_table = table.transpose()
    G = np.dot(transp_table, table)
    eigenvalues, _ = linalg.eig(G)
    print(len(eigenvalues))
    print([z for z in eigenvalues if z.imag != 0])
    eigenvalues = eigenvalues.real
    return eigenvalues


def E(eigenvalues_):
    ''' строит график E(m)

    :param eigenvalues_: массив собственных значений матрицы
    '''
    eigenvalues = eigenvalues_.copy()
    eigenvalues.sort()
    x = [i for i in range(len(eigenvalues))]
    y = [(sum(eigenvalues[i:])/sum(eigenvalues)) for i in x]
    plt.plot(x, y)
    # примечание  - график показывает, что все признаки являются эффективными
    # см http://www.machinelearning.ru/wiki/images/archive/a/a2/20150509140209%21Voron-ML-regression-slides.pdf
    # слайд 21 из 23

def ML_preprocessing(df_):
    '''Возвращает список интересующих столбцов. Фактически пытается сделать feature selection, хотя не думаю, что вэтом есть смысл'''
    # принцип выделения наиболе информативных признаков выделент тут
    # https://towardsdatascience.com/feature-selection-and-dimensionality-reduction-f488d1a035de
    df = df_.copy()
    df = df.drop(['Y'], axis='columns')

    # Удалить сильно коррелированные столбцы(более чем 0.6)
    columns = df.columns
    exclude_columns = []
    interest_columns = []
    corr_matrix = df.corr().abs()
    for column in columns:
        if column in exclude_columns:
            continue
        interest_columns.append(column)
        series = corr_matrix[column]
        names = [name for name in series.index if name not in exclude_columns and name not in interest_columns and name != column]
        for name in names:
            if series[name] > 0.5:
                exclude_columns.append(name)
    print(len(interest_columns))
    return list(set(interest_columns)), list(set(exclude_columns))


def runSVC(X_, Y_):
    # получение точности при обучении методом опорных векторов
    X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.3)
    knn = SVC()
    knn.fit(X_train, y_train)
    result = knn.predict(X_test)

    # проверка результатов
    n = len(y_test)
    good = 0
    for i in range(n):
        if result[i] == y_test[i]:
            good += 1
    print("Точность:", good / n)


def runML(df_, columns: list):
    '''Все что связанно с машинным обучением'''

    # сначала отработаем на всех столбцах
    df = df_.copy()
    Y = df['Y']
    X_ = df.drop(['Y'], axis='columns')
    X = X_.values
    Y = Y.values
    # добавили нормальзацию
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    runSVC(X,Y)

    # потом возьмем ограниченное количество столбов, выделенное с помощью feature selection
    new_x = X_[columns]
    new_x = new_x.values
    new_x = scaler.fit_transform(new_x)
    runSVC(new_x,Y)

def lab7(df_):
    # отрежем имена в первом столбце
    df = df_.copy()
    df = df.drop(['Y'], axis='columns')
    table = df.to_numpy()
    eigenvalues = get_eigenvalues(table)
    print(type(eigenvalues[0]))
    E(eigenvalues)

def main():
    names = ["B", "Limfoma", "Norma", "T"]
    dfs = []
    for i in names:
        print(i)
        new_df = pd.read_csv(f"./blasts/{i}.csv", sep=";")

        # для последующего мержа датафреймов - отредактируем столбики
        new_df = new_df.drop(new_df.columns[[0]], axis='columns')
        new_names = {name: name.strip() for name in new_df.columns}
        new_df.rename(columns=new_names, inplace=True)

        # приведем все столбики к типу float
        new_df = new_df.replace(to_replace =',', value = '.', regex = True)
        for col_name in new_df.columns:
            new_df[col_name] = new_df[col_name].astype(float)
        #print(new_df.dtypes)

        new_df["Y"] = i
        dfs.append(new_df)
    df = pd.concat(dfs)
    lab7(df)
    columns, _ = ML_preprocessing(df)
    runML(df, columns)
    plt.show()


if __name__ == "__main__":
    main()
