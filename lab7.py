import pandas as pd
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt


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
    eigenvalues = eigenvalues_.copy()
    eigenvalues.sort()
    x = [i for i in range(len(eigenvalues))]
    y = [(sum(eigenvalues[i:])/sum(eigenvalues)) for i in x]
    plt.plot(x, y)


def lab7(df):
    # отрежем имена в первом столбце
    table = df.to_numpy()[:, 1:]
    eigenvalues = get_eigenvalues(table)
    print(type(eigenvalues[0]))
    E(eigenvalues)
    plt.show()
    # print(G)
    # a = 20
    # print(a)

def main():
    names = ["B", "Limfoma", "Norma", "T"]
    df = None
    for i in names:
        new_df = pd.read_csv(f"./blasts/{i}.csv", sep=";")
        new_df = new_df.drop(new_df.columns[[0]], axis='columns')
        new_df["Y"] = i
        if df is None:
            df = new_df.copy()
        else:
            df.merge(new_df.copy())


if __name__ == "__main__":
    main()
