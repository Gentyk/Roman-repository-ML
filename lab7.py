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
    ''' строит график E(m)

    :param eigenvalues_: массив собственных значений матрицы
    '''
    eigenvalues = eigenvalues_.copy()
    eigenvalues.sort()
    x = [i for i in range(len(eigenvalues))]
    y = [(sum(eigenvalues[i:])/sum(eigenvalues)) for i in x]
    plt.plot(x, y)


def lab7(df_):
    # отрежем имена в первом столбце
    df = df_.copy()
    df = df.drop(['Y'], axis='columns')
    table = df.to_numpy()
    eigenvalues = get_eigenvalues(table)
    print(type(eigenvalues[0]))
    E(eigenvalues)
    plt.show()

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
        print(new_df.dtypes)

        new_df["Y"] = i
        dfs.append(new_df)
    df = pd.concat(dfs)
    lab7(df)



if __name__ == "__main__":
    main()
