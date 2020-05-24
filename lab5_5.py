from cv2 import cv2
import numpy as np
import pandas as pd
import collections
import math
#функция для поиска монет
def getCircle(src):
    circles = cv2.HoughCircles(
        src, cv2.HOUGH_GRADIENT, 2, 60, param1=300, param2=30, minRadius=30, maxRadius=50)
    x = []
    y = []
    max_value = []
#если нет монет, вернуть None
    if circles is None:
        return None, None, None
    else:
#если монет на изображении несколько, пройтись по всем
        for i in range(len(circles)):
            for circle in circles[i]:
                x_center = int(circle[0])
                y_center = int(circle[1])
                x.append(x_center)
                y.append(y_center)
                max_value.append(circle[2])
        return x,y, max_value
#функция для поиска цветовых характеристик
def color_characteristics(resized):
# компонента R
    r = np.array([resized[i][j][0] for i in range(len(resized)) for j in range(len(resized[i]))
                  if not (resized[i][j][0] == 255 and resized[i][j][1] == 255 and resized[i][j][2] == 255)])
# компонента G
    g = np.array([resized[i][j][1] for i in range(len(resized)) for j in range(len(resized[i]))
                  if not (resized[i][j][0] == 255 and resized[i][j][1] == 255 and resized[i][j][2] == 255)])
# компонента B
    b = np.array([resized[i][j][2] for i in range(len(resized)) for j in range(len(resized[i]))
                  if not (resized[i][j][0] == 255 and resized[i][j][1] == 255 and resized[i][j][2] == 255)])
    return np.mean(r), np.mean(g), np.mean(b), np.max(r), np.max(g), np.max(b), np.min(r), np.min(g), np.min(b), r, g, b
#Функция для поиска среднеквадратичного отклонения
def getCD(r, g, b, MX_R, MX_G, MX_B):
    CD_R = 0
    CD_G = 0
    CD_B = 0
    for i in range(len(r)):
            CD_R = CD_R + math.pow(r[i] - MX_R, 2)
            CD_G = CD_G + math.pow(g[i] - MX_G, 2)
            CD_B = CD_B + math.pow(b[i] - MX_B, 2)
    CD_R = math.sqrt(CD_R / len(r))
    CD_G = math.sqrt(CD_G / len(r))
    CD_B = math.sqrt(CD_B / len(r))
    return CD_R, CD_G, CD_B
#Функция для поиска текстурных признаков
def textural_features(mass):
    matrix = np.zeros((255,255), dtype=int) #нулевая матрица 255 на 255
    for i in range(len(mass) -1):
        matrix[mass[i] - 1][mass[i+1] - 1] = matrix[mass[i] - 1][mass[i+1] - 1] + 1
    return np.sum(matrix), get_textural_features(matrix), matrix

def get_textural_features(mass):
    CON = 0
    LUN = 0
    ENT = 0
    TR = 0
    AV = 0
    for i in range(len(mass)):
        TR = TR + mass[i][i]
        AV = AV + i
        for j in range(len(mass[i])):
            AV = AV + mass[i][j]
            ENT = ENT + mass[i][j] * math.log1p(math.log1p(mass[i][j]))
            LUN = LUN + mass[i][j] / (1 + math.pow(i - j, 2))
            CON = CON + math.pow(i-j, 2) * mass[i][j]
    return CON, LUN, ENT, TR, AV
#Функция для поиска морфологических признаков
def morphological_feature(radius):
    return math.pow(2*math.pi*radius, 2)/ 4* math.pi
#Наложение маски на изображение и соединение с белым фоном
def mask(src1, x, y, max_value):
    ###Черно-белая маска
    mask = np.full((src1.shape[0], src1.shape[1]), 0, dtype=np.uint8)
    cv2.circle(mask, (x, y), max_value, (255, 255, 255), -2)

    fg = cv2.bitwise_or(src1, src1, mask=mask)

    mask = cv2.bitwise_not(mask)
    background = np.full(src1.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=mask)

    return cv2.bitwise_or(fg, bk)
#Функция для того, чтобы изображение стало ярче
def make_img_brighter(src1):
    ###делаем ярче
    brightness = 40
    contrast = 20
    src1 = np.int16(src1)
    src1 = src1 * (contrast / 127 + 1) - contrast + brightness
    src1 = np.clip(src1, 0, 255)
    src1 = np.uint8(src1)
    return src1

def get_char(path):
    src =cv2.imread(path)
    src1 = src.copy()
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    x, y, max_value = getCircle(src)

    if (x == None and y == None and max_value == None):
        print('Error')
    else:
        for i in range(len(x)):
            src1 = make_img_brighter(src1)
            final = mask(src1, x[i], y[i], max_value[i])
            y_center = int(y[i] - max_value[i])
            x_center = int(x[i] - max_value[i])
            radius = int(max_value[i])
            if x_center < 0:
                x_center = 0
            if y_center < 0:
                y_center = 0
            resized = cv2.resize(np.array(final[y_center: y_center + 2 * radius, x_center: x_center + 2 * radius]),
                                 (150, 150),
                                 interpolation=cv2.INTER_CUBIC)

            MX_R, MX_G, MX_B, MAX_R, MAX_G, MAX_B, MIN_R, MIN_G, MIN_B, r, g, b = color_characteristics(resized)

            CD_R, CD_G, CD_B = getCD(r, g, b, MX_R, MX_G, MX_B)
            Energy_R, feature_R, matrix_R = textural_features(r)
            Energy_G, feature_G, matrix_G = textural_features(g)
            Energy_B, feature_B, matrix_B = textural_features(b)
            print("Цветовые признаки:")
            print("MAX по компоненте R: {}, MAX по компоненте G: {}, MAX по компоненте B: {}".format(MAX_R, MAX_G, MAX_B))
            print("MIN по компоненте R: {}, MIN по компоненте G: {}, MIN по компоненте B: {}".format(MIN_R, MIN_G, MIN_B))
            print("Cреднее по компоненте R: {}, Среднее по компоненте G: {}, Среднее по компоненте B: {}".format(MX_R, MX_G, MX_B))
            print(
                "Cреднекваратичное отклонение по компоненте R: {}, Cреднекваратичное по компоненте G: {}, Cреднекваратичное по компоненте B: {}".format(
                    CD_R, CD_G, CD_B))
            print("Текстурные признаки:")
            print("Энергия по компоненте R: {}, Энергия по компоненте G: {}, Энергия по компоненте B: {}".format(Energy_R, Energy_G,
                                                                                                                 Energy_B))
            print("Контраст по компоненте R: {}, Контраст по компоненте G: {}, Контраст по компоненте B: {}".format(feature_R[0],
                                                                                                                    feature_G[0],
                                                                                                                    feature_B[0]))
            print("LUN по компоненте R: {}, LUN по компоненте G: {}, LUN по компоненте B: {}".format(feature_R[1], feature_G[1],
                                                                                                     feature_B[1]))
            print("Энропия по компоненте R: {}, Энтропия по компоненте G: {}, Энтропия по компоненте B: {}".format(feature_R[2],
                                                                                                                   feature_G[2],
                                                                                                                   feature_B[2]))
            print("TR по компоненте R: {}, TR по компоненте G: {}, TR по компоненте B: {}".format(feature_R[3], feature_G[3],
                                                                                                  feature_B[3]))
            print("AV по компоненте R: {}, AV по компоненте G: {}, AV по компоненте B: {}".format(feature_R[4], feature_G[4],
                                                                                                  feature_B[4]))
            print("Морфологические признаки:")
            print("Коэффициент формы: {}".format(morphological_feature(max_value[i])))

if __name__ == "__main__":
    src =cv2.imread('./br-coins/classification_dataset/all/100_1477279626.jpg')
    src1 = src.copy()
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    x, y, max_value = getCircle(src)

    if (x == None and y == None and max_value == None):
        print('Error')
    else:
        for i in range(len(x)):
            src1 = make_img_brighter(src1)
            final = mask(src1, x[i], y[i], max_value[i])
            y_center = int(y[i] - max_value[i])
            x_center = int(x[i] - max_value[i])
            radius = int(max_value[i])
            if x_center < 0:
                x_center = 0
            if y_center < 0:
                y_center = 0
            resized = cv2.resize(np.array(final[y_center: y_center + 2 * radius, x_center: x_center + 2 * radius]),
                                 (150, 150),
                                 interpolation=cv2.INTER_CUBIC)

            MX_R, MX_G, MX_B, MAX_R, MAX_G, MAX_B, MIN_R, MIN_G, MIN_B, r, g, b = color_characteristics(resized)

            CD_R, CD_G, CD_B = getCD(r, g, b, MX_R, MX_G, MX_B)
            Energy_R, feature_R, matrix_R = textural_features(r)
            Energy_G, feature_G, matrix_G = textural_features(g)
            Energy_B, feature_B, matrix_B = textural_features(b)
            print("Цветовые признаки:")
            print("MAX по компоненте R: {}, MAX по компоненте G: {}, MAX по компоненте B: {}".format(MAX_R, MAX_G, MAX_B))
            print("MIN по компоненте R: {}, MIN по компоненте G: {}, MIN по компоненте B: {}".format(MIN_R, MIN_G, MIN_B))
            print("Cреднее по компоненте R: {}, Среднее по компоненте G: {}, Среднее по компоненте B: {}".format(MX_R, MX_G, MX_B))
            print(
                "Cреднекваратичное отклонение по компоненте R: {}, Cреднекваратичное по компоненте G: {}, Cреднекваратичное по компоненте B: {}".format(
                    CD_R, CD_G, CD_B))
            print("Текстурные признаки:")
            print("Энергия по компоненте R: {}, Энергия по компоненте G: {}, Энергия по компоненте B: {}".format(Energy_R, Energy_G,
                                                                                                                 Energy_B))
            print("Контраст по компоненте R: {}, Контраст по компоненте G: {}, Контраст по компоненте B: {}".format(feature_R[0],
                                                                                                                    feature_G[0],
                                                                                                                    feature_B[0]))
            print("LUN по компоненте R: {}, LUN по компоненте G: {}, LUN по компоненте B: {}".format(feature_R[1], feature_G[1],
                                                                                                     feature_B[1]))
            print("Энропия по компоненте R: {}, Энтропия по компоненте G: {}, Энтропия по компоненте B: {}".format(feature_R[2],
                                                                                                                   feature_G[2],
                                                                                                                   feature_B[2]))
            print("TR по компоненте R: {}, TR по компоненте G: {}, TR по компоненте B: {}".format(feature_R[3], feature_G[3],
                                                                                                  feature_B[3]))
            print("AV по компоненте R: {}, AV по компоненте G: {}, AV по компоненте B: {}".format(feature_R[4], feature_G[4],
                                                                                                  feature_B[4]))
            print("Морфологические признаки:")
            print("Коэффициент формы: {}".format(morphological_feature(max_value[i])))
