import math
import random
import cv2
import pandas


def cut_coin(image, grad = 50):
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(image_grey,grad,255,cv2.THRESH_BINARY_INV)
    output = cv2.connectedComponentsWithStats(thresh)
    stats = output[2]
    image_S = image.shape[1] * image.shape[0]
    for i, s in enumerate(stats):
        if s[4] < 800:
            continue
        if s[4] > image_S//4:
            continue
        height = s[3]
        width = s[2]
        if 0.85 < height/width < 1.15:
            new_image = image[s[1]:s[1]+height, s[0]:s[0]+width].copy()
            n_thresh = thresh.copy()[s[1]:s[1]+height, s[0]:s[0]+width]
            for j in range(width):
                for k in range(height):
                    if n_thresh[k,j] == 0:
                        new_image[k, j] = (0,0,0)
            return new_image



def get_coin_params(image):
    # добавим вычисление площади, периметр
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_grey,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(image_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    cv2.drawContours(image, contours, -1, (255, 0, 0), 1, cv2.LINE_AA, hierarchy, 1)
    for i, cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        if area > 900:
            S = round(cv2.contourArea(cnt))
            perimeter = round(cv2.arcLength(cnt, True))
    # MDL
    max_x, max_y = len(image), len(image[0])
    pixels = image.tolist()
    blue_coordinates = []
    sum_x = 0
    sum_y = 0
    for x in range(max_x):
        for y in range(max_y):
            if pixels[x][y] == [255,0,0]:
                blue_coordinates.append([x,y])
                sum_x += x
                sum_y += y
    center_x = sum_x//len(blue_coordinates)
    center_y = sum_y//len(blue_coordinates)
    n = len(blue_coordinates)
    mdl = 0
    rmax = 0
    rmin = 100
    for i in range(n-1):
        r = d(blue_coordinates[i][0], blue_coordinates[i][1], center_x, center_y)
        if r > rmax:
            rmax = r
        if r < rmin:
            rmin = r
        for j in range(i+1, n):
            diam = d(blue_coordinates[i][0], blue_coordinates[i][1], blue_coordinates[j][0], blue_coordinates[j][1])
            if diam > mdl:
                mdl = diam
    #cv2.imshow(f'{random.randint(100, 10000)}.png', image)
    return S, perimeter, mdl, rmax, rmin



def d(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)




def create_matrix(image):
    max_x, max_y = len(image), len(image[0])
    # создадим матрицу яркостей для каждого пикселя. черный фон - пометим -1
    brightness = [[-1 for _ in range(max_y)] for i in range(max_y)]
    for i in range(max_x):
        for j in range(max_y):
            if image[i][j] == [0,0,0]:
                continue
            brightness[i][j] = sum(image[i][j]) // 3

    # создадим матрицу пространственной смежности
    matrix = [[0 for _ in range(256)] for _ in range(256)]
    steps = [-1, 0, 1]
    sum_ = 0
    for x in range(max_x):
        for y in range(max_y):
            for x_step in steps:
                current_x = x+x_step
                if current_x<0 or current_x==max_x:
                    continue
                for y_step in steps:
                    current_y = y + y_step
                    if current_y < 0 or current_y == max_y or current_x==x and current_y==y:
                        continue
                    if brightness[current_x][current_y] == -1:
                        continue
                    matrix[brightness[x][y]][brightness[current_x][current_y]] +=1
                    sum_ += 1
    for x in range(256):
        for y in range(256):
            matrix[x][y] /= sum_
    return matrix


def get_features(coin, S,perimeter,mdl, rmax, rmin) -> dict:
    features = dict()
    # Color
    start_pixels = coin.tolist()
    pixels = []
    for i in start_pixels:
        for j in i:
            if j != [0,0,0]:
                pixels.append(j)
    features["MIN1"],features["MIN2"],features["MIN3"] = min(pixels)
    features["MAX1"],features["MAX2"],features["MAX3"] = max(pixels)
    n = len(pixels)
    b, g, r = 0, 0, 0
    for i in pixels:
        b += i[0]
        g += i[1]
        r += i[2]
    MXb = b/n
    MXg = g/n
    MXr = r/n
    features["MXb"],features["MXg"],features["MXr"] = [MXb, MXg, MXr]
    sum = [0,0,0]
    for i in pixels:
        sum[0] += (i[0] - MXb)**2
        sum[1] += (i[1] - MXg) ** 2
        sum[2] += (i[2] - MXr) ** 2
    for j in range(3):
        sum[j] = math.sqrt(sum[j]/n)
    features["SD1"], features["SD2"] ,features["SD3"] = sum

    # тукстуры
    matrix = create_matrix(start_pixels)
    features["ENERGY"] = 0
    features["CON"] = 0
    features["MRP"] = 0
    features["LUN"] = 0
    features["ENT"] = 0
    features["TR"] = 0
    features["AV"] = 0
    for i in range(256):
        sum_i=0
        for j in range(256):
            gij = matrix[i][j]
            features["ENERGY"] += gij ** 2
            features["CON"] += gij*(i-j)**2
            if gij > features["MRP"]:
                features["MRP"] = gij
            features["LUN"] += gij/(1+(i-j)**2)
            features["ENT"] += gij*math.log(gij) if gij >0 else 0
            if i==j:
                features["TR"] +=gij
            sum_i +=gij
        features["AV"] += i*sum_i

    features["KF"] = perimeter**2/(4*math.pi)
    features['KV'] = mdl*math.pi/(4*S)
    features["KO"] = rmax/rmin
    return features

def image_features(image_path):
    #print(image_path, "!")
    image = cv2.imread(image_path)
    start_pixels = image.tolist()
    sum_ = 0
    n= 0
    for j in start_pixels:
        for i in j:
            if i == [0,0,0]:
                continue
            sum_ += sum(i)/3
            n+=1
    #print(sum_/n)

    new_image, s, p, mdl, rmax, rmin = cut_coin(image.copy(), round(sum_/n/2)+30)

    features = get_features(new_image, s, p, mdl, rmax, rmin)
    features.update({"path": image_path})
    return features

def image_features2(image_path, name = None, por = 30):
    #print(image_path, "!")
    image = cv2.imread(image_path)
    start_pixels = image.tolist()
    sum_ = 0
    n= 0
    for j in start_pixels:
        for i in j:
            if i == [0,0,0]:
                continue
            sum_ += sum(i)/3
            n+=1
    #print(sum_/n)

    new_image = cut_coin(image.copy(), round(sum_/n/2)+por)
    s, p, mdl, rmax, rmin = get_coin_params(new_image)

    features = get_features(new_image, s, p, mdl, rmax, rmin)
    result = {"path": name or image_path}
    result.update(features)
    return result



if __name__ == "__main__":
    images = [
        "./br-coins/classification_dataset/all/5_1477145436.jpg",
        "./br-coins/classification_dataset/all/10_1477288182.jpg",
        "./br-coins/classification_dataset/all/25_1477286388.jpg",
       "./br-coins/classification_dataset/all/50_1477145148.jpg",
        "./br-coins/classification_dataset/all/100_1477279626.jpg"
    ]
    d = [image_features2(image_path) for image_path in images]
    df = pandas.DataFrame(d)
    df.to_excel(r'result_dataframe.xlsx', index = False, header=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()