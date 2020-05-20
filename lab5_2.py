import math

import cv2
import pandas


def cut_coin(image, grad = 50):
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_grey,grad,255,cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    y, height, x,width = None, None, None, None
    my_contour = None
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        if area < 900:
            continue
        x, y, width, height = cv2.boundingRect(cnt)
        if 0.80 < height/width < 1.15:
            new_image = image[y:y+height, x:x+width].copy()
            for j in range(width):
                for k in range(height):
                    if thresh[y+k,x+j] == 0:
                        new_image[k, j] = (0,0,0)
            my_contour = cnt
            break

    # добавим вычисление площади, периметр
    cv2.drawContours(image, contours, -1, (255, 0, 0), 1, cv2.LINE_AA, hierarchy, 1)
    S = round(cv2.contourArea(my_contour))
    perimeter = round(cv2.arcLength(my_contour, True))

    # MDL
    new_image_with_border = image[y:y+height, x:x+width]
    pixels = new_image_with_border.tolist()
    blue_coordinates = []
    sum_x = 0
    sum_y = 0
    for x in range(height):
        for y in range(width):
            if pixels[x][y] == [255,0,0]:
                blue_coordinates.append([x,y])
                sum_x += x
                sum_y += y
    if not blue_coordinates:
        cv2.imshow('2_2.png', thresh)
        cv2.imshow(f'1.png', new_image)
        raise Exception("нет границы")
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
    # cv2.imshow(f'{s[1]}_2.png', thresh)
    # cv2.imshow(f'{s[1]}.png', new_image)
    return new_image, S, perimeter, mdl, rmax, rmin



def d(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)




def create_matrix(image):
    max_x, max_y = len(image), len(image[0])
    # создадим матрицу яркостей для каждого пикселя. черный фон - пометим -1
    brightness = [[-1 for _ in range(max_y)] for i in range(max_x)]
    for i in range(max_x):
        for j in range(max_y):
            if image[i][j] == [0,0,0]:
                continue
            r = sum(image[i][j])
            brightness[i][j] = r // 3

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

    b, g, r = 0, 0, 0
    for i in start_pixels:
        for j in i:
            if j != [0,0,0]:
                pixels.append(j)
                b += j[0]
                g += j[1]
                r += j[2]
    n = len(pixels)
    for i, bright in enumerate(min(pixels)):
        features[f"MIN{i}"] = bright
    for i, bright in enumerate(max(pixels)):
        features[f"MAX{i}"] = bright
    MXb = b/n
    MXg = g/n
    MXr = r/n
    features["MX"] = [MXb, MXg, MXr]
    sum = [0,0,0]
    for i in pixels:
        sum[0] += (i[0] - MXb)**2
        sum[1] += (i[1] - MXg) ** 2
        sum[2] += (i[2] - MXr) ** 2
    for j in range(3):
        sum[j] = math.sqrt(sum[j]/n)
    features["SD"] = sum

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

def image_features(image_path, name = None, por = 30):
    #print(image_path, "!")
    try:
        image = cv2.imread(image_path)
        start_pixels = image.tolist()
        sum_ = 0
        n= 0
        for j in start_pixels:
            for i in j:
                sum_ += sum(i)/3
                n+=1
        #print(sum_/n)

        new_image, s, p, mdl, rmax, rmin = cut_coin(image.copy(), round(sum_/n/2)+por)

        features = get_features(new_image, s, p, mdl, rmax, rmin)
        result = {"path": name or image_path}
        result.update(features)
        #cv2.imshow(f'{image_path.split(".")}.png', new_image)
    except:
        #cv2.imshow(f'{image_path.split(".")}.png', new_image)
        raise
    return result

if __name__ == "__main__":
    images = [
        # "./br-coins/classification_dataset/all/5_1477145436.jpg",
        # "./br-coins/classification_dataset/all/10_1477288182.jpg",
        # "./br-coins/classification_dataset/all/25_1477286388.jpg",
        # "./br-coins/classification_dataset/all/50_1477145148.jpg",
        "./br-coins/classification_dataset/all/100_1477279908.jpg"
    ]
    d = [image_features(image_path, por=60) for image_path in images]
    df = pandas.DataFrame(d)
    # df.to_excel(r'result_dataframe.xlsx', index = False, header=True)
    df.to_csv(r'ccc.csv', index=False,  header=False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()