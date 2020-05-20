import cv2

image = cv2.imread("./br-coins/classification_dataset/all/5_1477145436.jpg")
# сначала сделаем фотку мажорной и переведем в чб
image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# потом надо разделить на черное и белое по порогу
# но никто не упоминал, что объект должен быть белым!!! это обязательно!
# из доки In OpenCV, finding contours is like finding white object from black background. So remember, object to be found should be white and background should be black.
# как известно - монетка темнее фона и потому надо из темного перевести в светлый
ret, thresh = cv2.threshold(image_grey,50,255,cv2.THRESH_BINARY_INV)

# используем финкцию
output = cv2.connectedComponentsWithStats(thresh)
# нам нужно только третье поле (всего их 4)
# там идет массив вида
# [
#  <координата x начала фрагмента>,
#  <координата y начала фрагмента>,
#  <ширина фрагмента>,
#  <высота фрагмента>,
#  <площадь фрагмента>
#  ]
stats = output[2]
for i, s in enumerate(stats):
    # теперь рассматриваем все зоны
    # сначала отсечем шумы - по площади
    if s[4] < 800:
        continue
    # ниже мы проверяем, чтобы стороны нашего фрагмента отличались не больше, чем на 15%
    height = s[3]
    width = s[2]
    if 0.85 < height/width < 1.15:
        # получили ч/б монетку вырезанную в квадрат
        new_image = image_grey[s[1]:s[1]+height, s[0]:s[0]+width]   #  img[y:y+h, x:x+w]
        # сделаем фон вокруг монетки черным
        for j in range(width):
            for k in range(height):
                if thresh[s[0] + j, s[1]+ k] == 0:
                    new_image[k, j] = 0 # если тут заменить на 255 - то будет фон белым и будет красивее

        cv2.imshow(f"frag{i}", new_image)
# фигня, которая д.б. в конце.
cv2.waitKey(0)
cv2.destroyAllWindows()
