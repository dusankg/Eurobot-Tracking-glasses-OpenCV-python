#import potrebnih biblioteka
import cv2
import numpy as np
import matplotlib.pylab as plt


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs): # samo primenljuje trashold na prosledjenu sliku i INVERTUJE je
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 3)
    image_bin = 255 - image_bin
    return image_bin

#Funkcionalnost implementirana u OCR basic
def resize_region(region): # resajzuje izdvojeni region na 28x28
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized
def scale_to_range(image):
    return image / 255



# rotiranje slike, proveriti da li treba, u zavisnosti gde bude poz kamera
def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

'''
nalazi dominantnu boju slike
prvi plan mi je bio da pronalazim samo oblike, pa da onda trazim boju tog oblika, odn case
'''
'''
def getDominantColor(image):
    image = cv2.resize(image, (50, 250), interpolation=cv2.INTER_NEAREST)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=4)
    labels = clt.fit_predict(image)
    label_counts = collections.Counter(labels)
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    return list(dominant_color)
'''
def extract_glasses_location(image_path):

    # funkcija vraca broj 1, 2 ili 3 u zavisnosti od prepoznaog layouta

    imgOrg = cv2.imread(image_path)
    imgOrg2 = imgOrg.copy()
    imgOrg2 = cv2.cvtColor(imgOrg2, cv2.COLOR_BGR2RGB)
    #imgOrg = imgOrg[1100:1400, 250:600]
    #imgOrg = imgOrg[1000:1450, 150:700]
    imgOrg = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2HSV)

    '''
    CRVENA
    '''

    # da izdvojim samo crvenu boju pravljenjem maske
    # za rgb
    #lower = [70, 0, 0]
    #upper = [255, 70, 70]
    # za hsv
    lower = [150, 170, 190]
    upper = [255, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(imgOrg, lower, upper)
    output = cv2.bitwise_and(imgOrg, imgOrg, mask=mask)

    imgGray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    imgGray = 255 - imgGray
    image_bin = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 501, 1)
    kernel = np.ones((4, 4))
    image_bin = cv2.dilate(image_bin, kernel, iterations=2)
    kernel = np.ones((4, 4))
    image_bin = cv2.erode(image_bin, kernel, iterations=3)
    #image_bin = cv2.dilate(image_bin, kernel, iterations=6)

    image_bin = 255 - image_bin

    img = image_bin
    #cv2.imshow("izdvojene crvene case", imgOrg)

    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    # hiearchy mi zapravo i ne treba jer sam namestio na external konture
    regions_array = []

    ret = -1
    if(len(contours) != 0):
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if(w>150 and h>150):
                region = image_bin[y:y+h+1,x:x+w+1];
                regions_array.append([resize_region(region), (x,y,w,h)])
                color = (255,0,0)
                cv2.rectangle(imgOrg2,(x,y),(x+w,y+h),(255,0,0),3)
    '''
    ZELENA
    '''
    # da izdvojim samo crvenu boju pravljenjem maske
    # za rgb
    # lower = [70, 0, 0]
    # upper = [255, 70, 70]
    # za hsv
    lower = [50, 220, 50]
    upper = [90, 255, 200]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(imgOrg, lower, upper)
    output = cv2.bitwise_and(imgOrg, imgOrg, mask=mask)

    imgGray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    imgGray = 255 - imgGray
    image_bin = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 501, 1)
    kernel = np.ones((4, 4))
    image_bin = cv2.dilate(image_bin, kernel, iterations=2)
    kernel = np.ones((4, 4))
    image_bin = cv2.erode(image_bin, kernel, iterations=3)
    # image_bin = cv2.dilate(image_bin, kernel, iterations=6)

    image_bin = 255 - image_bin

    img = image_bin
    # cv2.imshow("izdvojene crvene case", imgOrg)

    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    # hiearchy mi zapravo i ne treba jer sam namestio na external konture
    regions_array = []

    ret = -1
    if (len(contours) != 0):
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (w > 100 and h > 100):
                region = image_bin[y:y + h + 1, x:x + w + 1];
                regions_array.append([resize_region(region), (x, y, w, h)])
                cv2.rectangle(imgOrg2, (x, y), (x + w, y + h), (0, 255, 0), 3)

    plt.imshow(imgOrg2)
    plt.show()

    return ret