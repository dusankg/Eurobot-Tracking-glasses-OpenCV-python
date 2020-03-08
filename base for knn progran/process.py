# import libraries here
from __future__ import print_function
#import potrebnih biblioteka
import cv2
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# keras
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

#import winsound # biblioteka da dodam zvucni signal kad se zavrsi prokleto obucavanje na pentijumu...
#Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
plt.rcParams['figure.figsize'] = 16, 12

##### dodatne metode #####
def dilate(image):
    kernel = np.ones((4,1)) # idem samo po visini
    return cv2.dilate(image, kernel, iterations=6)
def erode(image):
    kernel = np.ones((2,1))
    return cv2.erode(image, kernel, iterations=1)

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
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
def matrix_to_vector(image):
    return image.flatten()
def prepare_for_ann(regions): # nemam blage veze sta tacno radi
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann
def convert_output(outputs):
    return np.eye(len(outputs))
def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]
def select_roi(image_orig, image_bin):      #prima originalnu i binarnu sliku, sa originalne bi trebalo da nalazi regione
    '''
    Prilagodjavam boje tako da vidim samo crvene case, zatim iscrtavam linije za nadjene konture
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    # hiearchy mi zapravo i ne treba jer sam namestio na external konture
    regions_array = []

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)

        region = image_bin[y:y+h+1,x:x+w+1];
        regions_array.append([resize_region(region), (x,y,w,h)])
        cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2]) #X_next - (X_current + W_current)
        region_distances.append(distance)


    return image_orig, sorted_regions, region_distances


def create_ann():
    '''
    Implementirati veštačku neuronsku mrežu sa 28x28 ulaznih neurona i jednim skrivenim slojem od 128 neurona.
    Odrediti broj izlaznih neurona. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    # Postaviti slojeve neurona mreže 'ann'
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    #ann.add(Dense(input_dim=256, activation='sigmoid'))
    ann.add(Dense(60, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    #ann.compile(optimizer='adam',
     ##           loss='sparse_categorical_crossentropy',
      #          metrics=['accuracy'])
    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=5000, batch_size=1, verbose=1, shuffle=False)

    return ann

def serialize_ann(ann):
    # serijalizuj arhitekturu neuronske mreze u JSON fajl
    model_json = ann.to_json()
    with open("serialized_model/neuronska.json", "w") as json_file:
        json_file.write(model_json)
    # serijalizuj tezine u HDF5 fajl
    ann.save_weights("serialized_model/neuronska.h5")


def load_trained_ann():
    try:
        # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
        json_file = open('serialized_model/neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        # ucitaj tezine u prethodno kreirani model
        ann.load_weights("serialized_model/neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        return None

def train_or_load_character_recognition_model(train_image_paths, serialization_folder):

    model = load_trained_ann()

    # prvo sam misli da stavim true ili false da li je linija zauzeta ili nije, ali bas i nece moci tako...
    possibleOutputs = ['true', 'false']

    return None
    if model == None:
        print("Zapoceto je treniranje modela!")

        # staviti slike iz kojih ucim i raditi da se prolazi for-om
        image_color = load_image('dataset/train/alphabet.png')
        img = image_bin(image_gray(image_color))
        img = dilate(img)
        selected_regions, letters, region_distances = select_roi(image_color.copy(), img)

        inputs = prepare_for_ann(letters)       # sortirani regioni po x osi
        outputs = convert_output(possibleOutputs)
        print('Broj prepoznatih regiona u slici:', len(letters))

        #model = train_ann(model, inputs1, outputs1)
        model = create_ann()
        model = train_ann(model, inputs, outputs)
        print("Duzina inputsa")
        print(len(inputs))
        cv2.imshow("image", selected_regions)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

        serialize_ann(model)

        # zvucni signal
        duration = 2000  # milliseconds
        freq = 440  # Hz
        #winsound.Beep(freq, duration)

    else:
        print("Model vec postoji i ucitan je!")
    return model

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
def getDominantColor(image):
    image = cv2.resize(image, (50, 250), interpolation=cv2.INTER_NEAREST)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=4)
    labels = clt.fit_predict(image)
    label_counts = collections.Counter(labels)
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    return list(dominant_color)

def extract_glasses_location(trained_model, image_path):

    # TODO - pronaci case obucenim modelom

    #image_color = load_image("dataset/train/alphabet.png")
    #image_color = load_image(image_path)

    imgOrg = cv2.imread(image_path)
    imgOrg = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2RGB)

    dominantColor = getDominantColor(imgOrg)
    if dominantColor[0] > 180:
        print('Nije zaklonjeno')

    else:
        print('Zaklonjeno')


    # da izdvojim samo crvenu boju pravljenjem maske
    lower = [100, 0, 0]
    upper = [255, 50, 80]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(imgOrg, lower, upper)
    output = cv2.bitwise_and(imgOrg, imgOrg, mask=mask)

    imgGray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    imgGray = 255 - imgGray
    image_bin = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 501, 1)
    kernel = np.ones((2, 1))
    image_bin = cv2.erode(image_bin, kernel, iterations=1)
    image_bin = cv2.dilate(image_bin, kernel, iterations=4)

    image_bin = 255 - image_bin

    img = image_bin
    #cv2.imshow("izdvojene crvene case", imgOrg)

    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    # hiearchy mi zapravo i ne treba jer sam namestio na external konture
    regions_array = []

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)

        region = image_bin[y:y+h+1,x:x+w+1];
        regions_array.append([resize_region(region), (x,y,w,h)])
        cv2.rectangle(imgOrg,(x,y),(x+w,y+h),(0,255,0),2)


    plt.imshow(imgOrg)
    plt.show()
    return "trebalo bi da se vrate koordinate casa, ili sta se budemo dogovorili"