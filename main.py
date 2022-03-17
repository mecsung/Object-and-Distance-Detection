import cv2
import numpy as np

# Distance constants
KNOWN_DISTANCE = 20  # INCHES
MOBILE_WIDTH = 3.0  # INCHES
MOBILE_HEIGHT = 6.0  # INCHES
PERSON_WIDTH = 16.0  # INCHES
LAPTOP_WIDTH = 13.0  # INCHES
KEYBOARD_WIDTH = 11.0  # INCHES
BOOK_WIDTH = 4.0  # INCHES

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for BBOX
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
# Defining fonts
FONTS = cv2.FONT_HERSHEY_COMPLEX

# Load COCO Dataset
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load SSD Algorithm and trained weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# Object Detection Function
def object_detector(img):
    classes, scores, boxes = net.detect(img, CONFIDENCE_THRESHOLD)
    # Creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        # label = "%s : %f" % (classNames[classid[0]], score)
        # draw rectangle on and label on object
        cv2.rectangle(img, box, color, 2)
        cv2.rectangle(img, (box[0] - 1, box[1] - 28), (box[0] + 150, box[1]), color, -1)
        cv2.putText(img, classNames[classid - 1], (box[0], box[1] - 10), FONTS, 0.5, (255, 255, 255), 1)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 77:  # cellphone class id
            data_list.append([classNames[classid - 1], box[2], (box[0], box[1] - 2)])
        elif classid == 1:  # person class id
            data_list.append([classNames[classid - 1], box[2], (box[0], box[1] - 2)])
        elif classid == 76:  # keyboard class id
            data_list.append([classNames[classid - 1], box[2], (box[0], box[1] - 2)])
        elif classid == 73:  # laptop class id
            data_list.append([classNames[classid - 1], box[2], (box[0], box[1] - 2)])
        elif classid == 84:  # book class id
            data_list.append([classNames[classid - 1], box[2], (box[0], box[1] - 2)])

        # returning list containing the object data.
    return data_list


# Getting Focal Length
def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# Getting Distance
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame

    return distance


# Reading the reference image from dir
ref_mobile = cv2.imread('ReferenceImages/image4.png')
ref_person = cv2.imread('ReferenceImages/image15.png')
ref_keyboard = cv2.imread('ReferenceImages/image8.jpg')
ref_laptop = cv2.imread('ReferenceImages/image9.jpg')
ref_book = cv2.imread('ReferenceImages/image6.jpg')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

keyboard_data = object_detector(ref_keyboard)
keyboard_width_in_rf = keyboard_data[0][1]

laptop_data = object_detector(ref_laptop)
laptop_width_in_rf = laptop_data[0][1]

book_data = object_detector(ref_book)
book_width_in_rf = book_data[0][1]

print(f"Mobile width in pixel: {mobile_width_in_rf}")

# Getting focal length
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_keyboard = focal_length_finder(KNOWN_DISTANCE, KEYBOARD_WIDTH, keyboard_width_in_rf)
focal_laptop = focal_length_finder(KNOWN_DISTANCE, LAPTOP_WIDTH, laptop_width_in_rf)
focal_book = focal_length_finder(KNOWN_DISTANCE, BOOK_WIDTH, book_width_in_rf)

cap = cv2.VideoCapture('SampleVideos/book.mp4')

while True:
    ret, frame = cap.read()
    # frame = cv2.imread('ReferenceImages/image9.jpg')

    data = object_detector(frame)
    for d in data:
        if d[0] == 'cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'keyboard':
            distance = distance_finder(focal_person, KEYBOARD_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'laptop':
            distance = distance_finder(focal_person, LAPTOP_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'book':
            distance = distance_finder(focal_person, BOOK_WIDTH, d[1])
            x, y = d[2]
        cv2.rectangle(frame, (x - 1, y - 3), (x + 150, y + 23), BLACK, -1)
        cv2.putText(frame, f'Dist: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, GREEN, 1)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
