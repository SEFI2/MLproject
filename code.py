import cv2
from matplotlib import pyplot as plt
import sys
import argparse

DEFAULT_IMAGE_PATH = 'car1.jpg'

def find_cars(args):
    car_cascade = cv2.CascadeClassifier('cars.xml')
    img = cv2.imread(args.image_path, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    return cars, img


# Draws frames for the cars
def show_frames(cars, img): 
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    plt.figure(figsize=(10,20))
    plt.imshow(img)

# Show positions of each car in image
def show_pos(cars): 
    for (x, y, w, h) in cars:
        print ("CAR detected: X={0} Y={1} W={2} H={3}".format(x, y, w, h))

def get_parser(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, default=DEFAULT_IMAGE_PATH)
    return parser.parse_args(args)

if __name__ == '__main__':
    args = get_parser(sys.argv[1:])
    cars, img = find_cars(args)
    show_frames(cars, img)
    show_pos(cars)
    print ("Cars detected: {0}".format(len(cars)))
    
