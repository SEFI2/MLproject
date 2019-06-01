import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
import argparse

DEFAULT_IMAGE_PATH = 'images/car1.jpg'

def find_cars(args):
    car_cascade = cv2.CascadeClassifier('cars.xml')
    img = cv2.imread(args.image_path, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    return cars, img

def find_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3, 0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def find_lines(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


# Draws frames for the cars
def show_frames(cars, img): 
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


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
    
    find_corners(img)
    find_lines(img)
    show_frames(cars, img)
    show_pos(cars)

    print ("Cars detected: {0}".format(len(cars)))
    
