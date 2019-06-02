import yaml
import numpy as np
import cv2
import sys
import argparse

# CONSTANTS
DEFAULT_PARKING_DATA_FILE = 'parking_data/parking_test_1.yml'
DEFAULT_VIDEO_PATH = 'videos/parking_test_2.mp4'
DEFAULT_CAR_CASCADE = 'models/car_classifier.xml'
DEFAULT_OUTPUT_PATH = 'output/output_video.avi'
LAPLACIAN_THRESHOLD = 2.8

WAIT_TIME = 1



def get_parser(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--parking_data_file', type=str, default=DEFAULT_PARKING_DATA_FILE)
    parser.add_argument('--video_path', type=str, default=DEFAULT_VIDEO_PATH)
    parser.add_argument('--mark_lot', type=bool, default=False)
    parser.add_argument('--car_cascade', type=str, default=DEFAULT_CAR_CASCADE) 
    parser.add_argument('--save_video', type=bool, default=False)
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(args)




# GLOBALS 
args = get_parser(sys.argv[1:])
car_cascade = None
refPt = []
data = []
image_to_crop = None
car_cascade = cv2.CascadeClassifier(args.car_cascade)
output_video = None 




def init_conf():
    open(args.parking_data_file, "a")


def is_car(img):
    cars = car_cascade.detectMultiScale(img, 1.1, 1)
    return cars != ()

def get_cars(img):
    return car_cascade.detectMultiScale(img, 1.1, 1)
            



def yaml_loader():
    with open(args.parking_data_file, "r+") as file_descr:
        data = yaml.load(file_descr)
        return data if data != None else []

def yaml_dump(data):
    with open(args.parking_data_file, "a") as file_descr:
        yaml.dump(data, file_descr)


def yaml_dump_write(data):
    with open(args.parking_data_file, "w") as file_descr:
        yaml.dump(data, file_descr)


def click_and_crop(event, x, y, flags, param):
    current_pt = {'points': []}
    global image_to_crop
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDBLCLK:
        refPt.append((x, y))
        cropping = False
    if len(refPt) == 4:
        if data == []:
            if yaml_loader() != None:
                data_already = len(yaml_loader())
            else:
                data_already = 0
        else:
            if yaml_loader() != None:
                data_already = len(data) + len(yaml_loader())
            else:
                data_already = len(data)

        cv2.line(image_to_crop, refPt[0], refPt[1], (255, 0, 0), 1)
        cv2.line(image_to_crop, refPt[1], refPt[2], (255, 0, 0), 1)
        cv2.line(image_to_crop, refPt[2], refPt[3], (255, 0, 0), 1)
        cv2.line(image_to_crop, refPt[3], refPt[0], (255, 0, 0), 1)

        temp_lst1 = list(refPt[2])
        temp_lst2 = list(refPt[3])
        temp_lst3 = list(refPt[0])
        temp_lst4 = list(refPt[1])

        current_pt['points'] = [temp_lst1, temp_lst2, temp_lst3, temp_lst4]
        data.append(current_pt)
        # data_already+=1
        refPt = []

def mark_parking_lots(args, img):
    global image_to_crop
    image_to_crop = cv2.resize(img, None, fx=1, fy=1)
    clone = image_to_crop.copy()
    cv2.namedWindow("Double click to mark points")
    cv2.imshow("Double click to mark points", image_to_crop)
    cv2.setMouseCallback("Double click to mark points", click_and_crop)
     # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("Double click to mark points", image_to_crop)
        key = cv2.waitKey(1) & 0xFF
        if cv2.waitKey(33) == 27:
            break
     # data list into yaml file
    if data != []:
        yaml_dump(data)
    cv2.destroyAllWindows()   

def helper_load_parking_data():
    parking_data = yaml_loader()
    if parking_data == None:
        parking_data = []

    parking_contours = []
    parking_bounding_rects = []
    parking_mask = []

    for park in parking_data:
        points = np.array(park['points'])
        rect = cv2.boundingRect(points)
        points_shifted = points.copy()
        points_shifted[:,0] = points[:,0] - rect[0] # shift contour to region of interest
        points_shifted[:,1] = points[:,1] - rect[1]
        parking_contours.append(points)
        parking_bounding_rects.append(rect)
        mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1,
                                    color=255, thickness=-1, lineType=cv2.LINE_8)
        mask = mask==255
        parking_mask.append(mask)
    
    parking_status = [False]*len(parking_data)
    parking_buffer = [None]*len(parking_data)
    
    return parking_data, parking_contours, parking_bounding_rects, parking_mask, parking_status, parking_buffer


#def laplacian_check()

def run_algo():
    global output_video
    cap = cv2.VideoCapture(args.video_path)
    if args.save_video == True:
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        output_video = cv2.VideoWriter(args.output_path, fourcc, 25.0,(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    parking_data_motion = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(5,19))
    
    parking_data_loaded = False

    while(cap.isOpened()):
        video_cur_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 # Current position of the video file in seconds
        video_cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) # Index of the frame to be decoded/captured next
        ret, frame_initial = cap.read()
        if ret == True:
            frame = cv2.resize(frame_initial, None, fx=1, fy=1)
        if ret == False:
            break
        if args.mark_lot and args.mark_lot == True:
            args.mark_lot = False
            mark_parking_lots(args, frame_initial)
            parking_data, parking_contours, parking_bounding_rects, parking_mask, parking_status, parking_buffer = helper_load_parking_data()
            parking_data_loaded = True
        if parking_data_loaded == False:
            parking_data_loaded = True
            parking_data, parking_contours, parking_bounding_rects, parking_mask, parking_status, parking_buffer = helper_load_parking_data()
         

        frame_blur = cv2.GaussianBlur(frame.copy(), (5,5), 3)
        frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
        frame_out = frame.copy()
       
        # detecting cars and vacant spaces
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            rect = parking_bounding_rects[ind]
            roi_gray = frame_gray[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])] # crop roi for faster calcluation
            cars_within = get_cars(roi_gray)
            #print ("CARS: ID = {0}, cars = {1}".format(ind, cars_within))
            
            laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
            points[:,0] = points[:,0] - rect[0] # shift contour to roi
            points[:,1] = points[:,1] - rect[1]
            delta = np.mean(np.abs(laplacian * parking_mask[ind]))
            status = delta < LAPLACIAN_THRESHOLD

            # If detected a change in parking status, save the current time
            if status != parking_status[ind] and parking_buffer[ind]==None:
                parking_buffer[ind] = video_cur_pos
                change_pos = video_cur_pos
            # If status is still different than the one saved and counter is open
            elif status != parking_status[ind] and parking_buffer[ind]!=None:
                if video_cur_pos - parking_buffer[ind] > WAIT_TIME:
                    parking_status[ind] = status
                    parking_buffer[ind] = None
            # If status is still same and counter is open
            elif status == parking_status[ind] and parking_buffer[ind]!=None:
                parking_buffer[ind] = None

        # changing the color on the basis on status change occured in the above section and putting numbers on areas
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            if parking_status[ind]:
                color = (0,255,0)
                rect = parking_bounding_rects[ind]
                roi_gray_ov = frame_gray[rect[1]:(rect[1] + rect[3]),
                               rect[0]:(rect[0] + rect[2])]  # crop roi for faster calcluation
                res = is_car(roi_gray_ov)
                if res:
                    parking_data_motion.append(parking_data[ind])
                    # del parking_data[ind]
                    color = (0,0,255)
            else:
                color = (0,0,255)
            
            cv2.drawContours(frame_out, [points], contourIdx=-1,
                                 color=color, thickness=2, lineType=cv2.LINE_8)
            
            

        if parking_data_motion != []:
            for index, park_coord in enumerate(parking_data_motion):
                points = np.array(park_coord['points'])
                color = (0, 0, 255)
                recta = parking_bounding_rects[ind]
                roi_gray1 = frame_gray[recta[1]:(recta[1] + recta[3]),
                                recta[0]:(recta[0] + recta[2])]  # crop roi for faster calcluation
                fgbg1 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
                roi_gray1_blur = cv2.GaussianBlur(roi_gray1.copy(), (5, 5), 3)
                fgmask1 = fgbg1.apply(roi_gray1_blur)
                bw1 = np.uint8(fgmask1 == 255) * 255
                bw1 = cv2.erode(bw1, kernel_erode, iterations=1)
                bw1 = cv2.dilate(bw1, kernel_dilate, iterations=1)
                (_, cnts1, _) = cv2.findContours(bw1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts1:
                    if cv2.contourArea(c) < 4:
                        continue
                    (x, y, w, h) = cv2.boundingRect(c)
                    classifier_result1 = is_car(roi_gray1)
                    if classifier_result1:
                        color = (0, 0, 255)  # Red again if car found by classifier
                    else:
                        color = (0,255, 0)
                classifier_result1 = is_car(roi_gray1)
                if classifier_result1:
                    # print(classifier_result)
                    color = (0, 0, 255)  # Red again if car found by classifier
                else:
                    color = (0, 255, 0)
                cv2.drawContours(frame_out, [points], contourIdx=-1,
                                     color=color, thickness=2, lineType=cv2.LINE_8)

        #cap.set(cv2.CAP_PROP_POS_FRAMES, video_cur_frame + 10)
        
        if args.save_video == True:
            output_video.write(frame_out)

        cv2.imshow('frame', frame_out)
        if cv2.waitKey(33) == 27:
            break

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()   

if __name__ == '__main__':
    init_conf()
    run_algo()

    # output video
    if args.save_video == True:
        output_video.release()





