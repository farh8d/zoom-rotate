import numpy as np
import cv2
from ultralytics import YOLO




class ZoomRotate:
    def __init__(self):
        try:
            self.model_yolo =  YOLO('localpackages/model/yolov8s_6_2023.pt') 
        except Exception as e :
            print("constructor Exception",e)
        
         
       


    def __detect_main_car(self , img):
        results = self.model_yolo(img , verbose=False , device="cpu")
        results = results[0].cpu().numpy().boxes.data

        if results.shape[0] == 0:
            return None
        
        # select cars and trucks
        indices = np.where((results[:, 5] != 2) & (results[:, 5] != 7))
        # Delete the rows where not car or truck
        cars = np.delete(results, indices, axis=0)
        
        diffs = cars[:, 3] - cars[:, 1]

        # Get the index of the row with the maximum difference (main car)
        max_index = np.argmax(diffs)

        # Get the row with the maximum difference
        biggest_car = cars[max_index]
       

        if cars.shape[0] == 0:
            return None
        dict = {
            "xmin":  biggest_car[0] ,
            "ymin":  biggest_car[1] ,
            "xmax":  biggest_car[2] , 
            "ymax":  biggest_car[3] ,
        }
        return dict
    

    
    def __make_new_image(self , img , bb , expected_car_heigh):
        # make black around the image for zoom out
        new_width = img.shape[1] + 20000
        new_height = img.shape[0] + 20000
        new_img = np.zeros((new_height, new_width, 3), np.uint8)
        new_img[10000:new_height-10000, 10000:new_width-10000] = img

        current_height = bb['ymax'] - bb['ymin']
        scale = current_height / expected_car_heigh
        cut_size_y , cut_size_x =  scale * img.shape[0] , scale * img.shape[1]
        car_center_x = int((bb['xmax'] + bb['xmin']) / 2) + 10000
        car_center_y = int((bb['ymax'] + bb['ymin']) / 2) + 10000

        cut_img = new_img[int(car_center_y - cut_size_y / 2):int(car_center_y + cut_size_y / 2) , int(car_center_x - cut_size_x / 2):int(car_center_x + cut_size_x / 2),:]
        return   cv2.resize(cut_img, ( img.shape[1] , img.shape[0]), interpolation=cv2.INTER_LINEAR)




    def  zoomIN_zoomOut(self , img , height_fraction = 0.45 ):
        # get cv2 image
        try:
            bb = self.__detect_main_car(img)  
        except Exception as e :
            print("detect_main_car Exception",e)
        
        # ckeck any car detection
        if bb is None:
            print("no car!!!")
            return img
        h = bb['ymax'] - bb['ymin']
        if (height_fraction - 0.01) * img.shape[0] < h < (height_fraction + 0.01) * img.shape[0]:
            return img
        
        try:
            final = self.__make_new_image(img, bb , int(height_fraction * img.shape[0]) ) 
        except Exception as e :
            print("make_new_image Exception",e)
         
        return final
    
    def rotate_image(self , img, angle = 0):
        # get cv2 image
        # this function get an image and angle then rotate image by this angle
        height, width = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
        return rotated_image

        


