import numpy as np
import cv2
import json
from PIL import Image




class Stabilizer2:


    def read_dict_from_json(self , file_path):
        with open(file_path, 'r') as json_file:
            dict = json.load(json_file)
            return dict
        


    def centerizer0(self , img , shift_x , shift_y):
        # Define the transformation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        if img.shape[2] == 4:
            borderValue = (255,255,255,0)
        else:
            borderValue = (255,255,255)

        # Apply the transformation to the image
        shifted_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]) ,  borderValue = borderValue , flags=cv2.INTER_LINEAR)

        return shifted_img




    def centerizer1(self , img , shift_x , shift_y):
        # Define the transformation matrix

        row1 = np.hstack(( cv2.flip(cv2.flip(img,1), 0) , cv2.flip(img, 0) , cv2.flip(cv2.flip(img,1), 0) ))
        row2 = np.hstack(( cv2.flip(img, 1) , img , cv2.flip(img, 1) ))
        row3 = np.hstack(( cv2.flip(cv2.flip(img,1), 0) , cv2.flip(img, 0) , cv2.flip(cv2.flip(img,1), 0) ))
        
        img1 = np.vstack((row1 , row2))
        img1 = np.vstack((img1 , row3))


        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        if img.shape[2] == 4:
            borderValue = (255,255,255,0)
        else:
            borderValue = (255,255,255)

        # Apply the transformation to the image
        shifted_img = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]) ,  borderValue = borderValue , flags=cv2.INTER_LINEAR)

        shifted_img = shifted_img[img.shape[0]:img.shape[0]*2  , img.shape[1]:img.shape[1]*2  , :]
        return shifted_img 



    def __make_new_image0(self , img , expected_car_heigh , bb):
        # make black around the image for zoom out
        new_width = img.shape[1] + 8000
        new_height = img.shape[0] + 8000
        new_img = np.ones((new_height, new_width, img.shape[2]), np.uint8) * 255
        if img.shape[2] == 4:
            new_img[:,:,3] = 0
        new_img[4000:new_height-4000, 4000:new_width-4000] = img

        current_height = bb['ymax'] - bb['ymin']
        scale = current_height / expected_car_heigh
        cut_size_y , cut_size_x =  scale * img.shape[0] , scale * img.shape[1]
        car_center_x = int((bb['xmax'] + bb['xmin']) / 2) + 4000
        car_center_y = int((bb['ymax'] +bb['ymin']) / 2) + 4000

        cut_img = new_img[int(car_center_y - cut_size_y / 2):int(car_center_y + cut_size_y / 2) , int(car_center_x - cut_size_x / 2):int(car_center_x + cut_size_x / 2),:]
        return   cv2.resize(cut_img, ( img.shape[1] , img.shape[0]), interpolation=cv2.INTER_LINEAR)




    def zoomIN_zoomOut0(self , img , bb ,yaw, height_fraction , yolo_after_centerizer):
        # h = bb['ymax'] - bb['ymin']
        # if (height_fraction - 0.01) * img.shape[0] < h < (height_fraction + 0.01) * img.shape[0]:
        #     return img
        side_scale = 1
        # if  (70<yaw<110) or (250<yaw<290):
        #     side_scale = 0.9


        try:
            final = self.__make_new_image0(img , int(height_fraction * img.shape[0] * side_scale) ,yolo_after_centerizer ) 
        except Exception as e :
            raise Exception("make_new_image Exception",e)
         
        return final 




    def __make_new_image1(self , img , expected_car_heigh , bb):
        # make black around the image for zoom out
       
        imgs_row = np.hstack((img , img , img))
        new_img = np.vstack((imgs_row , imgs_row , imgs_row))
        if img.shape[2] == 4:
            new_img[:,:,3] = 0

        current_height = bb['ymax'] - bb['ymin']
        scale = 0.95
        cut_size_y , cut_size_x =  scale * img.shape[0] , scale * img.shape[1]
        car_center_x = int((bb['xmax'] + bb['xmin']) / 2) + img.shape[1]
        car_center_y = int((bb['ymax'] + bb['ymin']) / 2) + img.shape[0]

        cut_img = new_img[int(car_center_y - cut_size_y / 2):int(car_center_y + cut_size_y / 2) , int(car_center_x - cut_size_x / 2):int(car_center_x + cut_size_x / 2),:]
        return   cv2.resize(cut_img, ( img.shape[1] , img.shape[0]), interpolation=cv2.INTER_LINEAR)




    def zoomIN_zoomOut1(self , img , bb , height_fraction , yolo_after_centerizer):
        # h = bb['ymax'] - bb['ymin']
        # if (height_fraction - 0.01) * img.shape[0] < h < (height_fraction + 0.01) * img.shape[0]:
        #     return img
        
        try:
            final = self.__make_new_image1(img , int(height_fraction * img.shape[0]) ,yolo_after_centerizer ) 
        except Exception as e :
            raise Exception("make_new_image Exception",e)
         
        return final 
    





 

    def rotate_image0(self , img, angle = 0):
        # get cv2 image
        # this function get an image and angle then rotate image by this angle
        height, width = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

        if img.shape[2] == 4:
            borderValue = (255,255,255,0)
        else:
            borderValue = (255,255,255)

        rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height) , borderValue = borderValue , flags=cv2.INTER_LINEAR)
        return rotated_image





    def rotate_image1(self , img, angle = 0):
        # get cv2 image
        # this function get an image and angle then rotate image by this angle


        row1 = np.hstack(( cv2.flip(cv2.flip(img,1), 0) , cv2.flip(img, 0) , cv2.flip(cv2.flip(img,1), 0) ))
        row2 = np.hstack(( cv2.flip(img, 1) , img , cv2.flip(img, 1) ))
        row3 = np.hstack(( cv2.flip(cv2.flip(img,1), 0) , cv2.flip(img, 0) , cv2.flip(cv2.flip(img,1), 0) ))
        
        img1 = np.vstack((row1 , row2))
        img1 = np.vstack((img1 , row3))


        height, width = img1.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

        if img.shape[2] == 4:
            borderValue = (255,255,255,0)
        else:
            borderValue = (255,255,255)

        rotated_image = cv2.warpAffine(img1, rotation_matrix, (width, height) , borderValue = borderValue , flags=cv2.INTER_LINEAR)

        rotated_image = rotated_image[img.shape[0]:img.shape[0]*2  , img.shape[1]:img.shape[1]*2  , :]


        return rotated_image
    


    def shiftingUp(self , img , yaw):

        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        height = img.shape[0]
        scale = 0.02
        x = np.linspace(0, 360, 360)
        y =  ((np.sin((x - 45) / 180 * 2 * np.pi) ) + 1)/2  *  (height * scale)
        shift_number = y[int(yaw)]

        img2 = Image.fromarray(img)
        img1 = img + 0
        img1[:,:,:3] = 255
        if img.shape[2] == 4:
            img1[:,:,3] = 0

        img1 = Image.fromarray(img1)
        img2 = img2.resize((img1.width, img1.height))
        img1.paste(img2, (0, int(-shift_number)), img2)
        return np.array(img1)


 


    def run(self , img_address , json_address ,  output_address ):  # Done: return True  Failed: throw exception
        try:
            dict = self.read_dict_from_json(json_address)
            angle = eval(dict["angle"])
            shift_X = eval(dict["centerizer_shift_x"])
            shift_Y = eval(dict["centerizer_shift_y"])
            yolo_after_centerizer = eval(dict["centerizer_bb"])
            height_fraction = eval(dict["height_fraction"])
            mode = eval(dict["mode"])
            yaw = eval(dict["yaw"])
        except Exception as e :
            raise Exception("stab run method Exception - > read json file",e)

        try:
            img = cv2.imread(img_address , cv2.IMREAD_UNCHANGED)  
            if mode == 0 or mode == 3:
                img1 = self.rotate_image0(img , -angle)
                img1 = self.centerizer0(img1 , shift_X , shift_Y)
                img1 = self.zoomIN_zoomOut0(img1 , yolo_after_centerizer ,yaw , height_fraction , yolo_after_centerizer)
                # img1 = self.shiftingUp( img1 , yaw)
            if mode == 1 or mode == 2 :
                img1 = self.rotate_image1(img , -angle)
                img1 = self.centerizer1(img1 , shift_X , shift_Y)
                img1 = self.zoomIN_zoomOut1(img1 , yolo_after_centerizer , height_fraction , yolo_after_centerizer)  

            img1[:,:,3] = cv2.threshold(img1[:,:,3], 254, 255, cv2.THRESH_BINARY)[1]
            
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            # img1[:,:,3] = cv2.erode(img1[:,:,3], kernel)
            cv2.imwrite(output_address , img1) 

        except Exception as e :
            raise Exception("stab run method Exception",e)
        
        return True
        










