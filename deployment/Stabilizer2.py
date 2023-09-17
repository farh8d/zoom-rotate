import numpy as np
import cv2
import json
from PIL import Image



class Stabilizer2:

    def read_dict_from_json(self , file_path):
        with open(file_path, 'r') as json_file:
            dict = json.load(json_file)
            return dict
        

    def centerizer(self , img , shift_x , shift_y):
        # Define the transformation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        if img.shape[2] == 4:
            borderValue = (255,255,255,0)
        else:
            borderValue = (255,255,255)

        # Apply the transformation to the image
        shifted_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]) ,  borderValue = borderValue)

        return shifted_img 


    def __make_new_image(self , img , expected_car_heigh , bb):
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
        return   cv2.resize(cut_img, ( img.shape[1] , img.shape[0]), interpolation=cv2.INTER_CUBIC)




    def zoomIN_zoomOut(self , img , bb , height_fraction , yolo_after_centerizer):
        h = bb['ymax'] - bb['ymin']
        if (height_fraction - 0.01) * img.shape[0] < h < (height_fraction + 0.01) * img.shape[0]:
            return img
        
        try:
            final = self.__make_new_image(img , int(height_fraction * img.shape[0]) ,yolo_after_centerizer ) 
        except Exception as e :
            raise Exception("make_new_image Exception",e)
         
        return final 
    


 




    def rotate_image(self , img, angle = 0):
        # get cv2 image
        # this function get an image and angle then rotate image by this angle
        # height, width = img.shape[:2]
        # rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

        # if img.shape[2] == 4:
        #     borderValue = (255,255,255,0)
        # else:
        #     borderValue = (255,255,255)

        # rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height) , borderValue = borderValue)
        # return rotated_image
        pil_image = Image.fromarray(img)

        rotated_image = pil_image.rotate(angle, resample=Image.BICUBIC, expand=True)

        return np.array(rotated_image)



    def run(self , img_address , json_address ,  output_address ):  # Done: return True  Failed: throw exception
        try:
            dict = self.read_dict_from_json(json_address)
            angle = eval(dict["angle"])
            shift_X = eval(dict["centerizer_shift_x"])
            shift_Y = eval(dict["centerizer_shift_y"])
            yolo_after_centerizer = eval(dict["centerizer_bb"])
            height_fraction = eval(dict["height_fraction"])
        except Exception as e :
            raise Exception("stab run method Exception - > read json file",e)

        try:
            img = cv2.imread(img_address , cv2.IMREAD_UNCHANGED)  
            img1 = self.rotate_image(img , -angle)
            img1 = self.centerizer(img1 , shift_X , shift_Y)
            img1 = self.zoomIN_zoomOut(img1 , yolo_after_centerizer , height_fraction , yolo_after_centerizer) 
            cv2.imwrite(output_address , img1) 

        except Exception as e :
            raise Exception("stab run method Exception",e)
        
        return True
        


        


