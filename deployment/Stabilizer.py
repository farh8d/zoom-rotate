import numpy as np
import cv2
from ultralytics import RTDETR

from torchvision import transforms
import onnxruntime
from PIL import Image
import json



class Stabilizer:
    def __init__(self , car_detection_model_path , angle_estimation_model_path):
        
        self.bb = {}

        try:
            self.model_yolo =  RTDETR(car_detection_model_path) 
        except Exception as e :
            print("constructor Exception - yolo",e)

        try:
            self.angle_input_size = 420 
            self.angle_model = onnxruntime.InferenceSession(angle_estimation_model_path)
            self.angle_transforms =  transforms.Compose([
                transforms.Resize((self.angle_input_size , self.angle_input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        except Exception as e :
            raise Exception("constructor Exception - yolo ",e)



    
        
         
       


    def __detect_main_car(self , img):
        
        results = self.model_yolo.predict(img , verbose=False , device="cpu" , conf=.7)
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


        # car bounding box doesnt violate image axis
        if (dict["xmin"] < img.shape[1] * 0.03) or (dict["xmax"] > img.shape[1] * 0.97) or (dict["ymin"] < img.shape[0] * 0.03) or (dict["ymax"] > img.shape[0] * 0.97):
            return None

        if ((dict["xmax"] - dict["xmin"]) < img.shape[1] * 0.2) and ((dict["ymax"] - dict["ymin"]) < img.shape[0] * 0.2) :
            return None



        return dict
    



    def centerizer(self , img ):
        try:
            self.bb = self.__detect_main_car(img)  
        except Exception as e :
            raise Exception("centerizer -> detect_main_car Exception",e)

        if self.bb is None:
            raise Exception("centerizer -> detect_main_car No detection")

        shift_x = (img.shape[1] / 2) -  int((self.bb['xmax'] + self.bb['xmin']) / 2)
        shift_y = (img.shape[0] / 2) - int((self.bb['ymax'] + self.bb['ymin']) / 2)

        # Define the transformation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        # Apply the transformation to the image
        shifted_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        self.bb['xmax'] += shift_x
        self.bb['xmin'] += shift_x
        self.bb['ymax'] += shift_y
        self.bb['ymin'] += shift_y

        return shifted_img , shift_x , shift_y , self.bb
    




    



    def rotate_image(self , img, angle = 0):
        # get cv2 image
        # this function get an image and angle then rotate image by this angle
        height, width = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height) , borderValue=(255,255,255))
        return rotated_image
    


    def estimate_angle(self , img):
        try:
            bb = self.__detect_main_car(img)  
        except Exception as e :
            raise Exception("estimate angle -> detect_main_car Exception",e)
        if bb is None:
            raise Exception("estimate angle -> detect_main_car No detection")

        img1 = img[int(bb["ymin"]):int(bb["ymax"]) , int(bb["xmin"]):int(bb["xmax"])]
        img1 = Image.fromarray(img1)
        img1 = self.angle_transforms(img1)
        img1 = np.array(img1)
        img1 = np.expand_dims(img1, 0).astype(np.float32)
        return self.angle_model.run(None, {"input":img1})[0][0][0]





    def write_dict_to_json(self, dict, file_path):
        with open(file_path, 'w') as json_file:
            json.dump(dict, json_file)




    def run(self , img_address , output_address , yaw , z_angle = None  , height_fraction = 0.4 , mode = 0):  # Done: return True  Failed: throw exception
        dict = {}
    
        try:
            img = cv2.imread(img_address , cv2.IMREAD_UNCHANGED)

            if img.shape[2] == 4:
                img = img[:,:,0:3]
            # if z_angle is None:
            if True:
                angle = self.estimate_angle(img) 
            else:
                angle = z_angle

            img1 = self.rotate_image(img , -angle)
            img1 , shift_x , shift_y , centerizer_bb = self.centerizer(img1)

            dict["yaw"] = str(yaw)
            dict["angle"] = str(angle)
            dict["centerizer_shift_x"] = str(shift_x)
            dict["centerizer_shift_y"] = str(shift_y)
            dict["centerizer_bb"] = str(centerizer_bb)
            dict["height_fraction"] = str(height_fraction)
            dict["mode"] = str(mode)

            self.write_dict_to_json(dict , output_address)

        except Exception as e :
            raise Exception("stab run method Exception",e)
        
        return True
    



    
        


        

        


      

