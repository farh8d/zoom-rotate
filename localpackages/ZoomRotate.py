import numpy as np
import cv2
import os
from ultralytics import YOLO
from google.cloud import storage




class ZoomRotate:
    def __init__(self):
        # model_path = '/tmp/yolov8s_6_2023.pt'                           # for google cloud
        model_path = 'localpackages/model/yolov8s_6_2023.pt'              # for local run
        self.bb = {}


        if not os.path.exists(model_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.environ.get('MY_CODE_DIR', '/workspace'),
                                                                        'localpackages/credentials/pirelly360-ai-deployment-firebase-adminsdk-txp88-8d93127582.json')
            self.download_file_from_gcs()


        try:
            self.model_yolo =  YOLO(model_path) 
        except Exception as e :
            print("constructor Exception",e)



    


    def download_file_from_gcs(self, bucket_name="pirelly360-ai-deployment.appspot.com",
                               file_name="yolov8s_6_2023.pt", local_directory="/tmp/",
                               new_file_name=None):
        """
        Downloads a file from Google Cloud Storage and saves it to a local directory,
        but only if the file does not already exist in the local directory.

        Parameters:
        bucket_name (str): the name of the Google Cloud Storage bucket.
        file_name (str): the path of the file in the Google Cloud Storage bucket.
        local_directory (str): the local directory where the file should be saved.
        new_file_name (str, optional): the new name for the downloaded file. If not specified,
            the original file name will be used.

        Returns:
        None
        """
        # Initialize a client object
        client = storage.Client()

        # Get the bucket and file objects
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Set the destination file name
        if new_file_name is None:
            new_file_name = os.path.basename(file_name)
        destination_file_name = os.path.join(local_directory, new_file_name)

        # Download the file only if it doesn't already exist in the local directory
        if not os.path.exists(destination_file_name):
            # Create the local directory if it doesn't exist
            if not os.path.exists(local_directory):
                os.makedirs(local_directory)

            # Download the file to the local directory with the new name
            blob.download_to_filename(destination_file_name)




    
        
         
       


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
    



    def centerizer(self , img ):
        try:
            self.bb = self.__detect_main_car(img)  
        except Exception as e :
            print("detect_main_car Exception",e)

        if self.bb is None:
            print("no car!!!")
            return img

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

        return shifted_img 
    



    
    def __make_new_image(self , img , expected_car_heigh):
        # make black around the image for zoom out
        new_width = img.shape[1] + 20000
        new_height = img.shape[0] + 20000
        new_img = np.zeros((new_height, new_width, 3), np.uint8)
        new_img[10000:new_height-10000, 10000:new_width-10000] = img

        current_height = self.bb['ymax'] - self.bb['ymin']
        scale = current_height / expected_car_heigh
        cut_size_y , cut_size_x =  scale * img.shape[0] , scale * img.shape[1]
        car_center_x = int((self.bb['xmax'] + self.bb['xmin']) / 2) + 10000
        car_center_y = int((self.bb['ymax'] + self.bb['ymin']) / 2) + 10000

        cut_img = new_img[int(car_center_y - cut_size_y / 2):int(car_center_y + cut_size_y / 2) , int(car_center_x - cut_size_x / 2):int(car_center_x + cut_size_x / 2),:]
        return   cv2.resize(cut_img, ( img.shape[1] , img.shape[0]), interpolation=cv2.INTER_LINEAR)




    def  zoomIN_zoomOut(self , img , height_fraction = 0.45  ):
        # get cv2 image
        # ckeck any car detection
        if self.bb is None:
            print("no car!!!")
            return img
        h = self.bb['ymax'] - self.bb['ymin']
        if (height_fraction - 0.01) * img.shape[0] < h < (height_fraction + 0.01) * img.shape[0]:
            return img
        
        try:
            final = self.__make_new_image(img , int(height_fraction * img.shape[0]) ) 
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

        


