from localpackages.ZoomRotate import ZoomRrotate as ZR
import glob
import cv2




obj = ZR()

def func():
    i = 0 
    for im in glob.glob("IMG_4384__8483/*"):
        i+=1
        img = cv2.imread(im)

        try:
            img = obj.rotate_image(img ,5 )
            img = obj.zoomIN_zoomOut(img , 0.40)  
        except Exception as e :
            print("usage Exception",e)

      
        cv2.imwrite("zoom_ultra/"+im.split("\\")[-1] , img)





func()


