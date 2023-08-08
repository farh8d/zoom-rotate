from localpackages.ZoomRotate import ZoomRotate as ZR
import glob
import cv2




obj = ZR()

def func():
    i = 0 
    for im in glob.glob("octo_org_sampling/*"):
        i+=1
        img = cv2.imread(im)

        try:
            img = obj.rotate_image(img ,0 )
            img = obj.centerizer(img)
            img = obj.zoomIN_zoomOut(img , 0.55)  
        except Exception as e :
            print("usage Exception",e)

      
        cv2.imwrite("OUT1/"+im.split("\\")[-1] , img)





func()


