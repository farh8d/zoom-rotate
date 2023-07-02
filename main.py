from localpackages.ZoomRotate import ZoomRotate as ZR
import io
import numpy as np
import cv2
import requests
from flask import send_file,Request
from zipfile import ZipFile, ZIP_DEFLATED


def editor(request: Request):
    """

    :param request: It is a json file that include url of images like:
           {
            'images': [ "url/one",
                        "url/two",
                        ...
                      ],
            'height_fraction' : 0.5
            'gyroscopic_angles' : [0 , 0 , -5 , 0 , 6 , ...]
           }

    :return: a zip file include all images in batch
             flask send_file(zip file)
    """

    height_fraction = request.json['height_fraction']
    gyroscopic_angles = request.json['gyroscopic_angles']
    images = []
    for i in request.json['images']:

        bytImage = requests.get(i).content
       
        image_np = np.frombuffer(bytImage, dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        images.append(img)

    outputs = rotate_zoom_all_files(images , height_fraction , gyroscopic_angles)

    imgs = zipping(outputs)
    return send_file(imgs, mimetype='application/zip', as_attachment=True, download_name='images.zip')



def rotate_zoom_all_files(images ,  height_fraction , gyroscopic_angles):
    obj = ZR()

    output_imgs = []
    i = 0
    for img in images:
        try:
            img1 = obj.rotate_image(img ,gyroscopic_angles[i] )
            img1 = obj.zoomIN_zoomOut(img1 , height_fraction)  
            output_imgs.append(img1)
        except Exception as e :
            print("usage Exception",e)

        i+=1
        

    return output_imgs
        





def zipping(imgs):
    """
    make a zip file from output images and find name of
    the file with paths

    :param outputs: output of final result of zoom_rotate model
    :param paths: an array of images path or url
    :return: a zip file
    """
# assume 'image_list' is a list of cv2 images
    image_data = []
    for img in imgs:
        success, encoded_img = cv2.imencode('.jpg', img)
        if success:
            image_data.append(encoded_img)

    # create a BytesIO object to hold the compressed image data
    zip_data = io.BytesIO()

    # create a ZipFile object to write the compressed images to
    zip_file = ZipFile(zip_data, mode='a')

    # write each compressed image to the ZIP archive
    for i, img_data in enumerate(image_data):
        zip_file.writestr(f'{i}.jpg', img_data.tobytes())

    # close the ZipFile object
    zip_file.close()
    zip_data.seek(0)
    # get the compressed data as bytes
    return zip_data
