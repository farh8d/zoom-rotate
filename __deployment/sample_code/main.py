import os,json
from google.cloud import storage
import localpackages.settings as settings
from localpackages.status_handler import status
# TODO: import AI Function class
from localpackages.AI_function_example import Example_class


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.environ.get('MY_CODE_DIR', ''), 'creds.json')


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


def dir_checker(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def file_remover(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)


def handler(event, context):
    refrence = context.resource.split('refs')[-1]
    try:    # STAGE 2 
        status(refrence,'pending')
        data = json.loads(event['delta']['data'])
        filename = data['inputPath'].split('/')[-1]
    except Exception as e:
        status(refrence,'error',msg=f'ERROR:INITIALIZE ERROR -> {e}')
        return
    
    try:    # STAGE 3
        dir_checker(settings.LOCAL_INPUT_PATH)
        download_blob(data['inputPathBucketName'],data['inputPath'],settings.LOCAL_INPUT_PATH + '/' + filename)
    except Exception as e:
        file_remover(settings.LOCAL_INPUT_PATH + '/' + filename)
        status(refrence,'error',msg=f'ERROR: LOADING INPUT DATA -> {e}')
        return

    try:    # STAGE 4: AI-Function must be called here.
        global func
        dir_checker(settings.LOCAL_OUTPUT_PATH)
        if  func.run(settings.LOCAL_INPUT_PATH + '/' + filename, settings.LOCAL_OUTPUT_PATH + '/' + filename):
            status(refrence,'ended')
        else:
            status(refrence,'error',msg=f'ERROR: RUNNING AI -> NO DETECTION')
            return
    except Exception as e:
        status(refrence,'error',msg=f'ERROR: RUNNING AI -> {e}')
        return

    try:    # STAGE 5
        upload_blob(data['outPutBucketName'], settings.LOCAL_OUTPUT_PATH + '/' + filename,data['outputPath'])
    except Exception as e:
        file_remover(settings.LOCAL_OUTPUT_PATH + '/' + filename)
        status(refrence,'error',msg=f'ERROR: UPLOADING OUTPUT DATA -> {e}')
        return

    try:    # STAGE 6
        file_remover(settings.LOCAL_INPUT_PATH + '/' + filename)
        file_remover(settings.LOCAL_OUTPUT_PATH + '/' + filename)
        status(refrence,'ended')
    except Exception as e:
        status(refrence,'error',msg=f'ERROR: DATA CLEAN UP -> {e}')
        return
   
    return


try:    # STAGE 1: cold start, Loading models
    dir_checker(settings.TRAINED_MODEL_LOCAL_PATH)
    download_blob(settings.TRAINED_MODEL_GCS_BUCKET,settings.TRAINED_MODEL_GCS_PATH_GLASS,settings.TRAINED_MODEL_LOCAL_PATH + settings.TRAINED_MODEL_GLASS_FILENAME)
    # TODO: create ai function object here
    func = Example_class(settings.TRAINED_MODEL_LOCAL_PATH + '/' + settings.TRAINED_MODEL_GLASS_FILENAME)
except Exception as e:
    file_remover(settings.TRAINED_MODEL_LOCAL_PATH + settings.TRAINED_MODEL_GLASS_FILENAME)
    print(f'ERROR LOADING AI MODEL -> {e}')
    raise Exception("ERROR ON COLD START")