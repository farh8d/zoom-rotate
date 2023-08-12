import firebase_admin
from firebase_admin import db,credentials

creds = credentials.Certificate('creds.json')

system_app = firebase_admin.initialize_app(creds, {'databaseURL': 'https://pirelly-system-requests.firebaseio.com'},name='status')
client_app = firebase_admin.initialize_app(creds, {'databaseURL': 'https://pirelly-client-request.firebaseio.com'},name='client')

def client(refrence,response):
	ref = db.reference(refrence,app=client_app)
	ref.update({'y_init': response })

def status(refrence,status,database,**kwargs):
    if database == 'client':
        ref = db.reference(refrence,app=client_app)
    elif database == 'system':
        ref = db.reference(refrence,app=system_app)

    if status == 'pending':
        ref.update({'status':status})
    elif status == 'ended':
        ref.update({'status':status})
    if status ==  'error':
        ref.update({'status':status,'errorMessage':kwargs['msg']})