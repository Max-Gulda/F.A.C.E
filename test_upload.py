import os
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.transcoder import RawJSONTranscoder, JSONTranscoder
import base64

# Couchbase connection parameters
server_url = 'couchbases://cb.a6cm4sbxf8i5qvak.cloud.couchbase.com'
bucket_name = 'Main'
accessname = 'FACE'
secret = 'Flempan123!'

def upload(files, document_id):
    try:
        # Connect to the Couchbase cluster
        cluster = Cluster(server_url, ClusterOptions(
            authenticator=PasswordAuthenticator(accessname, secret),
            transcoder=JSONTranscoder()
        ))
        bucket = cluster.bucket(bucket_name)
        collection = bucket.default_collection()

        # Create a dictionary to hold the file data
        file_data_dict = {}

        # Read each JPG file and store base64 encoded data in the dictionary
        for file in files:
            with open(file, 'rb') as file_obj:
                file_data = file_obj.read()
                file_data_base64 = base64.b64encode(file_data).decode('utf-8')
                file_name = file.split('/')[-1]
                file_data_dict[file_name] = file_data_base64

        # Insert the dictionary of file data as a single document
        collection.upsert(document_id, file_data_dict)

        print('Document uploaded successfully.')

    except Exception as e:
        print(f'Error uploading document: {e}')





# List of JPG files to upload
files = ['dataset/Max_Gulda/0.jpg', 'dataset/Max_Gulda/1.jpg']

# Specify the document ID
document_id = 'max_bild'



# Call the modified upload function with the list of files
upload(files, document_id)

