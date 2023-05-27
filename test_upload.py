import os
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.transcoder import RawJSONTranscoder

# Couchbase connection parameters
server_url = 'couchbases://cb.a6cm4sbxf8i5qvak.cloud.couchbase.com'
bucket_name = 'Main'
accessname = 'FACE'
secret = 'Flempan123!'

def upload(file, document_id):
    
    document_path = file

    try:
        # Connect to the Couchbase cluster
        cluster = Cluster(server_url, ClusterOptions(
            authenticator=PasswordAuthenticator(accessname, secret),
            transcoder=RawJSONTranscoder()
        ))
        bucket = cluster.bucket(bucket_name)
        collection = bucket.default_collection()

        # Read the pickle file
        with open(document_path, 'rb') as file:
            document_data = file.read()

        # Insert the document
        collection.upsert(document_id, document_data)

        print('Document uploaded successfully.')

    except Exception as e:
        print(f'Error uploading document: {e}')


upload('Data/face_embeddings_done_4classes.npz', 'embeddings')
upload('Data/svm_model_160x160.pkl', 'svm')

upload('dataset/Max_Gulda/0.jpg', 'bild')
