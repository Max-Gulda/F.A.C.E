import os
import re
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.exceptions import CollectionNotFoundException, InternalSDKException
from couchbase.transcoder import RawJSONTranscoder
import couchbase.exceptions

# Couchbase connection parameters
server_url = 'couchbases://cb.a6cm4sbxf8i5qvak.cloud.couchbase.com'
bucket_name = 'Main'
accessname = 'FACE'
secret = 'Flempan123!'

def sanitize_collection_name(collection_name):
    # Remove special characters and spaces from the collection name
    return re.sub(r'[^a-zA-Z0-9_]', '', collection_name)


def download(file, document_id):
    # Document details
    document_path = file

    try:
        # Connect to the Couchbase cluster
        cluster = Cluster(server_url, ClusterOptions(
            authenticator=PasswordAuthenticator(accessname, secret),
            transcoder=RawJSONTranscoder()
        ))
        bucket = cluster.bucket(bucket_name)
        collection = bucket.default_collection()

        # Retrieve the document
        result = collection.get(document_id)

        # Check if document retrieval was successful
        if result.success:
            # Document exists, get its content
            document_data = result.content_as[bytes]

            # Write the document content to a file
            with open(document_path, 'wb') as file:
                file.write(document_data)

            print('Document downloaded successfully.')
        else:
            # Document does not exist
            print('Document not found.')

    except Exception as e:
        print(f'Error accessing document: {e}')

# Dataset folder path




download('face_embeddings_done_4classes.npz', 'embeddings')
download('svm_model_160x160.pkl', 'svm')

download('max.jpg', 'bild')