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


#def download(file, document_id):
#    # Document details
#    document_path = file
#
#    try:
#        # Connect to the Couchbase cluster
#        cluster = Cluster(server_url, ClusterOptions(
#            authenticator=PasswordAuthenticator(accessname, secret),
#            transcoder=RawJSONTranscoder()
#        ))
#        bucket = cluster.bucket(bucket_name)
#        collection = bucket.default_collection()
#
#        # Retrieve the document
#        result = collection.get(document_id)
#
#        # Check if document retrieval was successful
#        if result.success:
#            # Document exists, get its content
#            document_data = result.content_as[bytes]
#
#            # Write the document content to a file
#            with open(document_path, 'wb') as file:
#                file.write(document_data)
#
#            print('Document downloaded successfully.')
#        else:
#            # Document does not exist
#            print('Document not found.')
#
#    except Exception as e:
#        print(f'Error accessing document: {e}')

# Dataset folder path
dataset_folder_path = 'dataset'

try:
    # Connect to the Couchbase cluster
    cluster = Cluster(server_url, ClusterOptions(
        authenticator=PasswordAuthenticator(accessname, secret),
        transcoder=RawJSONTranscoder()
    ))
    bucket = cluster.bucket(bucket_name)

    # Iterate through subdirectories in the dataset folder
    for subdir_name in os.listdir(dataset_folder_path):
        subdir_path = os.path.join(dataset_folder_path, subdir_name)

        # Check if the path is a subdirectory
        if os.path.isdir(subdir_path):
            # Sanitize the collection name
            collection_name = sanitize_collection_name(subdir_name)

            try:
                # Get the collection
                collection = bucket.scope("_default").collection(collection_name)

                # Create the collection
                collection_manager = bucket.collections()
                collection_manager.create_collection(collection_name)
                print(f"Collection '{collection_name}' created successfully.")

                # Iterate through files in the subdirectory
                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)

                    # Read the file
                    with open(file_path, 'rb') as file:
                        file_data = file.read()

                    # Insert the document
                    collection.upsert(filename, file_data)

                    print(f'File {filename} uploaded to collection {collection_name} successfully.')

            except CollectionNotFoundException:
                print(f"Collection '{collection_name}' does not exist.")
            except InternalSDKException as e:
                print(f"Failed to create collection: {e}")

    print('All files uploaded successfully.')

except Exception as e:
    print(f'Error uploading files: {e}')




#download('face_embeddings_done_4classes.npz', 'embeddings')
#download('svm_model_160x160.pkl', 'svm')

#download('max.jpg', 'bild')