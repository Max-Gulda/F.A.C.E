from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
import os
import base64
#pip install couchbase

def init():
    # Set the admin credentials
    admin_username = 'admin'
    admin_password = 'f.a.c.e.flempan'

    # Set the Couchbase connection parameters
    couchbase_url = 'couchbase://167.172.105.85:5984'
    bucket_name = 'face'

    try:
        # Connect to the Couchbase cluster
        cluster = Cluster(couchbase_url, ClusterOptions(
            authenticator=PasswordAuthenticator(admin_username, admin_password)
        ))

        # Open the face bucket
        bucket = cluster.bucket(bucket_name)
        collection = bucket.default_collection()

        # Fetch all documents in the bucket
        result = cluster.query(f'SELECT * FROM `{bucket_name}`')
        for row in result:
            doc = row[bucket_name]

            if 'type' in doc:
                # Create the local folder
                if not os.path.exists("Data"):
                    os.makedirs("Data")
                for attachment_name in doc['_attachments']:
                    attachment_info = doc['_attachments'][attachment_name]
                    if attachment_info.get('stub'):
                        # If the attachment is a stub, fetch the full data
                        attachment_data = bucket.get_attachment(row.doc_id, attachment_name)
                        attachment_info['data'] = attachment_data.content
                    # Save other attachments to a file
                    with open(os.path.join("Data", attachment_name), 'wb') as f:
                        f.write(attachment_info['data'])
                
                print("Training downloaded")
            
            elif 'fullName' in doc:
                # Create the local folder
                folder_name = f"{doc['fullName'].replace(' ', '_')}"
                if not os.path.exists("dataset/"+folder_name):
                    os.makedirs("dataset/"+folder_name)

                # Download the attachments and images
                for attachment_name in doc['_attachments']:
                    attachment_info = doc['_attachments'][attachment_name]
                    if attachment_info.get('stub'):
                        # If the attachment is a stub, fetch the full data
                        attachment_data = bucket.get_attachment(row.doc_id, attachment_name)
                        attachment_info['data'] = attachment_data.content
                    if attachment_name.lower().endswith('.jpg'):
                        # Save the image to a file
                        with open(os.path.join("dataset/"+folder_name, attachment_name), 'wb') as f:
                            # Encode the image data as base64 and write to file
                            image_base64 = base64.b64encode(attachment_info['data']).decode('utf-8')
                            f.write(base64.b64decode(image_base64))
                    else:
                        # Save other attachments to a file
                        with open(os.path.join("dataset/"+folder_name, attachment_name), 'wb') as f:
                            f.write(attachment_info['data'])

                # Print a message to indicate success
                print(f"Data for '{doc['fullName']}' has been downloaded to '{'dataset/'+folder_name}'.")
        
            else:
                print(f"No required field found for document {row.doc_id}.")
    
    except Exception as e:
        print(f"Error accessing Couchbase: {e}")


# Call the initialization function
init()
