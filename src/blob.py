import os, uuid
from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
    __version__,
)

connect_str = "DefaultEndpointsProtocol=https;AccountName=stg4hyunji;AccountKey=F/zcS4mSz2LsS9rEqjkrtV4oed5G9Kwt7TqfVeNr1dJwbrTdC9l/bFlkyzELG7YlawX7QPYOdtdHhZAFk/GBhg==;EndpointSuffix=core.windows.net"
container_name = "hyunji"  # Enter your continaer name from azure blob storage here!

# Create the BlobServiceClient object which will be used to create a container client
blob_service_client = BlobServiceClient.from_connection_string(connect_str)


def get_blob_info():
    connect_str = "DefaultEndpointsProtocol=https;AccountName=stg4hyunji;AccountKey=F/zcS4mSz2LsS9rEqjkrtV4oed5G9Kwt7TqfVeNr1dJwbrTdC9l/bFlkyzELG7YlawX7QPYOdtdHhZAFk/GBhg==;EndpointSuffix=core.windows.net"
    container_name = "hyunji"  # Enter your continaer name from azure blob storage here!
    return connect_str, container_name 

def upload_file_to_blob(upload_file_path, target=None):  # file path - >file path
    if target is None: 
          target = upload_file_path
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=target
    )
    print("\nUploading to Azure Storage as blob:\n\t" + upload_file_path)
    with open(upload_file_path, "rb") as data:
        blob_client.upload_blob(data)


def upload_directory_to_blob(
    upload_file_path, target=None, container_name="hyunji"
):  # directory name -> directory name
    print(f"\nUploading directory {upload_file_path} to Azure Storage {target}\n\t") 
    if target is None: target = upload_file_path
    files = os.listdir(upload_file_path)
    for dir in files:
        if "ipynb_checkpoints" in dir:
            continue
        file_name = os.path.join(upload_file_path, dir)
        if os.path.isfile(file_name):
           #print(f'upload [FILE] {file_name}')
           target_ = os.path.join(target, dir)
           blob_client = blob_service_client.get_blob_client(
               container=container_name, blob=target_
           )
           with open(file_name, "rb") as data:
               blob_client.upload_blob(data)
        elif os.path.isdir(file_name):
           print(f'== upload [DIR] {file_name} ==')
           upload_directory_to_blob(file_name)
           print(f'== DONE upload [DIR] {file_name} ==')

def download_file_from_blob(source, download_file_path):
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=source
    )
    print("\nDownloading blob to \n\t from container" + download_file_path)

    with open(download_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())


def download_directory_from_blob(source, download_directory_path):
    container_client = ContainerClient.from_connection_string(
        conn_str=connect_str, container_name=container_name
    )
    print(
        f"\nDownloading all blobs from the following directory {source} in container {container_name}"
    )
    blob_list = container_client.list_blobs()
    for blob in blob_list:
        if source in blob.name:
            blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=blob.name
            )
            os.makedirs(os.path.dirname(blob.name), exist_ok=True)
            with open(blob.name, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

if __name__ == "__main__":
   #upload_file_to_blob("requirements.txt")
   upload_directory_to_blob("config")
