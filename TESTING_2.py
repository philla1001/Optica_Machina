import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from googleapiclient.http import MediaIoBaseDownload
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os.path
import pickle
from googleapiclient.discovery import build

# If modifying or downloading files, these are the required scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def authenticate_google_drive():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)


def download_image_from_drive(service, file_id, destination):
    request = service.files().get_media(fileId=file_id)
    with open(destination, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()


if __name__ == '__main__':
    service = authenticate_google_drive()
    # List files in your Google Drive to find the image ID
    results = service.files().list(pageSize=10, fields="files(id, name)").execute()
    items = results.get('files', [])
    if not items:
        print('No files found.')
    else:
        for item in items:
            print(f'{item["name"]} ({item["id"]})')

    # Example: Download a specific file using the file ID
    file_id = '1jEZKiMNAB5PtmqZM84Q_k8D15SEvwnPE'
    destination = 'C:\\pjc13\\PycharmProjects\\Optica_Machina_1.0.0\\Dataset_Pics'
    download_image_from_drive(service, file_id, destination)
