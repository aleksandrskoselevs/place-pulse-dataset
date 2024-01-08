import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision import transforms
import pandas as pd
from torchvision.transforms.functional import crop

import os
from os import listdir
from os.path import isfile, join

import zipfile
from tqdm import tqdm
import requests
import shutil

from sklearn.model_selection import train_test_split

def crop_google_logo(img):
    return crop(img, 0, 0, img.size[1]- 25, img.size[0])

transform_cnn = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(crop_google_logo),
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
])

transform_transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(crop_google_logo),
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

class PlacePulseDataset(Dataset):
    """
    A PyTorch dataset class for the Place Pulse 2.0 dataset.

    Args:
        dataframe (pd.DataFrame, optional): The dataframe containing the dataset information. If not provided, it will be loaded from the 'qscores_tsv_path'. Defaults to None.
        qscores_tsv_path (str, optional): The path to the tsv file containing the dataset information. Defaults to 'data/qscores.tsv'.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Defaults to None.
        img_dir (str, optional): The directory path where the dataset images are stored. Defaults to 'data/images/'.
        return_location_id (bool, optional): Whether to return the location ID along with the image and rating. Defaults to False.
        study_id (int, optional): The ID of the study to filter the dataset. Defaults to None.
        transform_only_image (bool, optional): Whether to apply the transformation only to the image. Defaults to True.
        split (str, optional): The split of the dataset to use. Can be 'train' or 'val'. Defaults to None.
    """
    def __init__(self, dataframe=None, qscores_tsv_path='data/qscores.tsv', 
                 transform=None, img_dir='data/images/', 
                 return_location_id=False, study_id=None, transform_only_image=True,
                 split=None):
        self.transform = transform
        self.dataset_folder_path = img_dir 
        self.return_location_id = return_location_id
        self.transform_only_image = transform_only_image
        
        if qscores_tsv_path and isinstance(dataframe, pd.DataFrame):
            raise ValueError("Please provide either 'qscores_path' or 'dataframe', but not both. These parameters are mutually exclusive.")
        
        if qscores_tsv_path:
            dataframe = pd.read_csv(qscores_tsv_path, sep='\t')
        
        if qscores_tsv_path or isinstance(dataframe, pd.DataFrame):
            self.dataframe = dataframe

            self.dataframe['trueskill.score'] = torch.FloatTensor(self.dataframe['trueskill.score'].values)            
        
        if study_id:
            self.dataframe = self.dataframe[self.dataframe['study_id'] == study_id]
        
        if split:
            train_df, val_df = train_test_split(self.dataframe, test_size=0.4)
            
            if split == 'train':
                self.dataframe = train_df
            elif split == 'val':
                self.dataframe = val_df

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx):
        location_id = self.dataframe.iloc[idx]['location_id']        
        img = self.get_img_by_location_id(location_id)
        rating = self.dataframe.iloc[idx]['trueskill.score']

        if self.transform_only_image:
            if self.return_location_id:
                return img, rating, location_id        
            return img, rating
        
        else:
            return self.transform((img, rating))
    
    def get_img_by_location_id(self, location_id):
        extension = '.jpg'
        img_name = f"{location_id}{extension}"
        img = io.imread(f'{self.dataset_folder_path}{img_name}')

        if self.transform and self.transform_only_image:
            img = self.transform(img)        
        
        return img
    
    def get_sample_by_location_id(self, location_id):
        img = self.get_img_by_location_id(location_id)
        row = self.dataframe[self.dataframe['location_id'] == location_id]
        rating = row['trueskill.score'].values[0]

        if self.return_location_id:
            return img, rating, location_id

        return img, rating
    
    @staticmethod
    def get_q_score_only_for_files_in_folder(q_scores: pd.DataFrame, folder_path):
        """
        Filters the given 'q_scores' dataframe to only include the files that are present in the specified folder.

        Args:
            q_scores (pd.DataFrame): The dataframe containing the q scores.
            folder_path (str): The path to the folder containing the files.

        Returns:
            pd.DataFrame: The filtered dataframe containing the q scores for the files in the folder.
        """
        file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        location_ids_from_existing_files = [os.path.splitext(file_name)[0] for file_name in file_names]
        q_scores_clean = q_scores[q_scores['location_id'].isin(location_ids_from_existing_files)]

        return q_scores_clean    
    
    @staticmethod
    def download_archive():
        url = "https://www.dropbox.com/s/grzoiwsaeqrmc1l/place-pulse-2.0.zip?dl=1"
        response = requests.get(url, stream=True)

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open("place-pulse-2.0.zip", "wb") as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong")  

    @staticmethod
    def extract_archive(zip_file_path='place-pulse-2.0.zip', destination_folder='data') -> None:
        """
        Extracts the specified zip file to the destination folder.

        Args:
            zip_file_path (str, optional): The path to the zip file. Defaults to 'place-pulse-2.0.zip'.
            destination_folder (str, optional): The path to the destination folder. Defaults to 'data'.
        """
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_list = [f for f in zip_ref.namelist() if not f.startswith('__MACOSX/')]
            
            with tqdm(total=len(file_list), desc='Extracting files', unit='file') as pbar:
                for file in file_list:
                    zip_ref.extract(file, destination_folder)
                    
                    pbar.update()
    
    @staticmethod
    def preprocess() -> None:
        """
        Preprocesses the images by copying them from the source directory to the destination directory,
        renaming them with a unique file ID, and then removing the original images folder.
        """
        source_dir = 'data/images/'
        file_names = os.listdir(source_dir)

        destination_dir = 'data/images_preprocessed/'
        os.makedirs(destination_dir, exist_ok=True)

        for file_name in tqdm(file_names, desc='Preprocessing images', unit='file'):
            # This id seems to be unique. Checked for duplicates with:
            # find . -type f -exec basename {} \; | sort | uniq -D
            uniqiue_file_id = file_name.split("_")[2]
            new_file_name = f'{uniqiue_file_id}.jpg'

            file_path = os.path.join(source_dir, file_name)
            destination_path = os.path.join(destination_dir, new_file_name)

            shutil.copyfile(file_path, destination_path)

        print('Removing original images folder')
        shutil.rmtree(source_dir)
        os.rename(destination_dir, source_dir)
    
    @staticmethod
    def load() -> None:
        """
        Loads the dataset by downloading, extracting, preprocessing, and deleting the archive.
        """
        if os.path.exists('data'):
            print('Error: The "data" folder already exists.')
            return

        PlacePulseDataset.download_archive()
        zip_file_path='place-pulse-2.0.zip'
        PlacePulseDataset.extract_archive(zip_file_path=zip_file_path)
        print('Deleting archive.')
        os.remove(zip_file_path)
        PlacePulseDataset.preprocess()

