import cv2
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import numpy as np

class KITTIFlowDataset(Dataset):
    def __init__(self, 
                 data, 
                 filters, 
                 scale_factor = 3,
                 image_height = 120,
                 image_width = 400,
                 aspect_ratio = 0.3
        ):
        """
        Initialization method for the dataset.

        Parameters:
        data (list, np.array, etc.): Contains the features.
        labels (list, np.array, etc.): Contains the labels.
        """
        self.data = data
        self.filters = filters
        self.scale_factor = scale_factor
        self.image_height = image_height
        self.image_width = image_width

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Generates one sample of data. 
        
        Parameters:
        index (int): The index of the sample to return.

        Returns:
        X (Tensor): The feature tensor.
        y (Tensor): The label tensor.
        noise (Tensor): The noise tensor

        One data item here is equivalent to (number of frames) selected together.
        Dimensions of one data item: (N x W x H x C)

        Default value of N -> 4

        Labels/Y will have filter and clean imge
        """
        # Select the sample
        # self.data - file path [path of data,noise]
        # index - file index
        # self.data[index] - gives list of paths for n number of frames 
        frame_paths = self.data[index]
        noise_path = self.filters[index]

        X,y,noise = self.load_data(frame_paths,noise_path)
        
        # Return the sample
        return X,y,noise
    
    def crop_to_size(self,image):
        h_diff = image.shape[0] - self.image_height
        w_diff = image.shape[1] - self.image_width
        # print('Image shape before cropping',image.shape)
        aH = h_diff//2
        if (h_diff % 2) == 0:
            bH = aH
        else:
            bH = aH + 1
        
        aW = w_diff//2
        if (w_diff % 2) == 0:
            bW = aW
        else:
            bW = aW+1
        # print('Image shape after cropping',image[aH:-bH,aW:-bW].shape)
        return image[aH:-bH,aW:-bW]

    def ready_image(self,image):
        # Converting to numpy arrays
        numpy_image = np.array(image)
        
        # Converting to open-cv images 
        image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        
        # Resizing the images to the desired scale factor
        image = cv2.resize(image,(image.shape[1]//self.scale_factor,image.shape[0]//self.scale_factor))
        
        # Normalize the images between 0 and 1
        image_normalized = cv2.normalize(image, None, 0, 1.0,cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        return image_normalized
    
    def load_data(self,frame_paths,noise_path):
        X = []
        y = []
        # Broken Glass/Condensation etc.
        noise = Image.open(noise_path).convert("RGBA")
        
        for i,path in enumerate(frame_paths):   
            clean = Image.open(path).convert("RGBA")
            if i==0:
                noise = noise.resize(clean.size, Image.Resampling.LANCZOS)
                noise_normalized = self.ready_image(noise)
                noise_normalized = self.crop_to_size(noise_normalized)
            
            # Blend the images
            blended_image = Image.alpha_composite(clean, noise)

            img_normalized = self.ready_image(blended_image)
            clean_normalized = self.ready_image(clean)
            img_normalized = self.crop_to_size(img_normalized)
            clean_normalized = self.crop_to_size(clean_normalized)
            
            X.append(img_normalized)
            y.append(clean_normalized)
        
        # Finally convert to tensors
        # Change shape N,H,W,C -> N,C,H,W for NN training.
        X_tensor = torch.tensor(np.array(X)).permute(0,3,1,2)
        y_tensor = torch.tensor(np.array(y))
        noise_tensor = torch.tensor(np.array(noise_normalized))

        # Return the sample
        return X_tensor, y_tensor, noise_tensor