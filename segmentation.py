from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms
import torch
import numpy as np

class SegmentationModel():
    """
    Pre-Trained Mask RCNN model on KITTI to identify moving vehicles among other 
    things for getting foreground features.
    """

    def __init__(self,device):
        # Load the pre-trained Mask R-CNN model
        self.segmentation_model = maskrcnn_resnet50_fpn(pretrained=True).to(device)

        # Set the model to evaluation mode
        self.segmentation_model.eval()

        # Define the image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def combine_masks(self,masks):
        """
        Combining/Agregating masks for all objects into pixel values 
        in grayscale range. Likely to change this to one fixed value, 
        because we will relate all these pixel values with a single 
        velocity value from the IMU of the vehicle.
        """
        masks_numpy = masks.detach().cpu().numpy()
        image_shape = masks_numpy[0][0].shape
        composite_image = np.zeros((image_shape))
        val_size = 1.0/masks_numpy.shape[0]
        val = val_size
        
        for i in range(masks.shape[0]):
            rows = np.where(masks_numpy[i][0]>0.01)[0]
            cols = np.where(masks_numpy[i][0]>0.01)[1]
            composite_image[rows,cols] = val
            val += val_size
        return composite_image
    
    def get_segmentation_masks(self,frames,sm_confidence_threshold=0.5):
        """
        Returns a black/white image with segmentation masks/white 
        regions for a given frame.
        """
        sm_prediction = self.segmentation_model(frames)
        images = []
        for i in range(len(sm_prediction)):
            sm_masks = sm_prediction[i]['masks']
            """
            Can do it only for objects detected as cars, but 
            currently doing it for all objects detected, to avoid 
            missing on moving objects that are incorrectly labeled
            """
            # Since we are merging all masks together, doesn't really affect the 
            # number of masks here because all are combined. Our goal is not 
            # to identify vehicles but instead get some sort of clutersing of 
            # pixels.
            # Not providing thresholding scores, because we want true regions at 
            # pixel-level and not bounding boxes.
            
            # scores = sm_prediction[i]['scores']
            # sm_qualifying_indices = torch.where(scores > sm_confidence_threshold)
            # sm_qualifying_masks = sm_masks[sm_qualifying_indices]

            composite_image = self.combine_masks(sm_masks)
            images.append(composite_image)
        
        # Returns BS X C X H X W
        # print("Image Tensor",torch.tensor(images).shape)
        return torch.tensor(images).float().to(self.device)