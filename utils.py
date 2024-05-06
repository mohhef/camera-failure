import cv2
from cv2_plt_imshow import cv2_plt_imshow
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def convert_to_gray(x):
    """
    Converting a given image tensor array to a gray scale image 
    tensor (Batched implementation)
    Primarily for use in RPCA
    """
    # Weights for converting RGB to grayscale
    # The weights are based on the luminosity method
    
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    weights = torch.repeat_interleave(weights,repeats=4,dim=0)
    
    
    # Applying weights: (4,3, H, W) x (4,3, 1, 1) -> (H, W)
    image_tensor_gray = torch.sum(x*weights, dim=1)
    
    # Convert to NumPy
    np_image_gray = image_tensor_gray.cpu().detach().numpy()
    print(np_image_gray.shape)
    return np_image_gray


def load_image(imfile=None,device='cpu',img=None):
    """
    Loading image and resizing. 
    *Not in use currently, may use it later, putting it here 
    just in case.
    """
    if imfile==None:
        if img==None:
            print('Error! Provide image path or image.')
            print('imfile -> Image Path; img -> img')
            return None
    else:
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = cv2.resize(img,(img.shape[0]//3,img.shape[0]//3))

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def calculate_optical_flow(x,visualize=False):
    # batched_implementation of Farneback algorithm
    # x -> batch_size, N, C=1, H, W
    batch_size,N,C,H,W = x.shape
    flow_results = []
    batched_results = []
    for i in range(batch_size):
        for j in range(N-1):
                
            f1 = x[i][j].permute(1,2,0).cpu().numpy()
            f2 = x[i][j+1].permute(1,2,0).cpu().numpy()

            if C!=1:
                f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
            else:
                f1 = f1.squeeze(-1)
                f2 = f2.squeeze(-1)
            
            flow = cv2.calcOpticalFlowFarneback(
                f1, 
                f2, 
                None,
                pyr_scale = 0.5, levels = 3, 
                winsize = 15, iterations=3, 
                poly_n = 5, poly_sigma = 1.2, 
                flags = 0
            )
            
            # Flow vector -> (H,W,2)
            flow_results.append(flow)


            if visualize:
                # Compute the magnitude and angle of the 2D vectors
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                mask = np.zeros((H,W,3))
                mask[..., 1] = 255
                mask[..., 0] = angle * 180 / np.pi / 2

                # Set image value according to the optical flow magnitude (normalized)
                mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)


                # Convert HSV to RGB (BGR) color representation
                rgb = cv2.cvtColor(np.uint8(mask), cv2.COLOR_HSV2BGR)
                plt.figure(figsize=(10,3))
                plt.axis('off')
                cv2_plt_imshow(rgb)
                plt.show()
        # Flow results -> Channel-first flow results 
        # (BS,N-1,H,W,C) -> (BS,N,C,H,W)
        batched_results.append(np.array(flow_results))
    return torch.tensor(batched_results).permute(0,1,4,2,3)