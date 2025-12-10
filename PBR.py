import cv2
import numpy as np


def apply_pbr(scene_img, glass_img, show_plot=True):
    """Apply glass overlay with blur effect (Physics-Based Rendering)."""
    glass_img_new = np.copy(glass_img)
    glass_img_resized = cv2.resize(glass_img_new,(scene_img.shape[1],scene_img.shape[0]))
    glass_img_resized_gray = cv2.cvtColor(glass_img_resized,cv2.COLOR_BGR2GRAY)

    crack_array = glass_img_resized_gray>10
    row_mask = np.where(crack_array)[0]
    col_mask = np.where(crack_array)[1]
    scene_img_blended = np.copy(scene_img)
    scene_img_blended[row_mask,col_mask] = glass_img_resized[row_mask,col_mask]

    if crack_array.dtype != np.uint8:
        crack_array = (crack_array * 255).astype(np.uint8)

    # Ensure crack_array has the same dimensions as the images
    crack_array_resized = np.copy(crack_array)

    # Get the indices where the mask is true
    points_to_blur = np.argwhere(crack_array_resized > 0)

    # Kernel size for Gaussian blur; must be odd
    kernel_size = (15, 15)
    half_kernel_size = (kernel_size[0] // 2, kernel_size[1] // 2)

    # Create a copy of the image to apply blur
    result_img = scene_img_blended.copy()

    for (y, x) in points_to_blur:
        # Define the region of interest
        y1, y2 = max(0, y - half_kernel_size[0]), min(scene_img_blended.shape[0], y + half_kernel_size[0] + 1)
        x1, x2 = max(0, x - half_kernel_size[1]), min(scene_img_blended.shape[1], x + half_kernel_size[1] + 1)
        roi = scene_img_blended[y1:y2, x1:x2]

        # Apply Gaussian Blur to this region
        blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)

        # Blend the blurred region back into the result image
        result_img[y1:y2, x1:x2] = blurred_roi

    if show_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,4))
        plt.imshow(result_img)
        plt.show()

    return result_img


# Original script (runs when executed directly)
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    glass_img = cv2.imread('glass.png')
    scene_img = cv2.imread('test-kitti.png')
    result_img = apply_pbr(scene_img, glass_img, show_plot=True)
