"""
Salt and pepper noise degradation module.
"""

import numpy as np

def salt_and_pepper_noise(
        img: np.ndarray, 
        amount: float = 0.05, # Proportion of image pixels to replace with noise
        salt_vs_pepper: float = 0.5) -> np.ndarray:  # Proportion of salt vs pepper noise 
    """
    Apply salt and pepper noise to an image.

    Args:
        img: Input image as numpy array (H, W, C) in range [0, 255]
        amount: Proportion of image pixels to replace with noise (default 0.05)
        salt_vs_pepper: Proportion of salt vs pepper noise (default 0.5) \n
            its possible to only have salt or pepper by setting this to 0 or 1
    
    Returns:
        Noisy image with salt and pepper noise in range [0, 255]    
    
    Example:
        >>> img = cv2.imread('image.jpg')
        >>> noisy_img = salt_and_pepper_noise(img, amount=0.05, salt_vs_pepper=0.5)

    """


    out = np.copy(img)

    # Apply Salt
    num_salt = np.ceil(amount * img.size * salt_vs_pepper)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[tuple(coords)] = 255

    # Apply Pepper
    num_pepper = np.ceil(amount * img.size * (1.0 - salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    out[tuple(coords)] = 0
    
    return out

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test the function

    # Create a dummy image
    dummy_img = np.ones((256, 256, 3), dtype=np.uint8) * 128  # Gray image

    # Apply salt and pepper noise at different levels
    noisy_img_005_05 = salt_and_pepper_noise(dummy_img, amount=0.05, salt_vs_pepper=0.5)
    noisy_img_05_05 = salt_and_pepper_noise(dummy_img, amount=0.5, salt_vs_pepper=0.5)
    noisy_img_08_08 = salt_and_pepper_noise(dummy_img, amount=0.8, salt_vs_pepper=0.8)

    # Plot the results
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(dummy_img)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(noisy_img_005_05)
    axs[0, 1].set_title('Salt & Pepper Noise (0.05, 0.5)')
    axs[0, 1].axis('off')
    axs[1, 0].imshow(noisy_img_05_05)
    axs[1, 0].set_title('Salt & Pepper Noise (0.5, 0.5)')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(noisy_img_08_08)
    axs[1, 1].set_title('Salt & Pepper Noise (0.8, 0.8)')
    axs[1, 1].axis('off')

    plt.show()