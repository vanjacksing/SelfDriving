import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_2_images(img1, img2, caption1, caption2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1, cmap='gray')
    ax1.set_title(caption1, fontsize=40)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(caption2, fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)