from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import cv2



if __name__ == "__main__":
        all_shape  = [ Image.open(i).size for i in glob.iglob("Dataset/Images/*") ]
        all_width  = [ i[0] for i in all_shape]
        all_height = [ i[1] for i in all_shape]
        
        statistic = np.histogram(all_width, bins=np.arange(0, 1100, 100))
        sns.set()
        plt.subplot(1, 2, 1)
        plt.hist(all_width, bins=np.arange(0, 1100, 100))
        plt.title("Width")

        sns.set()
        plt.subplot(1, 2, 2)
        plt.hist(all_height, bins=np.arange(0, 2000, 100))
        plt.title("Height")
        plt.show()