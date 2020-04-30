import matplotlib.pyplot as plt
import numpy as np
import cv2

def image_segmentation(image):
        # image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        he_image = cv2.equalizeHist(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1_image = clahe.apply(he_image)
        
        blur = cv2.GaussianBlur(cl1_image,(5,5),0)
        # ret3, otsu_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_image = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 151, 4)
        # low_threshold, high_threshold = 140, 210
        # edges = cv2.Canny(otsu_image, low_threshold, high_threshold)

        # edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        # blur  = cv2.cvtColor(blur , cv2.COLOR_GRAY2RGB)

        # indices = np.where(np.all(edges == [255, 255, 255], axis=-1))
        # coords = zip(indices[0], indices[1])
        # for i, j in coords:
        #     edges[i, j] = [255, 255, 0]

        # orimage = cv2.bitwise_or(blur, edges)
        # image  = cv2.cvtColor(image , cv2.COLOR_GRAY2RGB)
        # orimage = cv2.bitwise_or(image, edges)
        return otsu_image    

def image_segmentation_kmeans(image, cluster=4):
        vectorized = image.reshape( (-1, 1) )
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts = 10 # ?????
        ret, label, center = cv2.kmeans(vectorized, cluster, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        segmentation_image = res.reshape((image.shape))

        def overlap(main, seg):
                main = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)
                seg = cv2.applyColorMap(seg, cv2.COLORMAP_JET)
                main = cv2.addWeighted(main, 0.8, seg, 0.2, 0)
                return main

        return segmentation_image, overlap(image, segmentation_image)


def contrast_enhancement_and_denoisy(image):
        he_image = cv2.equalizeHist(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1_image = clahe.apply(he_image)
        blur = cv2.GaussianBlur(cl1_image,(5,5),0)
        return blur


def plot_images(original, segmentation, overlap, cluster_num):
        num = len(original)
        row = 3


        figure_size = 15
        plt.figure(figsize=(figure_size,figure_size))
        for idx, (ori, seg, color) in enumerate(zip(original, segmentation, overlap)):
                plt.subplot(num, row, row * idx + 1 )
                plt.title('Original Image'), plt.xticks([]), plt.yticks([])
                plt.imshow(ori, cmap="gray")
                
                plt.subplot(num, row, row * idx + 2 )
                plt.title('Segmented Image when K = %i' % cluster_num)
                plt.xticks([]), plt.yticks([])
                plt.imshow(seg, cmap="gray")
                
                plt.subplot(num, row, row * idx + 3 )
                plt.title("Overlap Image")
                plt.xticks([]), plt.yticks([])
                plt.imshow(color)


        plt.show()

if __name__ == '__main__':
    # filename = "Data/000408/000408 102419 x/NN_191024_151623_BE78A8.PNG"
    filename = "Data/000408/000408 102419 x/NN_191024_151657_BE78BB.PNG"
    # filename = "Data/000411/000411 112119 x/NN_191101_150513_BEEE1B.PNG"
    # filename = "Data/004359/004359 030716 x/NN_160307_151345_9F2F6A.PNG"
    # filename = "Data/004359/004359 030716 x/NN_160307_151400_9F2F5C.PNG"

    original_image = cv2.imread(filename, 0)
    seg_image, colormap_image = image_segmentation_kmeans(original_image, cluster=6)
    
    
    new_image = contrast_enhancement_and_denoisy(original_image)
    seg_image_1, colormap_image_1 = image_segmentation_kmeans(new_image, cluster=6)
    

    ori_images = [ original_image, new_image ]
    seg_images = [ seg_image, seg_image_1 ]
    map_images = [ colormap_image, colormap_image_1]

    plot_images(ori_images, seg_images, map_images, 4)
  
