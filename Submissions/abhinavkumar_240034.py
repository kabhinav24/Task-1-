import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("mario.png")

if img is None:
    print("image not found so making blank image")
    img = np.ones((500,500,3), dtype=np.uint8) * 255


print("step 1 : converting image to gray")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


print("step 2 : smoothing image using simple filter")

k = 3
kernel = np.ones((k,k)) / 9   

smooth_img = cv2.filter2D(gray_img, -1, kernel)


print("step 3 : finding gradients using prewitt operator")

prewittx = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

prewitty = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

gx = cv2.filter2D(smooth_img, cv2.CV_64F, prewittx)
gy = cv2.filter2D(smooth_img, cv2.CV_64F, prewitty)


print("step 4 : calculating gradient magnitude")

mag = np.sqrt(gx*gx + gy*gy)

mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
mag_norm = mag_norm.astype(np.uint8)


print("step 5 : applying threshold to get edges")

th = 50   
_, edge = cv2.threshold(mag_norm, th, 255, cv2.THRESH_BINARY)

def show(imgs, names):
    fig, ax = plt.subplots(1, len(imgs), figsize=(12,4))
    
    if len(imgs) == 1:
        ax = [ax]
    
    for i in range(len(imgs)):
        if len(imgs[i].shape) == 2:
            ax[i].imshow(imgs[i], cmap='gray')
        else:
            ax[i].imshow(imgs[i])
        ax[i].set_title(names[i])
        ax[i].axis("off")
    
    plt.show()



show(
    [img, edge],
    ["original image", "final edge image"]
)
