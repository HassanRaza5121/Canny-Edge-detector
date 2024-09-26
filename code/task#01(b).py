from PIL import Image
import torch
import cv2
import numpy as np
class GradientAnalyzer:
    
    def __init__(self,path: str) -> None:
        self.path = path

    def load_image(self):

        image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        
        return image
    
    def derivative_mask(self):

        mask = cv2.Sobel
        
        return mask
    
    def Fx_Image(self):

        mask = self.derivative_mask()
        
        Image = self.load_image()
        
        fx_of_image = mask(Image, cv2.CV_64F, 1, 0, ksize = 3)
        
        return fx_of_image
    
    def Fy_Image(self):

        mask = self.derivative_mask()
        
        Image = self.load_image()
        
        fy_of_image = mask(Image,cv2.CV_64F, 0, 1, ksize = 3)
        
        return fy_of_image

    def gradient_magnitude(self):

        fy = self.Fy_Image()

        fx = self.Fx_Image()
        
        magnitude = np.sqrt( (fx**2) + (fy**2) )
        
        return magnitude
    
    def gradient_direction(self):

        fx = self.Fx_Image()

        fy = self.Fy_Image()

        theta = np.arctan((fy/fx)) # It contains the angle in radian.

        theta[np.isnan(theta)] = 0

        theta_degree = np.degrees(theta) # To see the degree values.

        return theta_degree
    
class non_maximum_supression(GradientAnalyzer):
    def __init__(self,path):
        self.path = path
    def non_max_supression(self):
        g_magnitude = super().gradient_magnitude()
        g_direction = super().gradient_direction()
        H, W = g_magnitude.shape
        g_direction = g_direction % 180 # ranges from 0-180 that's why 
        supressed = np.zeros((H,W))
        for i in range(1,H-1):
            for j in range(1,W-1):
                dir = g_direction[i,j]
                mag = g_magnitude[i,j]
                if (0 <= dir < 22.5) or (157.5 <= dir <= 180):
                    neighbor1 = g_magnitude[i, j - 1]
                    neighbor2 = g_magnitude[i, j + 1]
                elif 22.5 <= dir < 67.5:
                    neighbor1 = g_magnitude[i - 1, j + 1]
                    neighbor2 = g_magnitude[i + 1, j - 1]
                elif 67.5 <= dir < 112.5:
                    neighbor1 = g_magnitude[i - 1, j]
                    neighbor2 = g_magnitude[i + 1, j]
                elif 112.5 <= dir < 157.5:
                    neighbor1 = g_magnitude[i - 1, j - 1]
                    neighbor2 = g_magnitude[i + 1, j + 1]
                if mag >= neighbor1 and mag >= neighbor2:
                    supressed[i, j] = mag
                else:
                    supressed[i, j] = 0
        return supressed
    def hysteresis_thresholding(self):
        suppressed = self.non_max_supression()
        higher = np.percentile(suppressed,90)
        lower = 0.40*higher
        h,w = suppressed.shape
        for i in range(1,h-1):
            for j in range(1,w-1):
                if suppressed[i,j]<lower:
                    suppressed[i,j]=0
                elif suppressed[i,j]>=higher:
                    suppressed[i,j]=1
                elif suppressed[i,j]>lower and suppressed[i,j]<higher:
                    suppressed[i,j] = 0.5
        return suppressed
    def Canny_on_sobel(self):
        suppressed = self.non_max_supression()
        suppressed_8u = cv2.convertScaleAbs(suppressed)
        higher = np.percentile(suppressed_8u,90)
        lower = 0.40*higher
        edges = cv2.Canny(suppressed_8u,lower,higher)
        return edges
    





paths = ['E:/Computer Vision/Assignment 1/img1.jpg',
         'E:/Computer Vision/Assignment 1/img2.jpg',
         'E:/Computer Vision/Assignment 1/img3.jpg',
         'E:/Computer Vision/Assignment 1/img4.jpg',
         'E:/Computer Vision/Assignment 1/img5.jpeg',
         ]
for path in paths:

    obj = GradientAnalyzer(path)

    obj2 = non_maximum_supression(path)


    fy = obj.Fy_Image()

    fx = obj.Fx_Image()

    magnitude = obj.gradient_direction()
    directions = obj.gradient_direction()
    NMS = obj2.non_max_supression()
    hys_thres = obj2.hysteresis_thresholding()



    cv2.imshow('Images', hys_thres)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

 

