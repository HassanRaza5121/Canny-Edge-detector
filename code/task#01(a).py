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

        return theta
    




paths = ['E:/Computer Vision/Assignment 1/img1.jpg',
         'E:/Computer Vision/Assignment 1/img2.jpg',
         'E:/Computer Vision/Assignment 1/img3.jpg',
         'E:/Computer Vision/Assignment 1/img4.jpg',
         'E:/Computer Vision/Assignment 1/img5.jpeg',
         ]
for path in paths:

    obj = GradientAnalyzer(path)

    fy = obj.Fy_Image()

    fx = obj.Fx_Image()

    magnitude = obj.gradient_direction()

    directions = obj.gradient_direction()

    print(f"\nFx:{fx}/n Fy\n:{fy}\n Magnitude:\n{magnitude}\n directions:\n{directions}\n")

    cv2.imshow('Fx', magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

