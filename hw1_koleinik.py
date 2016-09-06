import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

#Resized image to 512x512
img = cv2.imread('me.jpg')
src=cv2.resize(img,(512,512))

#First split the image into the three color channels r,g,b.
#Online it said it may be useful to use hsv due to its range and intensity property?
b,g,r =cv2.split(src)

#we now have three images to work with b , g , and r
#need to apply Gaussian filters for Sigma 1,2,3
#Simga=1
blur_blue1=cv2.GaussianBlur(b,(15,15),1)
blur_green1=cv2.GaussianBlur(g,(15,15),1)
blur_red1=cv2.GaussianBlur(r,(15,15),1)

#Sigma=2
blur_blue2=cv2.GaussianBlur(b,(15,15),2)
blur_green2=cv2.GaussianBlur(g,(15,15),2)
blur_red2=cv2.GaussianBlur(r,(15,15),2)

#Sigma=3
blur_blue3=cv2.GaussianBlur(b,(15,15),3)
blur_green3=cv2.GaussianBlur(g,(15,15),3)
blur_red3=cv2.GaussianBlur(r,(15,15),3)

#Now we have the different filtered images for different sigma values
#Time to merge the values
merged_sigma1=cv2.merge([blur_blue1,blur_green1,blur_red1])
merged_sigma2=cv2.merge([blur_blue2,blur_green2,blur_red2])
merged_sigma3=cv2.merge([blur_blue3,blur_green3,blur_red3])

#looks good we can see that the it gets blurrier with a greater sigma value
#Three derivatives of the different sigma values
#X derivative
derivative_xoriginal=cv2.Sobel(src,cv2.CV_64F,1,0,ksize=5)  
derivative_x1=cv2.Sobel(merged_sigma1,cv2.CV_64F,1,0,ksize=5)
derivative_x2=cv2.Sobel(merged_sigma2,cv2.CV_64F,1,0,ksize=5)
derivative_x3=cv2.Sobel(merged_sigma3,cv2.CV_64F,1,0,ksize=5)

#Y derivative
derivative_yoriginal=cv2.Sobel(src,cv2.CV_64F,0,1,ksize=5)
derivative_y1=cv2.Sobel(merged_sigma1,cv2.CV_64F,0,1,ksize=5)
derivative_y2=cv2.Sobel(merged_sigma2,cv2.CV_64F,0,1,ksize=5)
derivative_y3=cv2.Sobel(merged_sigma3,cv2.CV_64F,0,1,ksize=5)

#magnitude images
edge_original=cv2.Canny(src,100,100,5)
edge_merge1=cv2.Canny(merged_sigma1,100,100,5)
edge_merge2=cv2.Canny(merged_sigma2,100,100,5)
edge_merge3=cv2.Canny(merged_sigma3,100,100,5)

#write images to filesystem

if(os.path.isdir('output')==False):
    os.mkdir('output')

os.chdir('output')
cv2.imwrite('Resized.jpg',src)
cv2.imwrite('Gaussian_Sigma1.jpg',merged_sigma1)
cv2.imwrite('Gaussian_Sigma2.jpg',merged_sigma2)
cv2.imwrite('Gaussian_Sigma3.jpg',merged_sigma3)
cv2.imwrite('Derivative_X_original.jpg',derivative_xoriginal)
cv2.imwrite('Derivative_X_Sigma1.jpg',derivative_x1)
cv2.imwrite('Derivative_X_Sigma2.jpg',derivative_x2)
cv2.imwrite('Derivative_X_Sigma3.jpg',derivative_x3)
cv2.imwrite('Derivative_Y_original.jpg',derivative_yoriginal)
cv2.imwrite('Derivative_Y_Sigma1.jpg',derivative_y1)
cv2.imwrite('Derivative_Y_Sigma2.jpg',derivative_y2)
cv2.imwrite('Derivative_Y_Sigma3.jpg',derivative_y3)
cv2.imwrite('Edge_orginal.jpg',edge_original)
cv2.imwrite('Edge_merge1.jpg',edge_merge1)
cv2.imwrite('Edge_merge2.jpg',edge_merge2)
cv2.imwrite('Edge_merge3.jpg',edge_merge3)



cv2.waitKey(0);

cv2.destroyAllWindows()
