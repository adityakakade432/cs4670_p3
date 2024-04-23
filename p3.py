import numpy as np
from PIL import Image

from scipy import ndimage, signal


############### ---------- Basic Image Processing ------ ##############

### TODO 1: Read an Image and convert it into a floating point array with values between 0 and 1. You can assume a color image
def imread(filename):
    image = Image.open(filename)
    image_array = np.array(image)
    image_array = image_array / 255
    print(image_array)
    return image_array


### TODO 2: Create a gaussian filter of size k x k and with standard deviation sigma
def gaussian_filter(k, sigma):
    truncate = (((k - 1)/2) - 0.5)/sigma
    init_array = np.zeros((k,k))
    index = int(k/2)
    init_array[index, index] = 1
    return ndimage.gaussian_filter(init_array, sigma = sigma, truncate = truncate)

### TODO 3: Compute the image gradient. 
### First convert the image to grayscale by using the formula:
### Intensity = Y = 0.2125 R + 0.7154 G + 0.0721 B
### Then convolve with a 5x5 Gaussian with standard deviation 1 to smooth out noise. (use scipy.signal.convolve)
### Convolve with [0.5, 0, -0.5] to get the X derivative on each channel and convolve with [[0.5],[0],[-0.5]] to get the Y derivative on each channel. (use scipy.signal.convolve) 
### Return the gradient magnitude and the gradient orientation (use arctan2)
def gradient(img):
    grayscale_image = img[:,:,0]*0.2125 + img[:,:,1]*0.7154 + img[:,:,2]*0.0721
    print(grayscale_image)
    conv_1 = signal.convolve(grayscale_image, gaussian_filter(5, 1))
    x_deriv = signal.convolve(conv_1, [[0.5, 0, -0.5]], mode = "same")
    y_deriv = signal.convolve(conv_1, [[0.5],[0],[-0.5]], mode ="same")

    orientationImage = np.arctan2(x_deriv, y_deriv)

    return np.sqrt(x_deriv**2 + y_deriv**2), orientationImage




##########----------------Line detection----------------

### TODO 4: Write a function to check the distance of a set of pixels from a line parametrized by theta and c. The equation of the line is:
### x cos(theta) + y sin(theta) + c = 0
### The input x and y are arrays representing the x and y coordinates of each pixel
### Return a boolean array that indicates True for pixels whose distance is less than the threshold
def check_distance_from_line(x, y, theta, c, thresh):
    pass


### TODO 5: Write a function to draw a set of lines on the image. The `lines` input is a list of (theta, c) pairs. Each line must appear as red on the final image
### where every pixel which is less than thresh units away from the line should be colored red
def draw_lines(img, lines, thresh):
    pass
 

### TODO 6: Do Hough voting. You get as input the gradient magnitude and the gradient orientation, as well as a set of possible theta values and a set of possible c
### values. If there are T entries in thetas and C entries in cs, the output should be a T x C array. Each pixel in the image should vote for (theta, c) if:
### (a) Its gradient magnitude is greater than thresh1
### (b) Its distance from the (theta, c) line is less than thresh2, and
### (c) The difference between theta and the pixel's gradient orientation is less than thresh3
def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
    pass
    

### TODO 7: Find local maxima in the array of votes. A (theta, c) pair counts as a local maxima if (a) its votes are greater than thresh, and 
### (b) its value is the maximum in a (nbhd x nbhd) neighborhood in the votes array.
### Return a list of (theta, c) pairs
def localmax(votes, thetas, cs, thresh,nbhd):
    pass


  
# Final product: Identify lines using the Hough transform    
def do_hough_lines(filename):

    # Read image in
    img = imread(filename)

    # Compute gradient
    gradmag, gradori = gradient(img)

    # Possible theta and c values
    thetas = np.arange(-np.pi-np.pi/40, np.pi+np.pi/40, np.pi/40)
    imgdiagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    cs = np.arange(-imgdiagonal, imgdiagonal, 0.5)

    # Perform Hough voting
    votes = hough_voting(gradmag, gradori,thetas, cs, 0.2, 0.5, np.pi/40)

    # Identify local maxima to get lines
    lines = localmax(votes, thetas, cs, 20, 11)

    # Visualize: draw lines on image
    result_img = draw_lines(img, lines, 0.5)

    # Return visualization and lines
    return result_img, lines
   
    
