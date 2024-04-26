import numpy as np
from PIL import Image

from scipy import ndimage, signal


############### ---------- Basic Image Processing ------ ##############

### TODO 1: Read an Image and convert it into a floating point array with values between 0 and 1. You can assume a color image
def imread(filename):
    image = Image.open(filename)
    image_array = np.array(image)
    image_array = image_array / 255
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
    conv_1 = signal.convolve(grayscale_image, gaussian_filter(5, 1), mode = "same")
    x_deriv = signal.convolve(conv_1, [[0.5, 0, -0.5]], mode = "same")
    y_deriv = signal.convolve(conv_1, [[0.5],[0],[-0.5]], mode = "same")

    orientationImage = np.arctan2(y_deriv, x_deriv)

    return np.sqrt(np.square(x_deriv) + np.square(y_deriv)), orientationImage




##########----------------Line detection----------------

### TODO 4: Write a function to check the distance of a set of pixels from a line parametrized by theta and c. The equation of the line is:
### x cos(theta) + y sin(theta) + c = 0
### The input x and y are arrays representing the x and y coordinates of each pixel
### Return a boolean array that indicates True for pixels whose distance is less than the threshold
def check_distance_from_line(x, y, theta, c, thresh):
    d = np.zeros(len(x))
    d = np.abs(x * np.cos(theta) + y * np.sin(theta) + c)
    to_return = np.full(len(x), False)
    for i in range(len(x)):
        if d[i] < thresh:
            to_return[i] = True
        else:
            to_return[i] = False
    return to_return


### TODO 5: Write a function to draw a set of lines on the image. The `lines` input is a list of (theta, c) pairs. 
### Each line must appear as red on the final image
### where every pixel which is less than thresh units away from the line should be colored red
def draw_lines(img, lines, thresh):
    img_copy = np.copy(img)
    x_len, y_len, _ = img.shape
    xs = list(range(x_len))
    ys = np.arange(y_len)

    xs = xs * y_len
    ys = np.repeat(ys, x_len)
    xs = np.array(xs)

    pixels = zip(xs, ys)
    pixels = list(pixels)

    for (theta, c) in lines:
        pixel_val = check_distance_from_line(xs, ys, theta, c, thresh)
        pixel_val = np.reshape(pixel_val, (x_len, y_len))

        for x in range(x_len):
            for y in range(y_len):
                if pixel_val[x, y]:
                    img_copy[x, y, 0] = 1
                    img_copy[x, y, 1] = 0
                    img_copy[x, y, 2] = 0
    return img_copy

 

### TODO 6: Do Hough voting. You get as input the gradient magnitude and the 
### gradient orientation, as well as a set of possible theta values and a set of possible c
### values. If there are T entries in thetas and C entries in cs, the output should be a T x C array. 
### Each pixel in the image should vote for (theta, c) if:
### (a) Its gradient magnitude is greater than thresh1
### (b) Its distance from the (theta, c) line is less than thresh2, and
### (c) The difference between theta and the pixel's gradient orientation is less than thresh3
def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
    height, width = gradmag.shape

    xs, ys = np.meshgrid(range(height), range(width), indexing='ij')
    print(ys)

    gradmag_flattened = gradmag.ravel()
    gradori_flattened = gradori.ravel()

    votes = np.zeros((len(thetas), len(cs)))

    valid_pixels = gradmag_flattened > thresh1
    
    filtered_xs = xs.ravel()[valid_pixels]
    filtered_ys = ys.ravel()[valid_pixels]
    filtered_ori = gradori_flattened[valid_pixels]

    for t, theta in enumerate(thetas):
        for c, c_val in enumerate(cs):
            
            d = np.abs(filtered_ys * np.cos(theta) + filtered_xs * np.sin(theta) + c_val)
            diff = np.abs(filtered_ori - theta)
            votes[t, c] = np.sum((d < thresh2) & (diff < thresh3))

    return votes

    

### TODO 7: Find local maxima in the array of votes. A (theta, c) pair counts as a local maxima if (a) its votes are greater than thresh, and 
### (b) its value is the maximum in a (nbhd x nbhd) neighborhood in the votes array.
### Return a list of (theta, c) pairs
def localmax(votes, thetas, cs, thresh,nbhd):
    theta_len, cs_len = votes.shape
    print(votes.shape)
    print(nbhd)
    max_filter = ndimage.filters.maximum_filter(votes, size=(nbhd, nbhd), mode='constant', cval=0)
    to_return = set()
    for theta_index in range(theta_len):
        for c_index in range(cs_len):
            val = max_filter[theta_index, c_index]
            if val > thresh:
                to_return.add((thetas[theta_index], cs[c_index]))
    print(len(to_return))
    return list(to_return)
    


  
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
   
    
