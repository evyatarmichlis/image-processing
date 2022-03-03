import os

import imageio
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import convolve
from skimage.color import rgb2gray

"--------------------------part1--------------------"
def read_image(filename, representation):
    """
    Reads an image file in a given representation and returns it.
    """
    im = imageio.imread(filename)
    if representation == 2:
        return im/255
    else:
        im = rgb2gray(im)
        return im/255

def blur_filter(filter_size):
    """
    create a blur filter with the filter size
    :return:  blur filter
    """
    filter = np.array([1])
    convol = np.array([1, 1])
    for i in range(filter_size - 1):
        filter = np.convolve(filter, convol)
    filter = filter.reshape((1, filter_size))
    return filter

def _convol(curr,filter):
    curr = convolve(curr, filter)
    curr = convolve(curr, filter.T)
    return curr

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    construct a Gaussian pyramid
    :param im:  a grayscale image
    :param max_levels:  the maximal number of levels in the resulting pyramid.
    :param filter_size:  the size of the Gaussian filter
    :return: Gaussian pyramid
    """
    curr = im
    pyramid = []
    pyramid.append(im)
    reduce_normal = np.power(2, filter_size - 1)
    filter = blur_filter(filter_size)/reduce_normal
    for i in range(max_levels-1):
        curr = _convol(curr,filter)
        sub_sampled = curr[0::2, 0::2]
        pyramid.append(sub_sampled)
        curr = sub_sampled
        if curr.shape == (16, 16):
            break
    return pyramid , filter
def build_laplacian_pyramid(im, max_levels, filter_size):
        """
        construct a laplacian pyramid
        :param im:  a grayscale image
        :param max_levels:  the maximal number of levels in the resulting pyramid.
        :param filter_size:  the size of the Gaussian filter
        :return: laplacian pyramid
        """
        pyramid_g,filter = build_gaussian_pyramid(im, max_levels, filter_size)
        lapyr = []

        for i in range(min(max_levels-1,len(pyramid_g)-1)):
            extend_curr = np.zeros((pyramid_g[i].shape[0], pyramid_g[i].shape[1]))
            extend_curr[0::2, 0::2] = pyramid_g[i+1]
            extend_curr = _convol(extend_curr, filter*2)
            l = pyramid_g[i] - extend_curr
            lapyr.append(l)
            # if extend_curr.shape == (16, 16):
            #     break
        lapyr.append(pyramid_g[-1])
        return lapyr, filter

def extend_laplacian(shape,li,filter_vec):
    """
    helper function for the laplacian_to_image , extend the laplacian to the size of original image
    :param shape: the shpae we want the laplacian 2 be
    :param li: the current laplacioan
    :param filter_vec: the filter vector
    :return:
    """
    extend_curr = li
    while extend_curr.shape!= shape:
        extend_curr = np.zeros((extend_curr.shape[0]*2, extend_curr.shape[1]*2),dtype=extend_curr.dtype)
        extend_curr[0::2, 0::2] = li
        extend_curr = _convol(extend_curr, filter_vec)
        li = extend_curr
    return extend_curr


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    reconstruction of an image from its Laplacian Pyramid.
    :param lpyr:Laplacian Pyramid
    :param filter_vec: the vector we used to create the pyramid
    :param coeff: a python list. The list length is the same as the number of levels in the pyramid lpyr
    :return: the origninal picture
    """
    image = lpyr[0]
    lpyr_extend =[]
    for i in range(len(lpyr)): # extend all the laplicans
        lpyr_extend.append(extend_laplacian(image.shape,lpyr[i],filter_vec*2))
    for i in range(len(lpyr)): # create the original picture
        image = image + (lpyr_extend[i]*coeff[i])
    return image


def stretching(img):
    "stretch the values of the image"
    return (img - np.min(img)/np.max(img)-np.min(img))


def render_pyramid (pyr , levels):
    """
    Create a single black image in which the pyramid levels of the
    given pyramid pyr are stacked horizontally
    :param pyr: either a Gaussian or Laplacian pyramid
    :param levels: number of image stack together
    """
    maxloop = min(len(pyr),levels)
    pyramid_img = stretching(pyr[0])
    for i in range(1,maxloop):# loop over the pyr and add the pics together
        stretch_img = stretching(pyr[i])
        pad_size =  pyr[0].shape[0] - pyr[i].shape[0]
        extend_img = np.pad(stretch_img,((0,pad_size), (0, 0)), mode="constant")
        #pad the images with zeros on the sides
        pyramid_img = np.concatenate((pyramid_img, extend_img), axis=1)
    return pyramid_img

def relpath(filename):

    return os.path.join(os.path.dirname(__file__), filename)

def display_pyramid(pyr, levels):
    """
    display the render pyramid
    """
    im = render_pyramid(pyr,levels)
    plt.imshow(im, cmap='gray')
    plt.show()

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    blend 2 picture together with the provide mask
    :param max_levels: number of levels in the pyrmaid we will create
    :param filter_size_im: size of the filter
    :param filter_size_mask: size of the filter
    :return: new blend image
    """
    l1 , vec  = np.array(build_laplacian_pyramid(im1,max_levels,filter_size_im))
    l2 = np.array(build_laplacian_pyramid(im2,max_levels,filter_size_im)[0])
    gm = np.array(build_gaussian_pyramid(mask.astype(np.float64),max_levels,filter_size_mask)[0])
    lout = np.multiply(gm,l1) + np.multiply(1-gm,l2)
    return laplacian_to_image(lout, vec ,np.ones(max_levels))


def blend(im1,im2,mask,max_levels, filter_size_im, filter_size_mask):
    "blend 2 RGB picture  together "
    final = [[],[],[]]
    for i in range(3):
        final[i] = (pyramid_blending(im1[:, :, i],im2[:, :, i],mask,max_levels, filter_size_im, filter_size_mask))
    return np.dstack((final[0],final[1],final[2]))


def  blending_example1():
    "first blend example"

    im1 = read_image(relpath("externals/barbie.jpg"), 2)
    im2 = read_image(relpath("externals/terminator.jpg"), 2)
    mask = read_image(relpath("externals/termask2.jpg"), 1).astype(np.bool)
    im_blend= blend(im2,im1,mask.astype(np.float64),9, 9, 5)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.imshow(im_blend )
    plt.show()
    return im1, im2, mask, im_blend


def blending_example2():
    "second blend example"

    im1 = read_image(relpath("externals/simba.jpg"), 2)
    im2 = read_image(relpath("externals/sefertora.jpg"), 2)
    mask = read_image(relpath("externals/mask_sefertora.jpg"), 1).astype(np.bool)
    im_blend= blend(im2,im1,mask.astype(np.float64),9,9,5)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.imshow(im_blend)
    plt.show()
    return im1, im2, mask, im_blend
