import numpy as np
import imageio
import skimage
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os.path

''' MAGIC NUMBERS'''
BLACK_WHITE = 1
COLOR = 2
HISTOGRAM_SIZE = 256
MAX_GRAY = 255
YIQ_CONVERT = np.array([[0.299, 0.587, 0.114],
                        [0.59590059, -0.27455667, -0.32134392],
                        [0.21153661, -0.52273617, 0.31119955]])  # the convert array to yiq


def read_image(filename, representation):
    '''
    read the image from the user
    :param filename: the file of the user
    :param representation: if  black_White or color
    :return:
    '''
    if not os.path.isfile(filename):  ## if the file don't exist
        return None
    img = imageio.imread(filename)
    if representation == BLACK_WHITE:
        img = skimage.color.rgb2gray(img)
    img_arr = np.array(img).astype(np.float64)
    return (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())  # normlized the martix


def imdisplay(filename, representation):
    '''
    display the image
    :param filename:
    :param representation:
    :return:
    '''
    if not os.path.isfile(filename):## if the file don't exist
        return None
    img = read_image(filename, representation) # use the read image function

    if representation == BLACK_WHITE:
        plt.imshow(img, cmap=plt.cm.gray)
    if representation == COLOR:
        plt.imshow(img)
    plt.show()


def rgb2yiq(imRGB):
    '''
    transform from rgb 2 yiq picture
    :param imRGB: the rgb picture
    :return: new yiq picture
    '''
    OrigShape = imRGB.shape
    imRGB = imRGB.reshape(-1, 3)
    return np.dot(imRGB, YIQ_CONVERT.transpose()).reshape(OrigShape) ## we do martix multiplication with the yiq matrix


def yiq2rgb(imYIQ):
    '''
    transform from yiq 2 rgb picture
    :param imRGB: the yiq picture
    :return: new rgb picture
    '''
    OrigShape = imYIQ.shape
    imYIQ = imYIQ.reshape(-1, 3)
    return np.dot(imYIQ, np.linalg.inv(YIQ_CONVERT).transpose()).reshape(OrigShape)## we do martix multiplication
                                                                                    # with the Invertible yiq matrix

def equalize(im_orig):
    """
    equalize the original picture
    :param im_orig:  the picture
    :return: new equalize picure , the histogram of the original picture and a the new histogram
    """
    his, bins = np.histogram(im_orig.flatten(), bins=HISTOGRAM_SIZE, range=[0, HISTOGRAM_SIZE])# take histogram from the picture
    histsum = np.cumsum(his) # take the cumsum matrix
    pix_num = his.sum()
    hsm_cum_nrm = np.ma.masked_equal(histsum / pix_num, 0) # do a normalized on the matrix
    hsm_cum_nrm = hsm_cum_nrm * hsm_cum_nrm.max()
    if hsm_cum_nrm.max() != MAX_GRAY or hsm_cum_nrm.min() != 0:
        cm = hsm_cum_nrm.nonzero()[0][0] # check were the first zero
        hsm_cum_nrm = (MAX_GRAY * (hsm_cum_nrm - hsm_cum_nrm[cm]
                                   / hsm_cum_nrm.max() - hsm_cum_nrm[cm]))
        hsm_cum_nrm = np.rint(hsm_cum_nrm)
    img_eq = hsm_cum_nrm[np.uint8(im_orig )].astype(np.float64)

    final_hist, bins_eq = np.histogram(img_eq.flatten(), bins=HISTOGRAM_SIZE,
                                       range=[0, HISTOGRAM_SIZE])
    return [img_eq/MAX_GRAY, his, final_hist]


def histogram_equalize(im_orig):
    '''
    we do the equalize process on the im_orig we got
    :param im_orig: the original picture
    :return: new equalize picture , the original histogram and the new histogram
    '''
    if im_orig.ndim == 2: ## if the picture is Black and white
        return equalize(im_orig * MAX_GRAY)
    else: # if the picture is color
        img = rgb2yiq(im_orig)
        y = img[:, :, 0]
        i = img[:, :, 1]
        q = img[:, :, 2]
        im_y, hist, hist_eq = equalize(y * MAX_GRAY) # we do the process only on the y value
        img_eq = np.dstack((y, i, q))
        img_eq = yiq2rgb(img_eq)
        return img_eq, hist, hist_eq


def init_quantize(n_quant,im_orig):
    """
    initialized the q and the z list
    :param n_quant: number of sections
    :param im_orig: the original picure
    :param hist: the histogram of the picture
    :return: new z and q list
    """
    # take the first borders of the picture with the linspace and interp function of numpy
    # to n_quant equal (by number of pixels) parts.
    nsort = np.sort(im_orig.flatten())
    nrange = np.arange(im_orig.size)
    parts = np.linspace(0, im_orig.size, n_quant + 1)
    z = np.interp(parts,nrange ,nsort ).astype(int)
    z[0] =  0

    return z




def quantize_main(im_orig, n_quant, n_iter):
    '''
    quantize the original picture
    :param im_orig: the original picture
    :param n_quant: number of q we want in the new picture
    :param n_iter: number of max iterations
    :return: the new quatize picture , an array of errors
    '''
    his, bin = np.histogram(im_orig.flatten(), bins=HISTOGRAM_SIZE, range=[0, HISTOGRAM_SIZE])
    his_nrm = his / his.sum()
    z = init_quantize(n_quant, im_orig)
    error = []
    iter = 0
    prev_error = 0
    while iter < n_iter:
        q = calculate_q(his_nrm, z, n_quant)
        curr_error = calculate_error(z, q, his_nrm, n_quant)
        z = calculate_z(q)
        if prev_error == curr_error:
                break
        prev_error = curr_error
        error.append(curr_error)
        iter += 1
    z[0] = 0
    im_q = im_orig
    for i in range(n_quant):
        index = np.logical_and(z[i]<=im_orig,z[i+1]>=im_orig)
        im_q[index] = q[i]
    im_q = im_q/MAX_GRAY
    return im_q, error


def calculate_q(hist, z, n_quant):
    '''
    calculate the new q
    :param hist: the histogram of the picture
    :param z: the border of the picture
    :param n_quant:  the number of quant we need
    :return: new q
    '''
    q = []
    for i in range(n_quant):
        denominator = np.sum(hist[z[i] : z[i + 1] ])
        if denominator != 0 :
            qi = np.sum(np.dot(np.arange(z[i], z[i + 1]) ,hist[z[i] : z[i + 1] ]))/\
                 np.sum(hist[z[i] : z[i + 1] ])
        else:
            qi = 0
        q.append(qi)
    return  q

def calculate_z( q):
    ''' calculate the new borders
    :param q: the q list
    :return:  new z list
    '''
    z =[]
    for i in range(1, len(q)):
        zi = (q[i-1] + q[i]) / 2
        z.append(int(zi))
    z.insert(0, 0)
    z.append(MAX_GRAY)
    return z

def calculate_error(z,q,hist,n_quant):
    ''' calculate the new error
    :param z: z list
    :param q: q list
    :param hist: histogram
    :param n_quant: number of quant
    :return:  new error
    '''
    error = 0
    for g in range(n_quant):
        curr_z =  np.arange(z[g]+1, z[g+1]+1).astype(int)
        dist = np.square(q[g] - curr_z)
        e = np.dot(dist,hist[z[g]+1:z[g+1]+1])
        error +=e
    return error



def quantize(im_orig, n_quant, n_iter):
    '''
    quantize the original picture
    :param im_orig: the original picture
    :param n_quant: number of q we want in the new picture
    :param n_iter: number of max iterations
    :return: the new quatize picture , an array of errors
    '''
    if im_orig.ndim == 2: # if the picture is black and white
        return quantize_main(im_orig * MAX_GRAY, n_quant, n_iter)
    else: # color
        img = rgb2yiq(im_orig)
        y = img[:, :, 0]
        i = img[:, :, 1]
        q = img[:, :, 2]
        im_y, error = quantize_main(y * MAX_GRAY, n_quant, n_iter)
        img_quant = np.dstack((im_y, i, q))
        img_quant = yiq2rgb(img_quant)
        return img_quant, error



def quantize_rgb(im_orig, n_quant):
    """
    the bonus task , do quantize on a rgb picture with  other librarys
    , base mainly on the kmeans2 function
    in the scipy library
    :param im_orig: the rgb picture
    :param n_quant: number of quant we want
    :return: the picture after quantize
    """
    import scipy.cluster.vq
    im_new_shape = im_orig.reshape(-1, 3)
    centr, label = scipy.cluster.vq.kmeans2(im_new_shape, n_quant)
    #use the kmeans2 method 2 do the rgb quantize
    rgb_quant = centr[label]
    im_quant = rgb_quant.reshape(im_orig.shape[0], im_orig.shape[1], 3)
    return im_quant



