# This is a sample Python script.
import os

import imageio
import skimage
import scipy
from scipy.io import wavfile
from skimage.color import rgb2gray


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
    if representation == 1:
        img = skimage.color.rgb2gray(img)
    img_arr = np.array(img).astype(np.float64)
    return (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())  # normlized the martix


"----------------PART 1-----------------"


def DFT(signal):
    """
    do the discrete dft on signal
    :param signal: one dim array - signala
    :return: the signal after dft
    """
    if signal.size == 0:
        return np.array(None)
    n = len(signal)
    arr = np.arange(n)
    x, y = np.meshgrid(arr, arr)  # create the matrix
    e = np.exp((-2j * np.pi) / n)
    dft = np.dot(np.power(e, x * y), signal)  # dft can be done by matrix mult
    dft = dft.reshape(signal.shape)
    return dft.astype(np.complex128)


def IDFT(fourier_signal):
    """
    do the inverse dft on the fourier
    signal to get back the signal
    :param fourier_signal:
    :return:
    """
    if signal == []:
        return []
    n = len(fourier_signal)
    arr = np.arange(n)
    x, y = np.meshgrid(arr, arr)
    e = np.exp(np.pi * 2j * x * y / n).astype('complex128')
    idft = fourier_signal.reshape(n, ).dot(e) / n
    idft = idft.reshape(fourier_signal.shape)
    return idft.astype(np.complex128)


def DFT2(image):
    '''
    do the dft on an image
    :param image: image - matrix with n*m pixels
    :return: the image after dft
    '''
    n = image.shape[0]  # rows
    m = image.shape[1]  # columns
    dft_img = np.zeros(image.shape).astype(np.complex128)
    for i in range(n):  # do dft on the rows
        dft_img[i, :] = DFT(image[i, :])
    for j in range(m):  # do dft on the cols
        dft_img[:, j] = DFT(dft_img[:, j])
    return dft_img.reshape(image.shape)


def IDFT2(fourier_image):
    '''
    do the dft on an image
    :param image: image - matrix with n*m pixels
    :return: the image after dft
    '''
    n = fourier_image.shape[0]  # rows
    m = fourier_image.shape[1]  # columns
    idft_img = np.zeros(fourier_image.shape).astype(np.complex128)
    for i in range(n):  # do idft on the rows
        idft_img[i, :] = IDFT(fourier_image[i, :])
    for j in range(m):  # do idft on the cols
        idft_img[:, j] = IDFT(idft_img[:, j])
    return idft_img.reshape(fourier_image.shape)


"----------------PART 2-----------------"


def change_rate(filename, ratio):
    '''
    changes the duration of an audio file by keeping the same samples, but changing the
     sample rate written in the file header
    :param filename: string representing the path to a WAV file
    :param ratio:  positive float64 representing the duration change.
    :return:
    '''
    rate, data = wavfile.read(filename)
    new_ratio = int(rate * ratio)
    wavfile.write("change_rate.wav", new_ratio, data)


def resize_fast(data, ratio):
    '''helper function for the resize  - :
     first case - if we need 2 speed up the data
    '''
    shrink = len(data) - int(len(data) / ratio)
    # how mach we need too shrink the data
    dft = np.fft.fftshift(DFT(data))
    dft = dft[int(shrink // 2): len(dft) - int((shrink // 2) + (shrink % 2))]  # cut
    # the data out of the len after speed
    return IDFT(dft)


def resize_slow(data, ratio):
    '''helper function for the resize  - :
     In case of slowing down, we add the needed amount of zeros at the high Fourier frequencies.
    '''
    new_lan = len(data)
    add = (1 / ratio) - 1
    new_lan = np.floor(new_lan * add)
    dft = np.pad(DFT(data), (int(new_lan // 2), int(new_lan // 2) + int(new_lan % 2)),mode='constant') # here the change
    # pad the data with zeros
    return IDFT(dft)


def resize(data, ratio):
    '''
     resize the date using the provided ratio
    :param data: data is a 1D ndarray of dtype float64 or complex128
    :param ratio:positive float64 representing the duration change.
    :return: resized data
    '''
    if data.size == 0 or ratio == 1:
        return data
    if ratio > 1:
        return resize_fast(data, ratio)
    if ratio < 1:
        return resize_slow(data, ratio)


def change_samples(filename, ratio):
    '''
    changes the duration of an audio file by reducing the number of samples
    using Fourier
    :param filename: a string representing the path to a WAV file
    :param ratio:and ratio is a positive float64 representing the duration change
    :return:
    '''
    rate, data = wavfile.read(filename)
    data = np.abs(resize(data, ratio))
    wavfile.write("change_samples.wav", rate, data)
    return data.astype(np.float64)


def resize_spectrogram(data, ratio):
    '''
    change the rate by changing the spectogram
    :param data:  1D ndarray of dtype float64 representing the original sample points
    :param ratio: positive float64 representing the rate change of the WAV file
    :return:
    '''
    spec = stft(data)
    new_spec = np.zeros(((spec.shape[0]), int(spec.shape[1] / ratio)), dtype='complex128')
    for i in range(spec.shape[0]):  # resize every row with the given ratio
        new_spec[i, :] = resize(spec[i, :], ratio)
    return istft(new_spec)

def resize_vocoder(data, ratio):
    '''
    scaling the spectrogram as done before, but includes the correction of
    the phases of each frequency according to the shift of each window
    :param data:
    :param ratio:
    :return:
    '''
    return istft(phase_vocoder(stft(data), ratio))


"----------------PART 3-----------------"


def conv_der(im):
    """
    computes the magnitude of image derivatives with convulsion
    :param im: grayscale images of type float64
    :return: output is the magnitude of the derivative of the image
    """

    x = [[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]]  # use the derivatives matrix convolution
    y = [[0, 0.5, 0], [0, 0, 0], [0, -0.5, 0]]
    x_dir = scipy.signal.convolve2d(im, x,mode='same')
    y_dir = scipy.signal.convolve2d(im, y,mode='same')
    magnitude = np.sqrt(np.abs(x_dir) ** 2 + np.abs(y_dir) ** 2)
    return magnitude.astype(np.float64)


def fourier_der(im):
    '''
    Write a function that computes the magnitude of the image derivatives
    using Fourier transform
    :param im: grayscale image
    :return: the magnitude
    '''

    n = np.shape(im)[0]
    m = np.shape(im)[1]
    dftim = DFT2(im)
    dftim = np.fft.fftshift(dftim)
    u = ((2j * np.pi) / n) * np.arange(-n // 2, n // 2)
    v = ( (2j * np.pi) / m) *np.arange(-m // 2, m // 2)
    dx =  IDFT2(np.fft.ifftshift(np.multiply(u,dftim.T).T))
    dy = IDFT2(np.fft.ifftshift(np.multiply(v,dftim)))
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude.astype(np.float64)




# Press the green button in the gutter to run the script.

import numpy as np
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec



