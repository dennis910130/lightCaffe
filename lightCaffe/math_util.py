__author__ = 'sichen'
import numpy as np


def im_pad(im, padding_size):
    """This function add padding_size zeros to the border of a image.

    Parameters
    ----------
    im : 3-D ndarray
        cab
    padding_size : int
        size of padding

    Returns
    -------
    padded_image : 3-D ndarray
        The padded version of image

    """

    return np.lib.pad(im, ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
                      'constant', constant_values=0)


def im2col(im, filter_size, stride):
    """This function ...

    Parameters
    ----------
    im : 3-D ndarray
        cab
    filter_size : int
        ...
    stride: int
        ...

    Returns
    -------
    patches : 2-D ndarray
        ...

    """

    out_size = (im.shape[1] - filter_size) / stride + 1
    out = np.empty((out_size * out_size, im.shape[0] * filter_size * filter_size), dtype=im.dtype)
    for i in xrange(out_size):
        for j in xrange(out_size):
            out[i*out_size+j, :] = im[:, i*stride:i*stride+filter_size, j*stride:j*stride+filter_size].ravel()
    return out


def convolve3d(im, filter, stride):
    """This function ...

    Parameters
    ----------
    im : 3-D ndarray
        cab
    filter : 4-D ndarray
        ncab
    stride: int
        ...

    Returns
    -------
    filter_output : 3-D ndarray
        cab

    """
    out_size = (im.shape[1] - filter.shape[2]) / stride + 1
    reshaped_filter = filter.reshape((filter.shape[0], -1))
    reshaped_im = im2col(im, filter.shape[2], stride)
    reshaped_output = np.dot(reshaped_im, reshaped_filter.T)
    return np.rollaxis(reshaped_output.reshape((out_size, out_size, -1)), 2)
