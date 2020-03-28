#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

# built-in modules
import logging
import argparse
# Standard modules
import cv2
import numpy as np
import skimage
import skimage.measure
import skimage.segmentation
# Custom modules
import scripts

logger = logging.getLogger('main')


def get_masks(img, n_seg=250):
    """
    Old version mask creator w/ SLIC (Simple Linear Iterative Clustering)
    :param img: image
    :param n_seg: number of segments
    """
    logger.debug('SLIC segmentation initialised')
    segments = skimage.segmentation.slic(img, n_segments=n_seg, compactness=10, sigma=1)
    logger.debug('SLIC segmentation complete')
    logger.debug('contour extraction...')
    masks = [[np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8), None]]
    
    for region in skimage.measure.regionprops(segments):
        masks.append([masks[0][0].copy(), region.bbox])
        x_min, y_min, x_max, y_max = region.bbox
        masks[-1][0][x_min:x_max, y_min:y_max] = skimage.img_as_ubyte(region.convex_image)
    
    logger.debug('contours extracted')
    return masks[1:]


def blur_mask_old(img):
    """
    Old version for masking
    :param img: image
    """
    assert isinstance(img, np.ndarray), 'img_col must be a numpy array'
    assert img.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img.ndim)
    
    blur_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for mask, loc in get_masks(img):
        logger.debug('Checking Mask: {0}'.format(np.unique(mask)))
        logger.debug('SuperPixel Mask Percentage: {0}%'.format(int((100.0 / 255.0) * (np.sum(mask) / mask.size))))
        _, _, blurry = blur_detector(img[loc[0]:loc[2], loc[1]:loc[3]])
        logger.debug('Blurry: {0}'.format(blurry))
        
        if blurry:
            blur_mask = cv2.add(blur_mask, mask)
    
    result = np.sum(blur_mask) / (255.0 * blur_mask.size)
    logger.info('{0}% of input image is blurry'.format(int(100 * result)))
    return blur_mask, result


def morphology(msk):
    """
    Apply erosion and dilation operators
    :param msk: mask
    """
    assert isinstance(msk, np.ndarray), 'msk must be a numpy array'
    assert msk.ndim == 2, 'msk must be a greyscale image'
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    msk = cv2.erode(msk, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel)
    msk[msk < 128] = 0
    msk[msk > 127] = 255
    return msk


def remove_border(msk, width=50):
    """
    Whiten mask borders
    :param msk: mask
    :param width: width to remove
    """
    assert isinstance(msk, np.ndarray), 'msk must be a numpy array'
    assert msk.ndim == 2, 'msk must be a greyscale image'
    
    dh, dw = map(lambda i: i // width, msk.shape)
    h, w = msk.shape
    msk[:dh, :] = 255
    msk[h-dh:, :] = 255
    msk[:, :dw] = 255
    msk[:, w-dw:] = 255
    return msk


def blur_mask(img):
    """
    Get blur mask in image
    :param img: image
    """
    assert isinstance(img, np.ndarray), 'img_col must be a numpy array'
    assert img.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img.ndim)
    
    msk, _, blurry = blur_detector(img)
    logger.debug('inverting img_fft')
    msk = cv2.convertScaleAbs(255 - (255 * msk / np.max(msk)))
    msk[msk < 50] = 0
    msk[msk > 127] = 255
    logger.debug('removing border')
    msk = remove_border(msk)
    logger.debug('applying erosion and dilation operators')
    msk = morphology(msk)
    logger.debug('evaluation complete')
    result = np.sum(msk) / (255.0 * msk.size)
    logger.info('{0}% of input image is blurry'.format(int(100 * result)))
    return msk, result, blurry


def evaluate(img_col, args):
    """
    Evaluate blur within image
    :param img_col: image
    :param args: arguments
    """
    np.seterr(all='ignore')
    assert isinstance(img_col, np.ndarray), 'img_col must be a numpy array'
    assert img_col.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img_col.ndim)
    assert isinstance(args, argparse.Namespace), 'args must be of type argparse.Namespace not {0}'.format(type(args))
    
    img_gry = cv2.cvtColor(img_col, cv2.COLOR_RGB2GRAY)
    rows, cols = img_gry.shape
    crow, ccol = rows // 2, cols // 2
    f = np.fft.fft2(img_gry)
    fshift = np.fft.fftshift(f)
    fshift[crow-75:crow+75, ccol-75:ccol+75] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_fft = np.fft.ifft2(f_ishift)
    img_fft = 20 * np.log(np.abs(img_fft))
    
    if args.display and not args.testing:
        cv2.destroyAllWindows()
        scripts.display('img_fft', img_fft)
        scripts.display('img_col', img_col)
        cv2.waitKey(0)
    
    result = np.mean(img_fft)
    return img_fft, result, result < args.thresh


def blur_detector(img_col, thresh=10, mask=False):
    """
    Detect blurs
    :param img_col: image
    :param thresh: threshold
    :param mask: if mask
    """
    assert isinstance(img_col, np.ndarray), 'img_col must be a numpy array'
    assert img_col.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img_col.ndim)
    
    args = scripts.gen_args()
    args.thresh = thresh
    if mask:
        return blur_mask(img_col)
    else:
        return evaluate(img_col=img_col, args=args)