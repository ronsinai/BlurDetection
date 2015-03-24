#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

# built-in modules
import logging
# Standard modules
import cv2
import numpy
import skimage
import skimage.measure
import skimage.segmentation
# Custom modules
import main
import scripts

logger = logging.getLogger('main')


def get_masks(img, n_seg=250):
    logger.debug('SLIC segmentation initialised')
    segments = skimage.segmentation.slic(img, n_segments=n_seg, compactness=10, sigma=1)
    logger.debug('SLIC segmentation complete')
    logger.debug('contour extraction...')
    masks = [[numpy.zeros((img.shape[0], img.shape[1]), dtype=numpy.uint8), None]]
    bboxs = []
    for region in skimage.measure.regionprops(segments):
        masks.append([masks[0][0].copy(), region.bbox])
        x_min, y_min, x_max, y_max = region.bbox
        bboxs.append(region.bbox)
        masks[-1][0][x_min:x_max, y_min:y_max] = skimage.img_as_ubyte(region.convex_image)
    logger.debug('contours extracted')
    return masks[1:], bboxs


def get_blur_mask(img):
    assert isinstance(img, numpy.ndarray), 'img_col must be a numpy array'
    assert img.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img.ndim)
    blur_mask = numpy.zeros(img.shape[:2], dtype=numpy.uint8)
    for mask, loc in get_masks(img):
        val, blurry = main.blur_detector(img[loc[0]:loc[2], loc[1]:loc[3]])
        if blurry:
            blur_mask = cv2.add(blur_mask, mask)
    result = numpy.sum(blur_mask)/(255.0*blur_mask.size)
    logger.info('{0}% of input image is blurry'.format(int(100*result)))
    return blur_mask, result