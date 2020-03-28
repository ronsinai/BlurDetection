#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

# Built-in Modules
import os
import argparse
import logging
# Standard Modules
import cv2
import numpy as np
# Custom Modules

logger = logging.getLogger('main')


def get_logger(level=logging.INFO, quite=False, debug=False, to_file=''):
    """
    This function initialises a logger to stdout.
    :param level: logging level (DEBUG / INFO / WARNING / CRITICAL)
    :param quite: if mute logging
    :param debug: if debug mode
    :param to_file: log file path
    :return: logger
    """
    assert level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.CRITICAL]
    logger = logging.getLogger('main')
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    
    if debug:
        level = logging.DEBUG
    logger.setLevel(level=level)
    
    if not quite:
        if to_file:
            fh = logging.FileHandler(to_file)
            fh.setLevel(level=level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        else:
            ch = logging.StreamHandler()
            ch.setLevel(level=level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
    
    return logger


def get_args(default=None, args_string=''):
    """
    :param default: dictionary of default arguments with keys as `dest`
    :param args_string: arguments passed on startup
    :return: command line arguments
    """
    if not default:
        default = {}
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image_paths', type=str, nargs='+', help="Filepath for input images or folder containing images")
    parser.add_argument('-p', '--superpixel', dest='superpixel', action='store_true', help='generate blur estimation for superpixels')
    parser.add_argument('-r', '--thresh', dest='thresh', default=10, type=str, help='threshold for deciding if blurry (between 0 & 1)')
    parser.add_argument('-m', '--mask', dest='mask', action='store_true', help='Conduct SLIC Segmentation to generate focus mask')
    parser.add_argument('-d', '--display', dest='display', action='store_true', help='display image steps')
    parser.add_argument('-e', '--debug', dest='debug', action='store_true', help='set logger to debug')
    parser.add_argument('-q', '--quite', dest='quite', action='store_true', help='silence the logger')
    parser.add_argument('-s', '--save', dest='save', action='store_true', help='save blur masks')
    parser.add_argument('-t', '--testing', dest='testing', action='store_true', help='enable testing method')
    
    if args_string:
        args_string = args_string.split(' ')
        args = parser.parse_args(args_string)
    else:
        args = parser.parse_args()
    
    return args


def gen_args():
    """
    Generate arguments
    """
    return get_args(args_string='USED_GEN_ARGS')


def find_images(path, recursive=True):
    """
    Find all images paths
    :param path: root path
    :param recursive: if recursive search
    """
    if os.path.isdir(path):
        return list(xfind_images(path, recursive=recursive))
    elif os.path.exists(path):
        return [path]
    else:
        raise ValueError('path is not a valid path or directory')


def xfind_images(directory, recursive=False, ignore=True):
    """
    Yield images for blur detection
    :param directory: root directory to search in
    :param recursive: if recursive
    :param ignore: if keep dash-separated image paths
    """
    assert os.path.isdir(directory), 'FileIO - get_images: Directory does not exist'
    assert isinstance(recursive, bool), 'FileIO - get_images: recursive must be a boolean variable'
    assert isinstance(ignore, bool), 'FileIO - get_images: ignore must be a boolean variable'
    
    ext = ['png', 'jpg', 'jpeg']
    for path_a in os.listdir(directory):
        path_a = os.path.join(directory, path_a)
        if os.path.isdir(path_a) and recursive:
            for path_b in xfind_images(path_a):
                yield path_b
        
        check_a = path_a.split('.')[-1] in ext
        check_b = ignore or ('-' not in path_a.split('/')[-1])
        if check_a and check_b:
            yield path_a


def display(title, img, max_size=200000):
    """
    Display image
    :param title: title for image
    :param img: image
    :param max_size: maximum size for image
    """
    assert isinstance(img, np.ndarray), 'img must be a numpy array'
    assert isinstance(title, str), 'title must be a string'
    
    rows, cols = img.shape[:2]
    scale = np.sqrt(min(1.0, float(max_size) / (rows * cols)))
    logger.debug('image is being scaled by a factor of {0}'.format(scale))
    
    shape = (int(scale * cols), int(scale * rows))
    img = cv2.resize(img, shape)
    cv2.imshow(title, img)