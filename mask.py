#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

# built-in modules
import logging
# Standard modules
import cv2
# Custom modules
import focus
import scripts

logger = logging.getLogger('main')


def main():
    img_path = input("Please Enter Image Path: ")
    img = cv2.imread(img_path)
    msk, _, _ = focus.blur_mask(img)
    scripts.display('img', img)
    scripts.display('msk', msk)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()