#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

# built-in modules
import logging
# Standard modules
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Custom modules
import focus
import scripts

logger = logging.getLogger('main')


def main():
    args = scripts.get_args()
    logger = scripts.get_logger(quite=args.quite, debug=args.debug)
    
    x_okay, y_okay = [], []
    x_blur, y_blur = [], []
    
    for path in args.image_paths:
        for img_path in scripts.find_images(path):
            logger.debug('evaluating {0}'.format(img_path))
            img = cv2.imread(img_path)
            assert isinstance(img, np.ndarray)

            if args.testing:
                scripts.display('dialog (blurry: Y?)', img)
                blurry = False
                if cv2.waitKey(0) in map(lambda i: ord(i), ['Y', 'y']):
                    blurry = True
                x_axis = [1, 3, 5, 7, 9]
                
                for x in x_axis:
                    img_mod = cv2.GaussianBlur(img, (x, x), 0)
                    y = focus.evaluate(img_mod, args=args)[0]
                    
                    if blurry:
                        x_blur.append(x)
                        y_blur.append(y)
                    else:
                        x_okay.append(x)
                        y_okay.append(y)
            elif args.mask:
                msk, _, blurry = focus.blur_mask(img)
                img_msk = cv2.bitwise_and(img, img, mask=msk)
                
                if args.display:
                    scripts.display('res', img_msk)
                    scripts.display('msk', msk)
                    scripts.display('img', img)
                    cv2.waitKey(0)
            else:
                img_fft, result, _ = focus.evaluate(img, args=args)
                logger.info('fft average of {0}'.format(result))
                
                if args.display:
                    scripts.display('input', img)
                    scripts.display('img_fft', img_fft)
                    cv2.waitKey(0)

    if args.display and args.testing:
        logger.debug('x_okay: {0}'.format(x_okay))
        logger.debug('y_okay: {0}'.format(y_okay))
        logger.debug('x_blur: {0}'.format(x_blur))
        logger.debug('y_blur: {0}'.format(y_blur))
        plt.scatter(x_okay, y_okay, color='g')
        plt.scatter(x_blur, y_blur, color='r')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()