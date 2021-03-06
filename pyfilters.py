from PIL import (ImageEnhance, Image)
import numpy as np
from cv2 import transform
import math

from typing import Tuple

Color = Tuple[int, int, int]
Size = Tuple[int, int]

class Filter:

    def __init__(self, path, output):
        self.image = Image.open(path)
        self.output = output
        
    def contrast(self,image):        
        return ImageEnhance.Contrast(image)

    def color(self,image):
        return ImageEnhance.Color(image)

    def brightness(self,image):
        return ImageEnhance.Brightness(image)

    def sharpness(self,image):
        return ImageEnhance.Sharpness(image)

    def sepia(self, image):
        width, height = image.size
        img = image
        mode = None

        
        if image.mode != "RGB":
            mode = image.mode
            img = image.convert('RGB')
        pixels = img.load() # create the pixel map
        
        for py in range(height):
            for px in range(width):
                r, g, b = img.getpixel((px, py))

                tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                
                if tr > 255:
                    tr = 255

                if tg > 255:
                    tg = 255

                if tb > 255:
                    tb = 255

                pixels[px, py] = (tr,tg,tb)

        return image.convert(mode)

    def sepia_cv(self, image, amount = 1):
        """
        Optimization on the sepia filter using cv2 
        """
        
        import cv2

        matrix = [[ 0.393 + 0.607 * (1 - amount), 0.769 - 0.769 * (1 - amount), 0.189 - 0.189 * (1 - amount)],
                  [ 0.349 - 0.349 * (1 - amount), 0.686 + 0.314 * (1 - amount), 0.168 - 0.168 * (1 - amount)],
                  [ 0.272 - 0.349 * (1 - amount), 0.534 - 0.534 * (1 - amount), 0.131 + 0.869 * (1 - amount)]                                  
        ]

        # Load the image as an array so cv knows how to work with it
        img = np.array(image)

        # Apply a transformation where we multiply each pixel rgb with the matrix for the sepia
        filt = transform( img, np.matrix(matrix) )
        
        # Check wich entries have a value greather than 255 and set it to 255
        filt[np.where(filt>255)] = 255

        # Create an image from the array 
        return Image.fromarray(filt)
                
    def rgb_to_hsv(self,rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    def hsv_to_rgb(self,hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def hueShift(self,img, amount):
        #https://stackoverflow.com/questions/27041559/rgb-to-hsv-python-change-hue-continuously
        arr = np.array(img)
        hsv = self.rgb_to_hsv(arr)
        hsv[..., 0] = (hsv[..., 0]+(amount/360)) % 1.0
        rgb = self.hsv_to_rgb(hsv)
        return Image.fromarray(rgb, 'RGB')

        
    def filter_aden(self):

        prod = self.contrast(self.image).enhance(0.9)
        prod = self.brightness(prod).enhance(1.2)
        prod = self.color(prod).enhance(0.85)
        prod = self.hueShift(prod, 15)
        self.prod = prod

    def filter_clarendon(self):

        prod = self.contrast(self.image).enhance(1.2)
        prod = self.brightness(prod).enhance(1)
        prod = self.color(prod).enhance(1.20)
        self.prod = prod

    def filter_early_bird(self):
        prod = self.sepia_cv(self.image)
        prod = self.contrast(prod).enhance(0.9)
        prod = self.brightness(prod).enhance(1)
        prod = self.color(prod).enhance(1)
        self.prod = prod

    def filter_gingham(self):
        prod = self.contrast(self.image).enhance(1.0)
        prod = self.brightness(prod).enhance(1.05)
        prod = self.color(prod).enhance(1)
        prod = self.hueShift(prod, 350)

        self.prod = prod

    def filter_hudson(self):

        prod = self.contrast(self.image).enhance(0.9)
        prod = self.brightness(prod).enhance(1.2)
        prod = self.color(prod).enhance(1.1)

        self.prod = prod


    def filter_inkwell(self):

        prod = self.contrast(self.image).enhance(1.1)
        prod = self.brightness(prod).enhance(1.1)
        prod = self.color(prod).enhance(1)
        prod = self.sepia_cv(prod)

        # Convert image to black and white
        prod = prod.convert('L')
        self.prod = prod

    def filter_sepia(self):
        self.prod = self.sepia_cv(self.image)

    def filter_xpro2(self):

        prod = self.contrast(self.image).enhance(1)
        prod = self.brightness(prod).enhance(1)
        prod = self.color(prod).enhance(1)
        prod = self.sepia_cv(prod, amount = 0.41)
        
        # TODO: Apply gradient
        self.prod = prod

    def filter_valencia(self):
        prod = self.contrast(self.image).enhance(1.08)
        prod = self.brightness(prod).enhance(1.08)
        prod = self.color(prod).enhance(1)
        prod = self.sepia_cv(prod, amount = 0.08)

        self.prod = prod
        
    def generate(self):

        try:
            self.prod.save(self.output)
        except:
            self.image.save(self.output)
            
#t = Filter('test.jpg', 'out.jpg')
#t.filter_aden()
#t.filter_clarendon()
#t.filter_early_bird()
#t.filter_gingham()
#t.filter_inkwell()
#t.filter_sepia()
#t.filter_xpro2()
#t.filter_valencia()
#t.generate()
