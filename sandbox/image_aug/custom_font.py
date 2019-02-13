import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import time

inverted = False

## Make canvas and set the color
cols = 475
rows = 207
img = np.zeros((rows, cols, 3), np.uint8)

if inverted:
    b, g, r, a = 255, 255, 255, 1
    rectangle_bgr = (0, 0, 0)
else:
    b, g, r, a = 0, 0, 0, 1
    rectangle_bgr = (255, 255, 255)

# set the rectangle background
cv2.rectangle(img, (0, 0), (cols, rows), rectangle_bgr, cv2.FILLED)

## Use simsum.ttc to write Chinese.
fontpath = "./LICENSE_PLATE_USA.ttf"
font = ImageFont.truetype(fontpath, 125)

img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
draw.text((30, 30), "A 12345", font=font, fill=(b, g, r, a))
img = np.array(img_pil)

## Display
cv2.imshow("res", img)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite("res_{}.png".format("inv" if inverted else "orig"), img)
