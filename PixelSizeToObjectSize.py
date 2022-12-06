import math, numpy as np

dist = 2.67 # m
pixelSize = 270
focalLength = 422.18

objSize = (dist * pixelSize) / focalLength

print(objSize)