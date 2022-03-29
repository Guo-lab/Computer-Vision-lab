import cv2 # '4.5.5'
import numpy as np # '1.19.5'

import matplotlib.pyplot as plt # '3.2.2'
from medpy import metric # '0.3.0'


#@ https://juejin.cn/post/6971243846176866312
# Dice Coeffcient,
# Hausdorff distance - HD95
# sensitivity,
# specificity,
# accuracy

#! ...
from medpy.metric import binary




def load_image(filename):
    image = cv2.imread(filename)
    return image
        
#% 1 sensitivity
#   ___T_P____
#    TP + FN
def sensitivity(output, target):
    tp = np.sum((output == 255) & (target == 255))
    tn = np.sum((output == 0) & (target == 0))
    fp = np.sum((output == 255) & (target == 0))
    fn = np.sum((output == 0) & (target == 255))

    sen = tp / (tp + fn)

    return sen

imgOut = load_image('./data/out/out7.png')
imgSrc = load_image('./data/mask/7.png')

print(sensitivity(imgOut, imgSrc))

imgOut = load_image('./data/out/out14.png')
imgSrc = load_image('./data/mask/14.png')

print(sensitivity(imgOut, imgSrc))

imgOut = load_image('./data/out/out16.png')
imgSrc = load_image('./data/mask/16.png')

print(sensitivity(imgOut, imgSrc))









#% 2. specificity
#   ___T_N____
#    TN + FP
def specificity(output, target):
    tp = np.sum((output == 255) & (target == 255))
    tn = np.sum((output == 0) & (target == 0))
    fp = np.sum((output == 255) & (target == 0))
    fn = np.sum((output == 0) & (target == 255))

    spec = tn / (tn + fp)

    return spec

imgOut = load_image('./data/out/out7.png')
imgSrc = load_image('./data/mask/7.png')

print(specificity(imgOut, imgSrc))

imgOut = load_image('./data/out/out14.png')
imgSrc = load_image('./data/mask/14.png')

print(specificity(imgOut, imgSrc))

imgOut = load_image('./data/out/out16.png')
imgSrc = load_image('./data/mask/16.png')

print(specificity(imgOut, imgSrc))












#% 3. accuracy (PA)
#   ___T_P__+___T_N____
#    TP + TN + FP + FN 
def accuracy(output, target):
    tp = np.sum((output == 255) & (target == 255))
    tn = np.sum((output == 0) & (target == 0))
    fp = np.sum((output == 255) & (target == 0))
    fn = np.sum((output == 0) & (target == 255))

    acc = (tn + tp) / (tn + tp + fp + fn)

    return acc

imgOut = load_image('./data/out/out7.png')
imgSrc = load_image('./data/mask/7.png')

print(accuracy(imgOut, imgSrc))

imgOut = load_image('./data/out/out14.png')
imgSrc = load_image('./data/mask/14.png')

print(accuracy(imgOut, imgSrc))

imgOut = load_image('./data/out/out16.png')
imgSrc = load_image('./data/mask/16.png')

print(accuracy(imgOut, imgSrc))











#% 4. dice 
#   ___2_*_T_P_____
#    2TP + FP + FN 
def dice(output, target):
    tp = np.sum((output == 255) & (target == 255))
    tn = np.sum((output == 0) & (target == 0))
    fp = np.sum((output == 255) & (target == 0))
    fn = np.sum((output == 0) & (target == 255))

    dice = 2* tp / (2 * tp + fp + fn)

    return dice

imgOut = load_image('./data/out/out7.png')
imgSrc = load_image('./data/mask/7.png')

print(dice(imgOut, imgSrc))

imgOut = load_image('./data/out/out14.png')
imgSrc = load_image('./data/mask/14.png')

print(dice(imgOut, imgSrc))

imgOut = load_image('./data/out/out16.png')
imgSrc = load_image('./data/mask/16.png')

print(dice(imgOut, imgSrc))
















#% 5. Hausdorff distance - HD95
#@ https://thejns.org/focus/view/journals/neurosurg-focus/51/2/article-pE14.xml

#//hd=binary.hd(Vseg, Vref, voxelspacing=voxelspacing)
#//hd95=binary.hd95(Vseg, Vref, voxelspacing=voxelspacing)
#@ https://blog.csdn.net/u012897374/article/details/112008872
#@ https://blog.csdn.net/hpulittle_804/article/details/118367573

Vseg = load_image('./data/out/out7.png')
Vref = load_image('./data/mask/7.png')

hd=binary.hd(Vseg, Vref, voxelspacing=None)
hd95=binary.hd95(Vseg, Vref, voxelspacing=None)

print(hd)
print(hd95)



Vseg = load_image('./data/out/out14.png')
Vref = load_image('./data/mask/14.png')

hd=binary.hd(Vseg, Vref, voxelspacing=None)
hd95=binary.hd95(Vseg, Vref, voxelspacing=None)

print(hd)
print(hd95)



Vseg = load_image('./data/out/out16.png')
Vref = load_image('./data/mask/16.png')

hd=binary.hd(Vseg, Vref, voxelspacing=None)
hd95=binary.hd95(Vseg, Vref, voxelspacing=None)

print(hd)
print(hd95)

#@ http://cgm.cs.mcgill.ca/~godfried/teaching/cg-projects/98/normand/main.html