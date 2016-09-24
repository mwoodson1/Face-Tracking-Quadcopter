import numpy as np
import cv2, time
import matplotlib.pyplot as plt
from classifier import *

#Read in random frame
img_path = 'data/im25_20521600.jpg'
img = cv2.imread(img_path,0)

cv2.imshow('Original',img)
cv2.waitKey(0)

#After background subtraction
img_path = 'data/sil10001.pbm'
img = cv2.imread(img_path,0)
cv2.imshow('Background Subtracted',img)
cv2.waitKey(0)

#Clean up the image after the background subtraction
cleaned = denoiseSilhouette(img)

cv2.imshow('Denoised',cleaned)
cv2.waitKey(0)

#Extract the contour
cx, cy, cnt = findMaxContour(cleaned)
dists = np.zeros(len(cnt))

#Drawing the centroid
drawing_cnt = np.zeros(img.shape)
drawing_cnt[cy,cx] = 255

drawing_cnt[cy+1,cx] = 255
drawing_cnt[cy+1,cx+1] = 255
drawing_cnt[cy+1,cx-1] = 255

drawing_cnt[cy-1,cx] = 255
drawing_cnt[cy-1,cx+1] = 255
drawing_cnt[cy-1,cx-1] = 255

fig, axs = plt.subplots(1,2)
axs = axs.ravel()
plt.ion()
fig.show()
#Find distance betweem center to all contour poiints
for i in xrange(len(cnt)):
    x = cnt[i][0][0]
    y = cnt[i][0][1]
    dists[i] = np.sqrt(((x-cx)**2)+((y-cy)**2))

    drawing_cnt[y,x] = 255
    axs[0].imshow(drawing_cnt,cmap='Greys_r')

    axs[1].plot(dists)
    plt.pause(0.00001)
    axs[0].cla()
    axs[1].cla()
    print "Pixel distance to centroid is: ",dists[i]


#L1 normalization of distances
dists_norm = dists / LA.norm(dists,1)

#Subsample the vector to a constant size
indices = np.linspace(0,len(dists_norm),360,endpoint=False,dtype=np.int32)
sampled_dists = dists_norm[indices]

plt.figure()
plt.plot(sampled_dists)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
