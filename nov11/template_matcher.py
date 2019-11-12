import cv2
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import blob_dog, blob_log, blob_doh

image = cv2.imread('./test_pic.png', cv2.IMREAD_GRAYSCALE)
ih = image.shape[0]
iw = image.shape[1]
template = image[1083:1154, 1650:1739]
th = template.shape[0]
tw = template.shape[1]
heat_map = np.zeros((ih, iw))
hh = heat_map.shape[0]
hw = heat_map.shape[1]

ret, mask = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY_INV)

print('Template: Height ' + str(template.shape[0]) + ', Width ' + str(template.shape[1]))
print('Image: Height ' + str(image.shape[0]) + ', Width ' + str(image.shape[1]))
print('Heat Map: Height ' + str(heat_map.shape[0]) + ', Width ' + str(heat_map.shape[1]))

# for x in range(hw):
#  for y in range(hh):
#    i = image[y:y+th, x:x+tw]
#    sis = np.sum(np.multiply(i,i))
#    heat_map[y,x] = np.sum(np.multiply(i,template))/(np.sqrt(sts*sis))
# cv2.imshow('i', product)
# cv2.waitKey(10)

result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

heat_map[int(th / 2):int(th / 2) + result.shape[0], int(tw / 2):int(tw / 2) + result.shape[1]] = result

masked_hm = np.multiply(mask / 255, heat_map)
# hist, bin_edges = np.histogram(masked_hm,bins=50)
# plt.hist(masked_hm, bins='auto'); plt.show()
# start_point = (900,515)
# cv2.circle(masked_hm,start_point,5,255)
# hist = cv2.calcHist([masked_hm],[0],None,[256],[0,256])
normed_masked = ((masked_hm + 1) * (255.0 / 2.0)).astype('uint8')
# thresh_val, mask_2= cv2.threshold(normed_masked, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh_val, mask_2 = cv2.threshold(normed_masked, 190, 255, cv2.THRESH_BINARY)
final = np.multiply(mask_2 / 255, masked_hm)

blobs_log = blob_log(final, max_sigma=30, num_sigma=10, threshold=.1)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = blob_dog(final, max_sigma=30, threshold=.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(final, max_sigma=30, threshold=.01)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(final)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()
# cv2.imshow('i',np.multiply(mask_2/255,masked_hm))
# cv2.imshow('i', mask)
# cv2.waitKey(0)
