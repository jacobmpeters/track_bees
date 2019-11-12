import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt


# video filename
input_video_FID = '/Users/Jake/Dropbox/HandBrake_compressedVids/swarm5_testG_-08162019122501-0000.mp4'
# time where bees are most spread out
time_optimal_spacing = (2 * 60 + 45) * 1000  # (min * 60 + sec) * 1000 --> ms


def rotate_template_image(image, angle_in_deg, borderValue=255):
    # rotate image by specified angle in degrees. border_val specifies the
    # value of border pixels. when you rotate with cv2.warpAffine, the image is
    # padded with 0s, but the border between the image and the pad is sometimes 255..

    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angle_in_deg, 1)

    rad = np.radians(angle_in_deg)
    sin = np.sin(rad)
    cos = np.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    ## delete this
    b_w = 80
    b_h = 80

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR, borderValue=borderValue)
    return outImg


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    # specify width OR height and image is resized with original aspect ratio

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter), r


# get frame of interest
video = cv2.VideoCapture(input_video_FID)
video.set(0, time_optimal_spacing)  # 0 --> CV_CAP_PROP_POS_MSEC
ret, raw_frame = video.read()
raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

scaled_frame, scale_factor = resize_with_aspect_ratio(raw_frame, width=1280)
print('select ROI to crop frame (top left to bottom right):')
fromCenter = False
r = [i / scale_factor for i in cv2.selectROI(scaled_frame, fromCenter)]
# cropped_frame = scaled_frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
frame = raw_frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
fh = frame.shape[0]  # frame height
fw = frame.shape[1]  # frame width

# threshold frame
ret, frame_mask = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY_INV)

# draw template ROI
# Select ROI
print('select ROI for template (from center):')
r = cv2.selectROI(frame)
template = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
th = template.shape[0]
tw = template.shape[1]
heat_map_zeros = np.zeros((fh, fw))
hh = heat_map_zeros.shape[0]
hw = heat_map_zeros.shape[1]

heatmap_list = []  # initialize list for templateMatch output matrix
for angle_in_degrees in np.arange(0, 360, 10):
    rot_template = rotate_template_image(template, angle_in_degrees, borderValue=255)
    rot_template = np.array(rot_template)
    rot_template[rot_template == 0] = 255
    ret, trunc_template = cv2.threshold(rot_template, 127, 255, cv2.THRESH_TRUNC)

    result = cv2.matchTemplate(frame, trunc_template, cv2.TM_CCOEFF_NORMED)
    heat_map = heat_map_zeros  # initialize
    heat_map[int(th / 2):int(th / 2) + result.shape[0], int(tw / 2):int(tw / 2) + result.shape[1]] = result
    masked_heat_map = np.multiply(frame_mask / 255, heat_map)  # apply mask by multiplying it by heat map
    heatmap_list.append(masked_heat_map)

    # visualize
    cv2.imshow('sweep through angles', masked_heat_map)
    #cv2.imshow('trunc template',trunc_template)
    cv2.waitKey(25)

# pool output matrices
width_out, height_out = heat_map.shape[::-1]  # get shape of output matrices from template matching
max_pool = np.empty_like(heat_map)

for i in range(height_out):
    for j in range(width_out):
        max_pool[i, j] = max([hm[i][j] for hm in heatmap_list])

cv2.imshow('max pool img', max_pool)
cv2.waitKey(0)

