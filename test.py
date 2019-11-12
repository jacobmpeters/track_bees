import numpy as np
import cv2
import pickle
from os import path

drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

def resizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    # specify width OR height and rimage is resized with original aspect ratio

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


# mouse callback function

def draw_circle(event, x, y, flags, param):
    global drawing, mask, r
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        mask[int(x / r), int(y / r)] = 1
        oy = int(y / r) * int(r)
        ox = int(x / r) * int(r)
        for i in range(int(r)):
            for j in range(int(r)):
                image[oy + i, ox + j] = 200
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            mask[int(x / r), int(y / r)] = 1
            oy = int(y / r) * int(r)
            ox = int(x / r) * int(r)
            for i in range(int(r)):
                for j in range(int(r)):
                    image[oy + i, ox + j] = 200
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


img = cv2.imread('/Users/Jake/PycharmProjects/trackBees2/test_pic.png')
temp = img[1083:1154, 1650:1739]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
image, r = resizeWithAspectRatio(temp, temp.shape[1] * 10)

draw_new_mask = False
pickle_file_exists = path.exists("mask.pickle")

if not draw_new_mask & pickle_file_exists:
    mask = pickle.load(open("mask.pickle", "rb"))
    print("We've got a pickle!")
    print(mask)
elif draw_new_mask or not pickle_file_exists:
    print("Dang, no pickle. Let's do this.")
    mask = np.zeros_like(temp)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while 1:
        cv2.imshow('image', image)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break
    mask = mask.astype('uint8')
    pickle.dump(mask, open("mask.pickle", "wb"))
    cv2.destroyAllWindows()
else:
    print("there is either no pickle file or something's up")


print(np.multiply(mask,temp))

result = cv2.matchTemplate(img, temp, cv2.TM_SQDIFF_NORMED)
cv2.imshow('CCORR',result)
k = cv2.waitKey(0)

#ccorr_normed
#(0.6675166487693787, 1.000000238418579, (390, 1929), (1650, 1083))

#ccoeff_normed
#(-0.7015367746353149, 1.0, (1417, 643), (1650, 1083))
