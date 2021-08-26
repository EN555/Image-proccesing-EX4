# ps2
import numpy as np
import os
from ex4_utils import *
import cv2


def displayDepthImage(l_img, r_img, disparity_range=(0, 5), method=disparitySSD):
    p_size = 5
    d_ssd = method(l_img, r_img, disparity_range, p_size)
    plt.matshow(d_ssd)
    plt.colorbar()
    plt.show()

def Homogeneous(src: np.ndarray):
    new_col = [[1], [1], [1], [1]]
    Hom_res = np.append(src, new_col, axis=1)
    return Hom_res

def unHomogeneous(dest: np.ndarray):
    new_mat = np.zeros((dest.shape[0], dest.shape[1] - 1))
    for i in range(new_mat.shape[0]):
        for j in range(new_mat.shape[1]):
            new_mat[i, j] = dest[i, j]/dest[i, 2]
    return new_mat

def main():
    ## 1-a
    # Read images
    i = 1
    # L = cv2.imread(os.path.join('input', 'pair%d-L.png' % i), 0) / 255.0
    # R = cv2.imread(os.path.join('input', 'pair%d-R.png' % i), 0) / 255.0
    L = cv2.cvtColor(cv2.imread('pair1-L.png'), cv2.COLOR_BGR2GRAY) / 255.0
    R = cv2.cvtColor(cv2.imread('pair1-R.png'), cv2.COLOR_BGR2GRAY) / 255.0
    # Display depth SSD
    displayDepthImage(L, R, (50, 100), method=disparitySSD)
    # Display depth NC
    # displayDepthImage(L, R, (0, 100), method=disparityNC)

    # src = np.array([[279, 552],
    #                 [372, 559],
    #                 [362, 472],
    #                 [277, 469]])
    # dst = np.array([[24, 566],
    #                 [114, 552],
    #                 [106, 474],
    #                 [19, 481]])
    # h = computeHomography(src, dst)
    # h_src = Homogeneous(src)
    # pred = h.dot(h_src.T).T
    #
    # pred = unHomogeneous(pred)
    # print(np.sqrt(np.square(pred-dst).mean()))


    # dst = cv2.imread(os.path.join('input', 'billBoard.jpg'), 0) / 255.0
    # src = cv2.imread(os.path.join('input', 'car.jpg'), 0) / 255.0
    # src = cv2.imread('car.jpg') / 255.0
    # dst = cv2.imread('billBoard.jpg') / 255.0
    # warpImag(src, dst)


if __name__ == '__main__':
    main()

