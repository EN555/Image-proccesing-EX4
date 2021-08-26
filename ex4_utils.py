import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from PIL import Image
from scipy.ndimage import filters


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)
    return: Disparity map, disp_map.shape = Left.shape
    """

    max_r = disp_range[1]
    disparity_map = np.zeros_like(img_l)
    # move on all the left image
    for row in range(k_size, img_l.shape[0] - k_size):
        for col in range(k_size, img_l.shape[1] - k_size):
            # define the matrix in the left image
            windowL = img_l[row - k_size: row + k_size + 1, col - k_size: col + k_size + 1]
            min_ssd = sys.maxsize   # take big number
            # move on the right image
            for i in range(max(col - max_r, k_size), min(col + max_r, img_l.shape[1] - k_size)):
                # define an matrix in the right image
                windowR = img_r[row - k_size: row + k_size + 1, i - k_size: i + k_size + 1]
                ssd = ((windowL - windowR) ** 2).sum()
                if ssd < min_ssd:
                    min_ssd = ssd
                    disparity_map[row, col] = np.abs(i - col)
    disparity_map *= (255 // max_r)
    return disparity_map


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    kernel = k_size * 2 + 1
    MaxRange = disp_range[1]
    row, col = img_l.shape
    # Save an array of different sum values
    s = np.zeros((row, col))
    s_l = np.zeros((row, col))
    s_r = np.zeros((row, col))

    all_dim = np.zeros((row, col, MaxRange))
    mean_kernel = np.ones((kernel, kernel))
    # normalize the image by mean filter
    norm_left = img_l - cv2.filter2D(img_l, -1, mean_kernel)
    norm_right = img_r - cv2.filter2D(img_r, -1, mean_kernel)

    for curr_range in range(MaxRange):
        # the numerator of the equation
        # every time do roll to the left size and calculate by the filter
        filters.uniform_filter(np.roll(norm_left, -curr_range - disp_range[0]) * norm_right, kernel, s)  # And normalization
        # denominator of the equation
        filters.uniform_filter(np.roll(norm_left, -curr_range - disp_range[0]) * np.roll(norm_left, -curr_range - disp_range[0]), kernel, s_l)
        filters.uniform_filter(norm_right * norm_right, kernel, s_r)
        all_dim[:, :, curr_range] = s / np.sqrt(s_l * s_r)
    return np.argmax(all_dim, axis=2)


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
        Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
        returns the homography and the error between the transformed points to their
        destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

        src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
        dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

        return: (Homography matrix shape:[3,3],
                Homography error)
    """
    loc = 0
    MAT = np.zeros((8, 9))
    for i in range(0, src_pnt.shape[0]):
        x = src_pnt[i][0]
        y = src_pnt[i][1]
        u = dst_pnt[i][0]
        v = dst_pnt[i][1]
        MAT[loc] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
        MAT[loc + 1] = [x, y, 1, 0, 0, 0, -u * x, -u * y, -u]
        loc = loc + 2
    res = np.linalg.svd(MAT)[2]
    h = res[-1, :] / res[-1, -1]
    H = h.reshape(3, 3)
    return H


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.
       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.
       output:
        None.
    """

    # dst_p = []
    # fig1 = plt.figure()
    #
    # def onclick_1(event):
    #     x = event.xdata
    #     y = event.ydata
    #     print("Loc: {:.0f},{:.0f}".format(x, y))
    #
    #     plt.plot(x, y, '*r')
    #     dst_p.append([x, y])
    #
    #     if len(dst_p) == 4:
    #         plt.close()
    #     plt.show()
    #
    # # display image 1
    # cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    # plt.imshow(dst_img)
    # plt.show()


    # check this points as dest point:
    dst_p = np.array([[286, 333],
                     [1800, 183],
                     [290, 743],
                     [1850, 688]])

    # check the points as src points:
    src_p = np.array([[0, 0],
                      [0, src_img.shape[1] - 1],
                      [src_img.shape[0] - 1, 0],
                      [src_img.shape[0] - 1, src_img.shape[1] - 1]])

    # calculate the homography
    h = computeHomography(dst_p, src_p)

    # calculate the warping
    for x in range(dst_img.shape[0] - 1):
        for y in range(dst_img.shape[1] - 1):
            res = h.dot(np.array([[x], [y], [1]]))
            normalized = h[2, :].dot(np.array([[x], [y], [1]]))
            res = res / normalized

            # check that the stitch keep on the border of the dest image
            if 0 < res[0] < src_img.shape[0] and 0 < res[1] < src_img.shape[1]:
                dst_img[y][x][:] = src_img[int(res[0]), int(res[1]), :]

    plt.matshow(dst_img)
    plt.colorbar()
    plt.show()

