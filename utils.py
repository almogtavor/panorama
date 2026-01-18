import numpy as np
from PIL import Image
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates, sobel, gaussian_filter
from scipy.ndimage.filters import convolve
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

GRAYSCALE_MAX = 255
GRAY_REPRESENTATION = 1


def read_image(filename, representation):
    """
    A function that reads an image from a given path and returns the image in the representation that was received as an
    argument - 1 for grayscale, 2 for RGB.
    """
    im = np.array(Image.open(filename)).astype(np.float64) / GRAYSCALE_MAX
    if representation == GRAY_REPRESENTATION:  # output image as grayscale
        im = rgb2gray(im)
    return im


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    A function that builds a gaussian pyramid of the given image. The pyramid is of maximal possible depth: Either in
    depth max_levels, or in the depth such that the image in the last level is not smaller then 16
    :param im: The original image
    :param max_levels: The maximal depth levels that should be constructed
    :param filter_size: The size of the gaussian filter that should be used for blurring the image between the different
    levels in the pyramid.
    :return: The gaussian pyramid and None (for backwards compatibility)
    """
    current_im_size = min(im.shape[0], im.shape[1])
    current_level = 1
    pyr = [im]
    while current_im_size >= 16 and current_level < max_levels:
        reduced_image = reduce(pyr[current_level - 1], filter_size)
        pyr.append(reduced_image)
        current_level += 1
        current_im_size /= 2
    return pyr


def reduce(img, filter_size):
    """
    A function that reduces the resolution of the given image by 2 (both width and length).
    The function first blurs the given image using a Gaussian filter and then reduces the resolution by
    choosing every second pixel in every second row.
    :param img: The image that should be reduced
    :param filter_size: The size of the gaussian filter that is used for blurring the image
    :return: The reduced image
    """
    sigma = (filter_size - 1) / 4
    blurred_image = gaussian_filter(img, sigma=sigma, mode='mirror')
    reduced_img = blurred_image[::2, ::2]
    return reduced_img


def blur_spatial(im, kernel_size):
    """
    A function that blurs an image in the spatial domain using convolution with a gaussian kernel
    :param im: The image that should be blurred
    :param kernel_size: The size of the gaussian kernel
    :return: The blurred image
    """
    if kernel_size == 1:
        return im

    sigma = (kernel_size - 1) / 4
    blurred_image = gaussian_filter(im, sigma=sigma, mode='mirror')
    return blurred_image


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int32)
    ret = np.zeros_like(image, dtype=bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret

def spread_out_corners(im, m, n, radius, harris_corner_detector):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :param harris_corner_detector: A function that detects corners in an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int32)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int32)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int32)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret

def visualize_points(im, points):
    """
    Visualize points on an image. This function overlays given points on an image.
    :param im: A 2D array representing an image.
    :param points: An array of points with shape (N,2) to be drawn on the image.
    """
    plt.imshow(im, cmap='gray')
    plt.scatter(points[:, 0], points[:, 1], marker='.', c='r')
    plt.show()

def dominant_orientation(im, x, y, rad=8):
    """
    Computes the dominant orientation of the gradient around a specified point in an image.

    param im: A 2d array representing an image.
    param x: int or float representing the x-coordinate of the center of the region to be examined.
    param y: int or float representing the y-coordinate of the center of the region to be examined.
    param rad: int, optional. The radius of the square region around (x, y) to consider for orientation computation. Defaults to 8.

    Returns: float representing the dominant orientation of the gradient around the specified point,
    given as an angle in radians.
    """
    Ix = sobel(im, axis=1)
    Iy = sobel(im, axis=0)

    xs = np.arange(-rad, rad + 1) + x
    ys = np.arange(-rad, rad + 1) + y
    X, Y = np.meshgrid(xs, ys)

    gx = map_coordinates(Ix, [Y, X], order=1, prefilter=False)
    gy = map_coordinates(Iy, [Y, X], order=1, prefilter=False)

    angles = np.arctan2(gy, gx)
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H

def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int32)
