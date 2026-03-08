import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates

from utils import *  # noqa: F403


MIN_SCORE = 0.7


def harris_corner_detector(im):
    """
    Implements the harris corner detection algorithm.
    :param im: A 2D array representing a grayscale image.
    :return: An array with shape (N,2), where its ith entry is the [x,y] coordinates of the ith corner point.
    """
    ix_filter = np.array([[1, 0, -1]])
    iy_filter = ix_filter.T

    ix = convolve2d(im, ix_filter, mode="same", boundary="symm")
    iy = convolve2d(im, iy_filter, mode="same", boundary="symm")

    ix2 = blur_spatial(ix * ix, 3)
    iy2 = blur_spatial(iy * iy, 3)
    ixy = blur_spatial(ix * iy, 3)

    alpha = 0.04
    response = (ix2 * iy2 - ixy * ixy) - alpha * (ix2 + iy2) ** 2

    corners = non_maximum_suppression(response)
    ys, xs = np.nonzero(corners)
    return np.stack([xs, ys], axis=1)


def feature_descriptor(im, points, desc_rad=3):
    """
    Samples descriptors at the given feature points.
    :param im: A 2D array representing a grayscale image.
    :param points: An array with shape (N,2) representing feature points coordinates in the image.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: An array of 2D patches, each patch i representing the descriptor of point i.
    """
    k = 2 * desc_rad + 1
    descriptors = np.zeros((points.shape[0], k, k))
    if points.size == 0:
        return descriptors

    offsets = np.arange(-desc_rad, desc_rad + 1)
    grid_x, grid_y = np.meshgrid(offsets, offsets)

    for i, (x, y) in enumerate(points):
        coords_x = grid_x + x
        coords_y = grid_y + y
        patch = map_coordinates(
            im, [coords_y, coords_x], order=1, prefilter=False
        ).reshape(k, k)
        patch = patch - patch.mean()
        norm = np.linalg.norm(patch)
        if norm > 0:
            patch = patch / norm
        descriptors[i] = patch
    return descriptors


def find_features(im):
    """
    Detects and extracts feature points from a specific pyramid level.
    :param im: A 2D array representing a grayscale image.
    :return: A list containing:
             1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                These coordinates are provided at the original image level.
            2) A feature descriptor array with shape (N,K,K)
    """
    pyr = build_gaussian_pyramid(im, 3, 7)
    points = spread_out_corners(
        im, m=7, n=7, radius=12, harris_corner_detector=harris_corner_detector
    )
    points_level3 = points / 4
    desc = feature_descriptor(pyr[2], points_level3, desc_rad=3)
    return points, desc


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    if desc1.size == 0 or desc2.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    desc1_flat = desc1.reshape(desc1.shape[0], -1)
    desc2_flat = desc2.reshape(desc2.shape[0], -1)

    scores = desc1_flat @ desc2_flat.T

    k_row = min(2, desc2.shape[0])
    k_col = min(2, desc1.shape[0])
    row_top = np.argpartition(scores, -k_row, axis=1)[:, -k_row:]
    col_top = np.argpartition(scores, -k_col, axis=0)[-k_col:, :]

    matches1 = []
    matches2 = []
    for i in range(desc1.shape[0]):
        for j in row_top[i]:
            if scores[i, j] > min_score and i in col_top[:, j]:
                matches1.append(i)
                matches2.append(j)

    return np.array(matches1, dtype=int), np.array(matches2, dtype=int)


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    if pos1.size == 0:
        return pos1.copy()
    ones = np.ones((pos1.shape[0], 1))
    pos1_h = np.hstack([pos1, ones])
    transformed = pos1_h @ H12.T
    transformed /= transformed[:, [2]]
    return transformed[:, :2]


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    if points1.shape[0] < 2 or points2.shape[0] < 2:
        return np.eye(3), np.array([], dtype=int)

    best_inliers = np.array([], dtype=int)
    best_h = np.eye(3)
    n_points = points1.shape[0]

    for iter_idx in range(num_iter):
        sample_idx = np.random.choice(n_points, 2, replace=False)
        p1_sample = points1[sample_idx]
        p2_sample = points2[sample_idx]
        h12 = estimate_rigid_transform(p1_sample, p2_sample, translation_only)

        p1_trans = apply_homography(points1, h12)
        d2 = np.sum((p1_trans - points2) ** 2, axis=1)
        inliers = np.where(d2 < inlier_tol)[0]

        if inliers.size > best_inliers.size:
            best_inliers = inliers
            best_h = h12

    if best_inliers.size > 0:
        best_h = estimate_rigid_transform(
            points1[best_inliers], points2[best_inliers], translation_only
        )

    if best_h[2, 2] != 0:
        best_h = best_h / best_h[2, 2]

    return best_h, best_inliers


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :param points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    h1, w1 = im1.shape
    h2, w2 = im2.shape
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2))
    canvas[:h1, :w1] = im1
    canvas[:h2, w1 : w1 + w2] = im2

    plt.figure(figsize=(10, 6))
    plt.imshow(canvas, cmap="gray")
    if points1.size > 0:
        plt.scatter(points1[:, 0], points1[:, 1], c="r", s=6)
        plt.scatter(points2[:, 0] + w1, points2[:, 1], c="r", s=6)

    inlier_set = set(inliers.tolist())
    for i in range(points1.shape[0]):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        color = "b" if i in inlier_set else "y"
        plt.plot([x1, x2 + w1], [y1, y2], color=color, linewidth=0.5)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def accumulate_homographies(H_successive, m):
    """
    Convert a list of successive homographies to a list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    num_frames = len(H_successive) + 1
    H2m = [None] * num_frames
    H2m[m] = np.eye(3)

    for i in range(m - 1, -1, -1):
        H = np.eye(3)
        for j in range(i, m):
            H = H_successive[j] @ H
        H2m[i] = H / H[2, 2]

    for i in range(m + 1, num_frames):
        H = np.eye(3)
        for j in range(m, i):
            H = H @ np.linalg.inv(H_successive[j])
        H2m[i] = H / H[2, 2]

    return H2m


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    corners = np.array(
        [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float64
    )
    warped = apply_homography(corners, homography)
    x_min, y_min = np.min(warped, axis=0)
    x_max, y_max = np.max(warped, axis=0)
    return np.array([[x_min, y_min], [x_max, y_max]])


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    h, w = image.shape
    bbox = compute_bounding_box(homography, w, h)
    x_min, y_min = bbox[0].astype(np.int32)
    x_max, y_max = bbox[1].astype(np.int32)

    xs = np.arange(x_min, x_max + 1)
    ys = np.arange(y_min, y_max + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)

    coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    inv_h = np.linalg.inv(homography)
    source = apply_homography(coords, inv_h)

    sampled = map_coordinates(
        image,
        [source[:, 1], source[:, 0]],
        order=1,
        prefilter=False,
    )
    return sampled.reshape(grid_y.shape)


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    channels = [warp_channel(image[:, :, c], homography) for c in range(3)]
    return np.dstack(channels)


##################################################################################################


def align_images(files, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []
    start_time = time.time()
    for idx, file in enumerate(files, start=1):
        image = read_image(file, 1)
        points_and_descriptors.append(find_features(image))
        if idx % 50 == 0 or idx == len(files):
            elapsed = time.time() - start_time
            print(f"Features {idx}/{len(files)} extracted - elapsed {elapsed:.1f}s")

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
        points1, points2 = (
            points_and_descriptors[i][0],
            points_and_descriptors[i + 1][0],
        )
        desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

        # Find matching feature points.
        ind1, ind2 = match_features(desc1, desc2, MIN_SCORE)
        points1, points2 = points1[ind1, :], points2[ind2, :]

        # Compute homography using RANSAC.
        if (i + 1) % 100 == 0:
            print(f"RANSAC call {i + 1}/{len(points_and_descriptors) - 1}")
        H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

        Hs.append(H12)
        if (i + 1) % 100 == 0 or i == len(points_and_descriptors) - 2:
            print(f"Homography {i + 1}/{len(points_and_descriptors) - 1} computed")

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    homographies = np.stack(accumulated_homographies)
    frames_for_panoramas = filter_homographies_with_translation(
        homographies, minimum_right_translation=5
    )
    homographies = homographies[frames_for_panoramas]
    return frames_for_panoramas, homographies


def generate_panoramic_images(
    data_dir,
    file_prefix,
    num_images,
    out_dir,
    number_of_panoramas,
    translation_only=False,
):
    """
    combine slices from input images to panoramas.
    The naming convention for a sequence of images is file_prefixN.jpg, where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    :param out_dir: path to output panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """

    file_prefix = file_prefix
    files = [
        os.path.join(data_dir, "%s%03d.jpg" % (file_prefix, i + 1))
        for i in range(num_images)
    ]
    files = list(filter(os.path.exists, files))
    print("found %d images" % len(files))
    image = read_image(files[0], 1)
    h, w = image.shape

    frames_for_panoramas, homographies = align_images(files, translation_only)

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    bounding_boxes = np.zeros((frames_for_panoramas.size, 2, 2))
    for i in range(frames_for_panoramas.size):
        bounding_boxes[i] = compute_bounding_box(homographies[i], w, h)
        if (i + 1) % max(
            1, frames_for_panoramas.size // 5
        ) == 0 or i == frames_for_panoramas.size - 1:
            print(f"Bounding boxes {i + 1}/{frames_for_panoramas.size}")

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(bounding_boxes, axis=(0, 1))
    bounding_boxes -= global_offset

    slice_centers = np.linspace(
        0, w, number_of_panoramas + 2, endpoint=True, dtype=np.int32
    )[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
        slice_center_2d = np.array([slice_centers[i], h // 2])[None, :]
        # homography warps the slice center to the coordinate system of the middle image
        warped_centers = [apply_homography(slice_center_2d, h) for h in homographies]
        # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
        warped_slice_centers[i] = (
            np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]
        )

    panorama_size = np.max(bounding_boxes, axis=(0, 1)).astype(np.int32) + 1

    # boundary between input images in the panorama
    x_strip_boundary = (warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2
    x_strip_boundary = np.hstack(
        [
            np.zeros((number_of_panoramas, 1)),
            x_strip_boundary,
            np.ones((number_of_panoramas, 1)) * panorama_size[0],
        ]
    )
    x_strip_boundary = x_strip_boundary.round().astype(np.int32)

    panoramas = np.zeros(
        (number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64
    )
    for i, frame_index in enumerate(frames_for_panoramas):
        # warp every input image once, and populate all panoramas
        image = read_image(files[frame_index], 2)
        warped_image = warp_image(image, homographies[i])
        x_offset, y_offset = bounding_boxes[i][0].astype(np.int32)
        y_bottom = y_offset + warped_image.shape[0]

        for panorama_index in range(number_of_panoramas):
            # take strip of warped image and paste to current panorama
            boundaries = x_strip_boundary[panorama_index, i : i + 2]
            image_strip = warped_image[
                :, boundaries[0] - x_offset : boundaries[1] - x_offset
            ]
            x_end = boundaries[0] + image_strip.shape[1]
            panoramas[panorama_index, y_offset:y_bottom, boundaries[0] : x_end] = (
                image_strip
            )
        if (i + 1) % max(
            1, frames_for_panoramas.size // 5
        ) == 0 or i == frames_for_panoramas.size - 1:
            print(f"Warped frame {i + 1}/{frames_for_panoramas.size}")

    os.makedirs(out_dir, exist_ok=True)
    for i, panorama in enumerate(panoramas):
        plt.imsave("%s/panorama%02d.png" % (out_dir, i + 1), panorama)


if __name__ == "__main__":
    import sys
    import ffmpeg

    video_name = sys.argv[1] if len(sys.argv) > 1 else "mt_cook.mp4"
    video_name_base = video_name.split(".")[0]
    os.makedirs(f"dump/{video_name_base}", exist_ok=True)
    ffmpeg.input(f"videos/{video_name}").output(
        f"dump/{video_name_base}/{video_name_base}%03d.jpg"
    ).run()
    num_images = len(os.listdir(f"dump/{video_name_base}"))
    print(f"Generated {num_images} images")

    # Visualize feature points on two sample images
    print("Extracting and visualizing feature points...")
    image1 = read_image(f"dump/{video_name_base}/{video_name_base}200.jpg", 1)
    image2 = read_image(f"dump/{video_name_base}/{video_name_base}300.jpg", 1)

    # Extract feature points and descriptors
    points1, desc1 = find_features(image1)
    points2, desc2 = find_features(image2)

    # Visualize points on first image
    print(f"Found {len(points1)} feature points in image 1")
    visualize_points(image1, points1)

    # Visualize points on second image
    print(f"Found {len(points2)} feature points in image 2")
    visualize_points(image2, points2)

    # Match features between the two images
    print("Matching features between images...")
    ind1, ind2 = match_features(desc1, desc2, MIN_SCORE)
    matched_points1 = points1[ind1]
    matched_points2 = points2[ind2]
    print(f"Found {len(ind1)} matches")

    # Run RANSAC to find inliers
    H12, inliers = ransac_homography(
        matched_points1, matched_points2, 100, 6, translation_only=False
    )
    print(f"Found {len(inliers)} inliers out of {len(matched_points1)} matches")

    # Display matches with inliers and outliers
    display_matches(image1, image2, matched_points1, matched_points2, inliers)

    # Generate panoramic images
    print("\nGenerating panoramic images...")
    generate_panoramic_images(
        f"dump/{video_name_base}/",
        video_name_base,
        num_images=num_images,
        out_dir=f"out/{video_name_base}",
        number_of_panoramas=2,
    )
