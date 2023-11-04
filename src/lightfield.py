import numpy as np
import cv2
import skimage
from scipy.interpolate import interp2d

from matplotlib import pyplot as plt

import utils


def process_lightfield_img(img, lenslet_size=16):
    h, w = img.shape[:2]
    h, w = h // lenslet_size, w // lenslet_size

    lightfield_img = np.zeros((lenslet_size, lenslet_size, h, w, 3))
    for u in range(lenslet_size):
        for v in range(lenslet_size):
            for c in range(3):
                lightfield_img[u, v, :, :, c] = img[u::lenslet_size,
                                                    v::lenslet_size, c] / 255.0

    return lightfield_img


def create_aperture_img(lightfield_img, lenslet_size=16):

    _, axes = plt.subplots(nrows=lenslet_size, ncols=lenslet_size, figsize=(14, 8),
                           gridspec_kw={'wspace': 0, 'hspace': 0})

    for u in range(lenslet_size):
        for v in range(lenslet_size):
            axes[u, v].imshow(lightfield_img[u, v, :, :, :])
            axes[u, v].axis('off')

    plt.show()


def refocusing_focal_stack_simulation(lightfield_img, ds, lenslet_size=16):
    # express the coordinates u and v relative to the center of each lenslet
    max_uv = (16 - 1) / 2  # 7.5
    us = np.arange(16) - max_uv
    vs = np.arange(16) - max_uv

    h, w = lightfield_img.shape[2:4]
    refocused_img = np.zeros((h, w, 3, len(ds))).astype(np.float32)

    xs = np.arange(w)
    ys = np.arange(h)

    for u in us:
        if abs(u) > lenslet_size / 2:
            continue
        for v in vs:
            if abs(v) > lenslet_size / 2:
                continue
            idx_u, idx_v = int(u + max_uv), int(v + max_uv)
            f0 = interp2d(xs, ys, lightfield_img[idx_u, idx_v, :, :, 0])
            f1 = interp2d(xs, ys, lightfield_img[idx_u, idx_v, :, :, 1])
            f2 = interp2d(xs, ys, lightfield_img[idx_u, idx_v, :, :, 2])

            d_idx = 0
            for d in ds:
                du, dv = d * u, d * v
                xnew = xs - dv
                ynew = ys + du  # why +du?
                refocused_img[:, :, 0, d_idx] += f0(xnew, ynew)
                refocused_img[:, :, 1, d_idx] += f1(xnew, ynew)
                refocused_img[:, :, 2, d_idx] += f2(xnew, ynew)
                d_idx += 1

    for d_idx in range(len(ds)):
        refocused_img[:, :, :, d_idx] /= (lenslet_size * lenslet_size)

    return refocused_img


def inverse_gamma_encoding(img):
    mask = (img <= 0.04045)

    inv_img = np.zeros_like(img)
    inv_img[mask] = img[mask] / 12.92
    inv_img[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4

    return inv_img


def get_luminance(xyz_img):
    # The luminance channel is the Y channel
    return xyz_img[:, :, 1]


def rgb2xyz(srgb_img):
    lrgb_img = inverse_gamma_encoding(srgb_img)
    xyz_img = utils.lRGB2XYZ(lrgb_img)

    return xyz_img


def depth_from_focus(srgb_imgs, kernel_size1=11, kernel_size2=11, sigma1=5, sigma2=5):
    stack_depth = srgb_imgs.shape[-1]
    h, w = srgb_imgs.shape[:2]
    sharpness_weights = np.zeros((h, w, stack_depth))

    for i in range(stack_depth):
        srgb_img = srgb_imgs[:, :, :, i]
        xyz_img = rgb2xyz(srgb_img)
        luminance_channel = get_luminance(xyz_img)

        low_freq_img = cv2.GaussianBlur(
            luminance_channel, ksize=(kernel_size1, kernel_size1), sigmaX=sigma1)
        high_freq_img = luminance_channel - low_freq_img

        w_sharpness = cv2.GaussianBlur(
            high_freq_img**2, ksize=(kernel_size2, kernel_size2), sigmaX=sigma2)

        sharpness_weights[:, :, i] = w_sharpness

    den = np.repeat(np.sum(sharpness_weights, axis=2)
                    [..., np.newaxis], repeats=3, axis=2)
    num = np.sum(np.repeat(
        sharpness_weights[:, :, np.newaxis, :], repeats=3, axis=2) * srgb_imgs, axis=3)

    return num / den, sharpness_weights


def create_depth_map(sharpness_weights, ds):
    num = np.zeros(sharpness_weights.shape[:2])
    for i in range(len(ds)):
        num += sharpness_weights[:, :, i] * ds[i]

    den = sharpness_weights.sum(axis=2)

    return 1.0 - num / den


def create_2d_collage(refocused_imgs, lenslet_sizes, ds, figsize=(22, 13)):
    num_d = refocused_imgs.shape[-1]
    num_a = refocused_imgs.shape[0]

    fig, axes = plt.subplots(
        num_a, num_d, figsize=figsize, constrained_layout=True)
    for i in range(num_a):
        for j in range(num_d):
            axes[i, j].imshow(refocused_imgs[i, :, :, :, j])
            axes[i, j].axis('off')
            axes[i, j].set_title(
                'lenslet size = {}, depth = {}'.format(lenslet_sizes[i], ds[j]))

    plt.show()


def get_afi(refocused_imgs, x, y):
    num_d = refocused_imgs.shape[3]
    num_a = refocused_imgs.shape[-1]

    afi = np.zeros((num_a, num_d, 3)).astype(np.float32)
    for i in range(num_a):
        for j in range(num_d):
            afi[i, j, :] = refocused_imgs[x, y, :, j, i]

    return afi


def get_per_pixel_depth_map(refocused_imgs, ds):

    # refocused_imgs: [num_apertures, h, w, c, num_depths]
    h, w, c, num_depths, num_apertures = refocused_imgs.shape

    depth_map = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            minvar_idx = np.argmin(
                np.sum(np.var(refocused_imgs[i, j, :], -1), 0))
            depth_map[i, j] = ds[minvar_idx]

    return depth_map


def create_focal_aperture_stack_and_confocal_stereo(lightfield_img, ds):
    lenslet_sizes = [2, 4, 6, 8, 10, 12, 14, 16]

    refocused_imgs = []
    for size in lenslet_sizes:
        refocused_img = refocusing_focal_stack_simulation(
            lightfield_img, ds, size)
        refocused_imgs.append(refocused_img)

    refocused_imgs = np.array(refocused_imgs)

    create_2d_collage(refocused_imgs, lenslet_sizes, ds)

    # [h, w, c, num_depths, num_apertures]
    refocused_imgs = np.transpose(refocused_imgs, (1, 2, 3, 4, 0))

    num_pixels = 30
    h, w = lightfield_img.shape[2:4]

    depth_map = get_per_pixel_depth_map(refocused_imgs, ds)

    return 1 - depth_map


def main():
    img_path = '../data/chessboard_lightfield.png'
    img = skimage.io.imread(img_path)

    depths = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]

    # Load the lightfield image and create from it a 5-dimensional array
    lightfield_img = process_lightfield_img(img)

    # Show the sub-aperture views
    create_aperture_img(lightfield_img)

    # Create refocused images
    refocused_img = refocusing_focal_stack_simulation(lightfield_img, depths)

    # Create all-in-focus image and the depth map
    all_in_focus_img, sharpness_weights = depth_from_focus(
        refocused_img, kernel_size1=21, kernel_size2=31, sigma1=2, sigma2=20)
    plt.imshow(all_in_focus_img)
    plt.show()

    depth_map = create_depth_map(sharpness_weights, depths)
    plt.imshow(depth_map)
    plt.show()

    # Focal-aperture stack and confocal stereo
    depth_map = create_focal_aperture_stack_and_confocal_stereo(
        lightfield_img, depths)
    plt.imshow(depth_map)
    plt.show()


if __name__ == '__main__':
    main()
