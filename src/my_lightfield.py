import numpy as np
import cv2
from scipy.signal import correlate2d

from matplotlib import pyplot as plt
from matplotlib import patches as patches

import utils


def read_video(video_path):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_FPS, 30)

    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1280, 720)) / 255.0
        frames.append(frame)

    return frames


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


def rgb2grayscale(srgb_img):
    xyz_img = rgb2xyz(srgb_img)
    grayscale_img = get_luminance(xyz_img)

    return grayscale_img


def get_filtered_image(image, filter):
    return correlate2d(image, filter, mode='valid')


def compute_weights(frame, gray_template_diff, filter, size, x_start, y_start, window_size):

    size_2 = size // 2
    window = frame[x_start-size_2:x_start+window_size+size_2,
                   y_start-size_2:y_start+window_size+size_2]

    # Transfer the window to a grayscale image
    gray_window = rgb2grayscale(window)

    weights = np.zeros((window_size, window_size)).astype(np.float32)
    for i in range(window_size):
        for j in range(window_size):
            crop = gray_window[i:i+size, j:j+size]
            crop_filtered = get_filtered_image(crop, filter)
            crop_diff = crop - crop_filtered

            num = (np.abs(crop_diff * gray_template_diff)).sum()
            den = np.sqrt((gray_template_diff**2).sum() * (crop_diff**2).sum())

            weights[i, j] = num / den

    return weights


def get_refocused_img(frames, template, size, x_start, y_start, window_size, step=5):
    num_frames = len(frames)
    size_2 = size // 2
    window_size_2 = window_size // 2
    h, w = frames[0].shape[:2]

    # Create the box filter and normalize the sum to 1
    box_filter = np.ones((size, size)).astype(np.float32)
    box_filter /= (size * size)

    # Transfer the template to a grayscale image
    gray_template = rgb2grayscale(template)

    # Compute g[i, j] filtered with a box filter
    gray_template_filtered = get_filtered_image(gray_template, box_filter)
    gray_template_diff = gray_template - gray_template_filtered

    refocused_img = np.zeros_like(frames[0]).astype(np.float32)
    num_img = 0
    for idx in range(0, num_frames, step):
        frame = frames[idx]

        weights = compute_weights(
            frame, gray_template_diff, box_filter, size, x_start, y_start, window_size)

        shift = np.argmax(weights)
        sx = shift // window_size
        sy = shift - sx * window_size

        sx = sx - window_size_2
        sy = sy - window_size_2

        rx_start = -sx if sx < 0 else 0
        ry_start = -sy if sy < 0 else 0
        rx_end = h if sx < 0 else h - sx
        ry_end = w if sy < 0 else w - sy

        fx_start = 0 if sx < 0 else sx
        fy_start = 0 if sy < 0 else sy
        fx_end = h + sx if sx < 0 else h
        fy_end = w + sy if sy < 0 else w

        refocused_img[rx_start:rx_end, ry_start:ry_end,
                      :] += frame[fx_start:fx_end, fy_start:fy_end, :]
        num_img += 1

    refocused_img /= num_img

    return refocused_img


def main():
    video_path = '../data/lightfield.mp4'
    frames = read_video(video_path)[100:250]

    # Get the middle frame of the video sequence
    num_frames = len(frames)
    mid_frame = frames[int(num_frames / 2)]

    # bottle
    x_start, y_start = 200, 380

    size = 40
    window_size = 210

    offset_x = 20
    offset_y = 80

    # Show the template in the middle frame
    rect = patches.Rectangle(
        (x_start, y_start), size, size, linewidth=2, edgecolor='r', facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    plt.imshow(mid_frame)
    plt.show()

    template = mid_frame[y_start:y_start+size, x_start:x_start+size]
    refocused_img = get_refocused_img(frames, template, size,
                                      y_start-offset_y, x_start-offset_x, window_size, step=2)

    plt.imshow(refocused_img)
    plt.show()


if __name__ == '__main__':
    main()
