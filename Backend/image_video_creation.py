# This file is used for testing purpose only


"""import cv2
import os


def img_to_video(image_folder, video_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        for _ in range(5 * fps):  # repeat each image 5*fps times
            video.write(img)

    cv2.destroyAllWindows()
    video.release()


# usage
img_to_video(
    "images\\multi-digit_images_10",
    "videos\\op_video\\output_longer.mp4",
    30,
)"""


import cv2
import os
import numpy as np


def img_to_video(image_folder, video_name, fps, b_space):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    # Create a blank image with the same dimensions as your frames
    blank_image = np.zeros((height, width, layers), np.uint8)

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        for _ in range(5 * fps):  # repeat each image 5*fps times
            video.write(img)
        for _ in range(b_space * fps):  # insert blank space for b_space duration
            video.write(blank_image)

    cv2.destroyAllWindows()
    video.release()


# usage
img_to_video(
    "images\\multi-digit_images_10",
    "videos\\op_video\\output_longer.mp4",
    30,
    30,  # duration of blank space in seconds
)
