import queue
import threading

import cv2 as cv
import os
import numpy as np
import math
import datetime as dt
from pathlib import Path
from typing import Tuple


class ComputerScreen:
    def __init__(self, width: int = 0, height: int = 0):
        self.height = height
        self.width = width

    @staticmethod
    def get_screen_size(screen_id: int) -> Tuple[int, int]:
        """Return the screen resolution of the selected screen (only valid for linux)
        params:
            screen_id   : index of the screen from which the resolution is to be obtained
        returns:
            Tuple with height and width in pixel of the selected screen
            """
        import subprocess
        output = subprocess.Popen(
            'xrandr | grep "\*" | cut -d" " -f4', shell=True, stdout=subprocess.PIPE).communicate()[0]
        screen_resolution_list_str = output.decode('utf-8')[:-1].split('\n')
        screen_resolution_list = [
            (int(res.split('x')[0]), int(res.split('x')[1])) for res in screen_resolution_list_str]
        return screen_resolution_list[screen_id]

    def get_height_with_aspect_ratio(self, input_width: int, input_height: int) -> Tuple[int, int]:
        return self.width, math.floor(input_height / input_width * self.width)

    def get_width_with_aspect_ratio(self, input_width: int, input_height: int) -> Tuple[int, int]:
        return math.floor(input_height / input_width * self.height), self.height


def show_image(
        name: str,
        image: np.ndarray,
        ms_sleep: int,
        window_size: Tuple[int, int] = (-1, -1),
        window_position: Tuple[int, int] = (-1, -1)):
    """
    Split the images in the 9 bands and show it in the screen
    params:
        image       : Raw image obtained from the camera
        ms_sleep    : Milliseconds waited between frames
    returns: None
    """

    cv.namedWindow(name, cv.WINDOW_NORMAL)
    if window_size != (-1, -1):
        cv.resizeWindow(name, *window_size)
    if window_position != (-1, -1):
        cv.moveWindow(name, *window_position)
    cv.imshow(name, image)
    cv.waitKey(1)


def generate_new_capturing_folder(output_path: Path) -> Path:
    capturing_path = output_path.joinpath(dt.datetime.now().strftime('%Y_%m_%d__%H_%M'))
    folder_count = 0
    while capturing_path.is_dir():
        capturing_path = output_path.joinpath(
            dt.datetime.now().strftime('%Y_%m_%d__%H_%M') + str(folder_count))
        folder_count += 1

    return capturing_path


def generate_arducam_mosaic(image: np.ndarray) -> np.ndarray:
    image_scaled = (image.astype(np.float32) / 4095.0) * 255.0
    image_to_split = image_scaled.astype(np.uint8)
    band_height = image_to_split.shape[1]
    band_width = image_to_split.shape[2]
    mosaic = np.empty((band_height * 2, band_width * 2), dtype=np.uint8)
    mosaic[0:band_height, 0:band_width] = image_to_split[0, :, :]
    mosaic[0:band_height, band_width:2 * band_width] = image_to_split[1, :, :]
    mosaic[band_height:2 * band_height, 0:band_width] = image_to_split[2, :, :]
    mosaic[band_height:2 * band_height, band_width:2 * band_width] = image_to_split[3, :, :]

    return mosaic

def arducam_mosaic_thread(input_queue: queue.Queue, output_queue: queue.Queue, stop_event : threading.Event):
    while not stop_event.is_set():
        if not input_queue.empty():
            image = input_queue.get()
            mosaic = generate_arducam_mosaic(image)
            output_queue.put(mosaic)

