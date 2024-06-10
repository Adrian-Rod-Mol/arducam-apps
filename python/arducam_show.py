from pathlib import Path
from argparse import ArgumentParser, Namespace

import numpy
from numba import cuda
import numba
import cv2 as cv
import numpy as np
from src.utils import (
    read_arducam_image,
    ImageDisplay
)

resolution_map = {
    "LOW": {"width": 1328, "height": 990, "band_width": int(1328 / 2), "band_height": int(990 / 2), "framerate": 30},
    "MEDIUM": {"width": 2024, "height": 1520, "band_width": int(2024 / 2), "band_height": int(1520 / 2),
               "framerate": 15},
    "HIGH": {"width": 4056, "height": 3040, "band_width": int(4056 / 2), "band_height": int(3040 / 2), "framerate": 15}
}


@cuda.jit
def gpu_reflectance(image, white, black, reflectance):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bd = cuda.blockDim.x
    pos = bx * bd + tx
    if pos < image.shape[0]:
        value = (image[pos] - black[pos] * 0.8) / (white[pos] - black[pos] * 0.8)
        if value < 0:
            value = 0
        elif value > 1:
            value = 1
        reflectance[pos] = value * 4095
@cuda.jit
def gpu_reflectance_with_kernel(image, white, black, kernel, reflectance):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bd = cuda.blockDim.x
    pos = bx * bd + tx
    if pos < image.shape[0]:
        image_float = numpy.float32(image[pos])
        first = image_float*kernel[pos] - black[pos] * 0.8
        white_float = numpy.float32(white[pos])
        second = white_float * kernel[pos] - black[pos] * 0.8
        value = first / second
        if value < 0:
            value = 0
        elif value > 1:
            value = 1
        reflectance[pos] = value * 4095


def get_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_folder",
        help="Path to the folder in which the images are stored",
        type=str,
        required=False,
        default="outputs",
        action='store')

    parser.add_argument(
        "-res",
        "--resolution",
        help="Selected resolution for the Arducam camera",
        type=str,
        required=True,
        default="MEDIUM",
        choices=["LOW", "MEDIUM", "HIGH"],
        action='store')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--raw',
        action='store_true',
        help="Process raw images without calibration")
    group.add_argument('--cal',
                       action='store_true',
                       help="Process images with calibration")

    parser.add_argument(
        "-bc",
        "--black_calibration",
        help="Path to the black calibration image",
        type=str,
        required=False,
        default="black.raw",
        action='store')

    parser.add_argument(
        "-wc",
        "--white_calibration",
        help="Path to the white calibration image",
        type=str,
        required=False,
        default="white.raw",
        action='store')

    args = parser.parse_args()

    if args.cal and (args.black_calibration is None or args.white_calibration is None):
        parser.error("--cal requires --black_calibration and --white_calibration")

    return args

def select_interpolation_type(white_ref: np.ndarray, current_res) -> list:
    white_resh = white_ref.reshape(4, current_res["band_height"], current_res["band_width"])
    type_list = []
    for i in range(white_resh.shape[0]):
        half_height = int(current_res["band_height"]/2)
        half_width = int(current_res["band_width"]/2)
        pixel_square = white_resh[i, half_height:half_height+2, half_width:half_width+2]

        pixel_square = pixel_square/np.max(pixel_square)
        matrix = np.where(pixel_square > 0.9, True, False)
        print(pixel_square)
        print(matrix)
        type_list.append(1)
    return type_list




def calculate_filter_kernel(white_ref: np.ndarray, current_res) -> np.ndarray:
    white_resh = white_ref.reshape(4, current_res["band_height"], current_res["band_width"])
    kernel = np.empty((4, current_res["band_height"], current_res["band_width"]), dtype=np.float32)
    for i in range(4):
        band_max = np.max(white_resh[i, :, :].astype(np.float32))
        print(band_max)
        band_kernel = band_max / white_resh[i, :, :]
        kernel[i, :, :] = band_kernel
    return kernel


def main():
    args = get_arguments()

    try:
        current_res = resolution_map[args.resolution]
    except KeyError:
        print("Input resolution not implemented")
        return

    # Default window size
    image_display = ImageDisplay(0, current_res["width"], current_res["height"])

    input_path = Path(args.input_folder)
    if not input_path.is_dir():
        print("Input folder not found")
        return False

    image_path_list = input_path.glob("*.raw")
    index = 0
    image_path_sorted = sorted(image_path_list, key=lambda x: int(x.stem))
    image_display.setup_window("Arducam")

    if args.cal:
        threads_per_block = 256
        blocks_per_grid = int(
            np.ceil(
                (current_res["height"] * current_res["width"] + threads_per_block - 1) / threads_per_block))
        black_cal = np.fromfile(args.black_calibration, dtype=np.uint16)
        white_cal = np.fromfile(args.white_calibration, dtype=np.uint16)
        a = select_interpolation_type(white_cal, current_res)
        black_d = cuda.to_device(black_cal)
        white_d = cuda.to_device(white_cal)
        while True:
            raw = np.fromfile(image_path_sorted[index], dtype=np.uint16)
            raw_d = cuda.to_device(raw)
            reflectance = np.zeros(shape=current_res["width"] * current_res["height"], dtype=np.float32)
            ref_d = cuda.to_device(reflectance)
            gpu_reflectance[blocks_per_grid, threads_per_block](raw_d, white_d, black_d, ref_d)
            image = ref_d.copy_to_host()
            image = image.reshape(4, current_res["band_height"], current_res["band_width"])
            key = image_display.study_frame("Arducam", image, index)
            if key == ord('a') and index > 0:
                index -= 1
            elif key == ord('d') and index < len(image_path_sorted) - 1:
                index += 1
            elif key == ord('q'):
                break

    elif args.raw:
        while True:
            image = read_arducam_image(image_path_sorted[index], current_res)
            key = image_display.study_frame("Arducam", image, index)
            if key == ord('a') and index > 0:
                index -= 1
            elif key == ord('d') and index < len(image_path_sorted) - 1:
                index += 1
            elif key == ord('q'):
                break


if __name__ == "__main__":
    main()
