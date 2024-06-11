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
def blue_demosaicing(image, out_image, image_data):
    """
    image_data:
        0: start position of the image
        1: size of the image
        2: total number of rows
        3: total number of columns
    """
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bd = cuda.blockDim.x
    pos = bx * bd + tx
    if pos < image_data[1]:
        row_and_col_index = numba.float32(pos) / image_data[3]
        row = numba.uint32(row_and_col_index)
        col_index = numba.uint32(pos - row*image_data[3])
        row_odd_or_even = row % 2
        col_odd_or_even = col_index % 2
        if row_odd_or_even == 0:
            # A real blue value, copied directly to the output image
            if col_odd_or_even == 0:
                out_image[image_data[0] + pos] = image[image_data[0]+pos]
            else:
                # Can't do an interpolation with the last value. The previous blue is assigned
                if col_index == (image_data[3] - 1):
                    out_image[image_data[0] + pos] = image[image_data[0] + pos - 1]
                # In the first row, an interpolation between the previous blue value and the next is made
                else:
                    new_blue = (image[image_data[0] + pos - 1] + image[image_data[0] + pos + 1]) / 2
                    out_image[image_data[0] + pos] = new_blue
        else:
            # On the last row
            if row == image_data[2] - 1:
                if col_odd_or_even == 0:
                    # If it is even, the previous blue is copied
                    out_image[image_data[0] + pos] = image[image_data[0] + pos - image_data[3]]
                else:
                    if col_index == (image_data[3] - 1):
                        out_image[image_data[0] + pos] = image[image_data[0] + pos - image_data[3] - 1]
                    # Otherwise, the two superior columns are interpolated
                    else:
                        new_blue = (image[image_data[0] + pos - image_data[3] - 1]
                                    + image[image_data[0] + pos - image_data[3] + 1]) / 2
                        out_image[image_data[0] + pos] = new_blue

            else:
                if col_odd_or_even == 0:
                    # If the column is even, an interpolation between the superior and inferior blue is made
                    new_blue = (image[image_data[0] + pos - image_data[3]] + image[image_data[0] + pos + image_data[3]]) / 2
                    out_image[image_data[0] + pos] = new_blue
                else:
                    if col_index == (image_data[3] - 1):
                        # If is the last column, the interpolation is only between the two previous corners
                        new_blue = (image[image_data[0] + pos - image_data[3] - 1]
                                    + image[image_data[0] + pos + image_data[3] - 1]) / 2
                        out_image[image_data[0] + pos] = new_blue
                    else:
                        # If is odd, an interpolation between the four blues around the position is made
                        new_blue = (image[image_data[0] + pos - image_data[3] - 1]
                                    + image[image_data[0] + pos - image_data[3] + 1]
                                    + image[image_data[0] + pos + image_data[3] - 1]
                                    + image[image_data[0] + pos + image_data[3] + 1]) / 4
                        out_image[image_data[0] + pos] = new_blue


@cuda.jit
def red_demosaicing(image, out_image, image_data):
    """
    image_data:
        0: start position of the image
        1: size of the image
        2: total number of rows
        3: total number of columns
    """
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bd = cuda.blockDim.x
    pos = bx * bd + tx
    if pos < image_data[1]:
        row_and_col_index = numba.float32(pos) / image_data[3]
        row = numba.uint32(row_and_col_index)
        col_index = numba.uint32(pos - row*image_data[3])
        row_odd_or_even = row % 2
        col_odd_or_even = col_index % 2
        if row_odd_or_even == 0:
            # On the first row
            if row == 0:
                if col_odd_or_even == 0:
                    if col_index == 0:
                        out_image[image_data[0] + pos] = image[image_data[0] + pos + image_data[3] + 1]
                    # Otherwise, the two inferior columns are interpolated
                    else:
                        new_red = (image[image_data[0] + pos + image_data[3] - 1]
                                    + image[image_data[0] + pos + image_data[3] + 1]) / 2
                        out_image[image_data[0] + pos] = new_red
                else:
                    # If it is odd, the red below is copied
                    out_image[image_data[0] + pos] = image[image_data[0] + pos + image_data[3]]
                    
            else:
                if col_odd_or_even == 0:
                    if col_index == 0:
                        # If is the first column, the interpolation is only between the two next corners
                        new_red = (image[image_data[0] + pos - image_data[3] + 1]
                                    + image[image_data[0] + pos + image_data[3] + 1]) / 2
                        out_image[image_data[0] + pos] = new_red
                    else:
                        # If is even, an interpolation between the four reds around the position is made
                        new_red = (image[image_data[0] + pos - image_data[3] - 1]
                                    + image[image_data[0] + pos - image_data[3] + 1]
                                    + image[image_data[0] + pos + image_data[3] - 1]
                                    + image[image_data[0] + pos + image_data[3] + 1]) / 4
                        out_image[image_data[0] + pos] = new_red
                    
                else:
                    # If the column is odd, an interpolation between the superior and inferior red is made
                    new_red = (image[image_data[0] + pos - image_data[3]] + image[
                        image_data[0] + pos + image_data[3]]) / 2
                    out_image[image_data[0] + pos] = new_red

        else:
            if col_odd_or_even == 0:
                # Can't do an interpolation with the first value. The next red is assigned
                if col_index == 0:
                    out_image[image_data[0] + pos] = image[image_data[0] + pos + 1]
                else:
                    # An interpolation between the previous red value and the next is made
                    new_red = (image[image_data[0] + pos - 1] + image[image_data[0] + pos + 1]) / 2
                    out_image[image_data[0] + pos] = new_red
            else:
                # A real red value, copied directly to the output image
                out_image[image_data[0] + pos] = image[image_data[0] + pos]

@cuda.jit()
def nir_filtering(image, out_image, image_data, filter):
    """
        image_data:
            0: start position of the image
            1: size of the image
            2: total number of rows
            3: total number of columns
        """
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bd = cuda.blockDim.x
    pos = bx * bd + tx
    if pos < image_data[1]:
        row_and_col_index = numba.float32(pos) / image_data[3]
        row = numba.uint32(row_and_col_index)
        col_index = numba.uint32(pos - row * image_data[3])
        row_odd_or_even = row % 2
        col_odd_or_even = col_index % 2
        out_image[image_data[0] + pos] = image[image_data[0] + pos] * filter[row_odd_or_even * 2 + col_odd_or_even]
                        

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
        reflectance[pos] = value


@cuda.jit
def gpu_reflectance_with_kernel(image, white, black, kernel, reflectance):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bd = cuda.blockDim.x
    pos = bx * bd + tx
    if pos < image.shape[0]:
        image_float = numba.float32(image[pos])
        first = image_float * kernel[pos] - black[pos] * 0.8
        white_float = numba.float32(white[pos])
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
        half_height = int(current_res["band_height"] / 2)
        half_width = int(current_res["band_width"] / 2)
        pixel_square = white_resh[i, half_height:half_height + 2, half_width:half_width + 2]

        pixel_square = pixel_square / np.max(pixel_square)
        matrix = np.where(pixel_square > 0.9, True, False)

        if np.all(matrix == np.array([[False, False], [False, True]])):
            # Great response in the red filter
            type_list.append(0)
        elif np.all(matrix == np.array([[False, True], [True, False]])):
            # Great response in the green filter
            type_list.append(1)
        elif np.all(matrix == np.array([[True, False], [False, False]])):
            # Great response in the blue filter
            type_list.append(2)
        elif np.all(matrix == np.array([[True, True], [True, True]])):
            # Great response in all filters
            type_list.append(3)
    return type_list


def calculate_filter_kernel(white_ref: np.ndarray, current_res) -> np.ndarray:
    white_resh = white_ref.reshape(4, current_res["band_height"], current_res["band_width"])
    kernel = np.empty((4, 2, 2), dtype=np.float32)
    for i in range(4):
        half_width = int(current_res["band_width"] / 2)
        half_height = int(current_res["band_height"] / 2)
        band_max = np.max(white_resh[i, :, :].astype(np.float32))
        band_kernel = band_max / white_resh[i, :, :]
        kernel[i, :, :] = band_kernel[half_height:half_height + 2, half_width:half_width + 2]
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
        correction_blocks_per_grid = int(
            np.ceil(
                (current_res["band_height"] * current_res["band_width"] + threads_per_block - 1) / threads_per_block))
        black_cal = np.fromfile(args.black_calibration, dtype=np.uint16)
        white_cal = np.fromfile(args.white_calibration, dtype=np.uint16)
        type_list = select_interpolation_type(white_cal, current_res)
        filter_list = calculate_filter_kernel(white_cal, current_res)
        black_d = cuda.to_device(black_cal)
        white_d = cuda.to_device(white_cal)
        while True:
            raw = np.fromfile(image_path_sorted[index], dtype=np.uint16)
            raw_d = cuda.to_device(raw)
            reflectance = np.zeros(shape=current_res["width"] * current_res["height"], dtype=np.float32)
            ref_d = cuda.to_device(reflectance)
            gpu_reflectance[blocks_per_grid, threads_per_block](raw_d, white_d, black_d, ref_d)
            corrected_image = np.zeros(shape=current_res["width"] * current_res["height"], dtype=np.float32)
            corrected_d = cuda.to_device(corrected_image)
            for i, image_type in enumerate(type_list):
                arr = np.array([
                    i * current_res["band_width"] * current_res["band_height"],
                    current_res["band_width"] * current_res["band_height"],
                    current_res["band_height"],
                    current_res["band_width"]
                ], dtype=np.uint32)
                image_data = cuda.to_device(arr)
                if image_type == 0:
                    red_demosaicing[correction_blocks_per_grid, threads_per_block](ref_d, corrected_d, image_data)
                elif image_type == 2:
                    blue_demosaicing[correction_blocks_per_grid, threads_per_block](ref_d, corrected_d, image_data)
                elif image_type == 3:
                    current_filter = filter_list[i].flatten()
                    filter_d = cuda.to_device(current_filter)
                    nir_filtering[correction_blocks_per_grid, threads_per_block](ref_d, corrected_d, image_data, filter_d)

            image = corrected_d.copy_to_host().reshape(4, current_res["band_height"], current_res["band_width"])*4095
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
