from pathlib import Path
from argparse import ArgumentParser, Namespace
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

    parser.add_argument(
        "-ms",
        help="Milliseconds between images shown.",
        type=int,
        required=False,
        default=500,
        action='store')

    return parser.parse_args()


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
