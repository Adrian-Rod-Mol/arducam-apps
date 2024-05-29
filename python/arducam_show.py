from pathlib import Path
from argparse import ArgumentParser, Namespace
from src.utils import (
    read_arducam_image,
    ComputerScreen,
    show_image,
    generate_arducam_mosaic,
)

resolution_map = {
    "LOW": {"width": 1328, "height": 990, "band_width": int(1328 / 2), "band_height": int(990 / 2), "framerate": 30},
    "MEDIUM": {"width": 2024, "height": 1520, "band_width": int(2024 / 2), "band_height": int(1520 / 2), "framerate": 15},
    "HIGH": {"width": 4064, "height": 3040, "band_width": int(4064 / 2), "band_height": int(3040 / 2), "framerate": 15}
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
    window_size = (660, 480)
    image_shape = (current_res["width"], current_res["height"])
    screen_size = ComputerScreen.get_screen_size(0)
    screen = ComputerScreen(*screen_size)
    window_size = screen.get_width_with_aspect_ratio(*image_shape)
    window_size = (int(window_size[1] * 0.95), int(window_size[0] * 0.95))

    input_path = Path(args.input_folder)
    if not input_path.is_dir():
        print("Input folder not found")
        return False

    image_path_list = input_path.glob("*.raw")
    image_path_sorted = sorted(image_path_list, key=lambda x: int(x.stem))
    for image_path in image_path_sorted:
        print(image_path)
        image = read_arducam_image(image_path, current_res)
        #mosaic = generate_arducam_mosaic(image)
        show_image("Arducam", image, args.ms, window_size, (0, 0))


if __name__ == "__main__":
    main()