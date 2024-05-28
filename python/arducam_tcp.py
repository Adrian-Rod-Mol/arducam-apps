import io

import numpy as np
import socket
import threading
import queue
import time
import select
from pathlib import Path
from argparse import ArgumentParser, Namespace
from src.utils import (
    generate_new_capturing_folder,
    ComputerScreen,
    show_image,
    generate_arducam_mosaic,
)
from src.terminal_tcp_interface import (
    UserAction,
    Message,
    print_terminal,
    terminal_input,
    send_message
)

resolution_map = {
    "LOW": {"width": 1328, "height": 990, "band_width": int(1328 / 2), "band_height": int(990 / 2), "framerate": 30},
    "MEDIUM": {"width": 2024, "height": 1520, "band_width": int(2024 / 2), "band_height": int(1520 / 2), "framerate": 15},
    "HIGH": {"width": 4056, "height": 3040, "band_width": int(4056 / 2), "band_height": int(3040 / 2), "framerate": 15}
}

TCP_PORT = 32233
TCP_MSG_PORT = 32211
IMG_BYTES = 4371840
LOOP_TIMEOUT = 2


def get_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--ip",
        help="IP in which the server is going to be started.",
        type=str,
        required=False,
        default="10.42.0.1",
        action='store')

    parser.add_argument(
        "--no_show",
        help="Show the images that are being received by TCP.",
        required=False,
        action='store_true')

    parser.add_argument(
        "--save",
        help="Save in a file the images that are being received by TCP.",
        required=False,
        action='store_true')

    parser.add_argument(
        "-o",
        "--output_folder",
        help="Path to the folder in which the images need to be saved",
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

    return parser.parse_args()


def receive_thread(server_ip: str,
                   server_port: int,
                   data_queue: queue.Queue,
                   client_connected: threading.Event,
                   start: threading.Event,
                   finish: threading.Event):
    img_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    img_server.bind((server_ip, server_port))
    try:
        while not finish.is_set():
            if start.is_set():
                img_server.listen(1)
                img_client, _ = img_server.accept()
                client_connected.set()
                mean_time = 0
                count = 0
                while start.is_set():
                    start_time = time.perf_counter_ns()
                    bytes_to_receive = IMG_BYTES
                    loop_timeout = 0
                    image_data = bytes()
                    while bytes_to_receive > 0 and loop_timeout < LOOP_TIMEOUT:
                        start_loop = time.time()
                        r, _, _ = select.select([img_client], [], [])
                        if r:
                            data = img_client.recv(bytes_to_receive)

                            if data:
                                image_data += data
                                bytes_to_receive -= len(data)
                        loop_timeout += time.time() - start_loop
                    data_queue.put(image_data)
                    mean_time += time.perf_counter_ns() - start_time
                    if loop_timeout < LOOP_TIMEOUT:
                        count += 1

                img_client.close()
                if count != 0:
                    mean_time /= count
                    print_terminal(0, f"Mean time elapsed receiving {count} images:  {mean_time / 1000000} ms")

    except Exception as e:
        print(e)

    finally:
        img_server.close()
        client_connected.clear()
        print_terminal(0, "Image receiving thread finished correctly.")
        return


def decode_thread(current_res: dict,
                  data_queue: queue.Queue,
                  image_queue: queue.Queue,
                  client_connected: threading.Event,
                  start: threading.Event,
                  finish: threading.Event):
    while not finish.is_set():
        count = 0
        mean_time = 0
        while not client_connected.is_set():
            if finish.is_set():
                return

        while start.is_set():
            start_time = time.perf_counter_ns()
            image = np.empty((IMG_BYTES,), dtype=np.uint8)
            loop_timeout = 0
            while loop_timeout < LOOP_TIMEOUT:
                start_loop = time.time()
                if not data_queue.empty():
                    data = data_queue.get()
                    if len(data) != 0:
                        image[:] = np.frombuffer(data, np.uint8)
                        break
                loop_timeout += time.time() - start_loop
            image = image.view(np.uint16)
            image_reshaped = image.reshape((4, current_res["band_height"], current_res["band_width"]))
            image_queue.put(image_reshaped)
            mean_time += time.perf_counter_ns() - start_time
            if loop_timeout < LOOP_TIMEOUT:
                count += 1
            else:
                start.clear()
        if count != 0:
            mean_time /= count
            print_terminal(0, f"Mean time elapsed processing {count} images:  {mean_time / 1000000} ms")
    print_terminal(0, "Image decoding thread finished correctly.")
    return


def main():
    # Define
    user_action_map = [
        UserAction(
            "Start the capturing process",
            "START",
            False
        ),
        UserAction(
            "Stop the capturing process",
            "STOP",
            False
        ),
        UserAction(
            "Set the exposure time",
            "EXPOSURE",
            True,
            20000,
            100
        ),
        UserAction(
            "Close the capturing script",
            "CLOSE",
            False
        ),
    ]

    data_queue = queue.Queue()
    image_queue = queue.Queue()
    msg_queue = queue.Queue()
    process_msg_queue = queue.Queue()
    client_event = threading.Event()
    start_event = threading.Event()
    finish_event = threading.Event()

    args = get_arguments()

    try:
        current_res = resolution_map[args.resolution]
    except KeyError:
        print("Input resolution not implemented")
        return

    msg_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    msg_server.bind((args.ip, TCP_MSG_PORT))
    msg_server.listen(1)

    print_terminal(0, "Waiting for client connection...")
    msg_conn, msg_addr = msg_server.accept()
    print_terminal(0, "A client has connected to the message queue.")
    msg_queue.put(Message(False, args.resolution))

    # Define default output or capturing folder to prevent errors and undefined behaviour
    output_folder = Path(args.output_folder)
    if not output_folder.is_dir():
        output_folder.mkdir()
    capturing_folder = output_folder

    # Default window size
    window_size = (660, 480)
    if not args.no_show:
        image_shape = (current_res["width"], current_res["height"])
        screen_size = ComputerScreen.get_screen_size(0)
        screen = ComputerScreen(*screen_size)
        window_size = screen.get_width_with_aspect_ratio(*image_shape)
        window_size = (int(window_size[1] * 0.95), int(window_size[0] * 0.95))

    show_count = 0
    save_count = 0
    skip_count = 0
    frames_to_skip = np.floor(current_res["framerate"]/5)
    mean_call_time = 0
    call_time = 0
    _threads = [
        threading.Thread(target=terminal_input, args=(user_action_map, process_msg_queue,)),
        threading.Thread(target=send_message, args=(msg_conn, msg_queue, finish_event,)),
        threading.Thread(target=receive_thread,
                         args=(args.ip, TCP_PORT, data_queue, client_event, start_event, finish_event,)),
        threading.Thread(target=decode_thread,
                         args=(current_res,
                               data_queue, image_queue, client_event, start_event, finish_event,)),
    ]

    try:
        for thread in _threads:
            thread.start()
        while not finish_event.is_set():
            if not process_msg_queue.empty():
                current_msg = process_msg_queue.get()
                if current_msg.key == "CLOSE":
                    msg_queue.put(current_msg)
                    time.sleep(1)
                    finish_event.set()
                    client_event.clear()
                    start_event.clear()
                    return

                elif current_msg.key == "START":
                    start_event.set()
                    if args.save:
                        capturing_folder = generate_new_capturing_folder(output_folder)

                    msg_queue.put(current_msg)

                elif current_msg.key == "STOP":
                    msg_queue.put(current_msg)
                    # Wait for all the data to be saved in the file
                    if args.save:
                        while not image_queue.empty():
                            image = image_queue.get()
                            filename = f"{save_count:08d}.raw"
                            file_path = capturing_folder.joinpath(filename)
                            image.tofile(file_path)
                            save_count += 1
                    if show_count != 0:
                        print_terminal(0, f"Mean time between calls to show image: {mean_call_time/show_count/10**6} ms")
                        
                    save_count = 0
                    show_count = 0

                elif current_msg.key == "EXPOSURE":
                    # Checks if image thread has been initialized or is currently receiving images
                    if start_event.is_set():
                        print_terminal(1, f"Can't set exposure while capturing.")
                    else:
                        print_terminal(0, f"Setting exposure to: {current_msg.value} us")
                        msg_queue.put(current_msg)

            if not image_queue.empty():
                image = image_queue.get()
                if not args.no_show:
                    if skip_count == frames_to_skip:
                        mosaic = generate_arducam_mosaic(image)
                        show_image("Arducam", mosaic, 0, window_size, (100, 100))
                        if call_time != 0:
                            mean_call_time += time.perf_counter_ns() - call_time
                        call_time = time.perf_counter_ns()
                        show_count += 1
                        skip_count = 0
                    else:
                        skip_count += 1

                if args.save:
                    filename = f"{save_count:08d}.raw"
                    file_path = capturing_folder.joinpath(filename)
                    image.tofile(file_path)
                    save_count += 1

    except KeyboardInterrupt:
        print("Code finished")

    except ValueError as e:
        print(e)
    finally:
        msg_conn.close()
        for _thread in _threads:
            if _thread.is_alive():
                _thread.join()


if __name__ == '__main__':
    main()
