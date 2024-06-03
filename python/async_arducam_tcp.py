import numpy as np
import threading
import asyncio
import time
from pathlib import Path
from argparse import ArgumentParser, Namespace
from src.utils import (
    generate_new_capturing_folder,
    ImageDisplay,
)
from src.async_terminal_tcp import (
    UserAction,
    print_terminal,
    async_terminal,
    message_server
)

resolution_map = {
    "LOW": {"width": 1328, "height": 990, "band_width": int(1328 / 2), "band_height": int(990 / 2), "framerate": 30},
    "MEDIUM": {"width": 2024, "height": 1520, "band_width": int(2024 / 2), "band_height": int(1520 / 2),
               "framerate": 15},
    "HIGH": {"width": 4056, "height": 3040, "band_width": int(4056 / 2), "band_height": int(3040 / 2), "framerate": 5}
}

TCP_PORT = 32233
TCP_MSG_PORT = 32211
TCP_CONF_PORT = 32121
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


async def read_image_task(reader: asyncio.StreamReader, img_bytes: int,
                          data_queue: asyncio.Queue, client_connected: asyncio.Event, start: asyncio.Event):
    mean_time = 0
    count = 0
    while start.is_set():
        start_time = time.perf_counter_ns()
        bytes_to_receive = img_bytes
        try:
            while bytes_to_receive > 0:
                data = await asyncio.wait_for(reader.read(bytes_to_receive), LOOP_TIMEOUT)
                if data:
                    await data_queue.put(data)
                    bytes_to_receive -= len(data)
            mean_time += time.perf_counter_ns() - start_time
            count += 1

        except asyncio.TimeoutError:
            print_terminal(1, "Waiting for the server to send the image timed out.")

    if count != 0:
        mean_time /= count
        print_terminal(0, f"Mean time elapsed receiving {count} images:  {mean_time / 1000000} ms")


async def receive_image_callback(reader, writer, img_bytes: int,
                                 data_queue: asyncio.Queue, client_connected: asyncio.Event, start: asyncio.Event):
    client_connected.set()
    await asyncio.shield(asyncio.create_task(read_image_task(reader, img_bytes, data_queue, client_connected, start)))

    writer.close()
    await writer.wait_closed()

    client_connected.clear()


async def receive_image_server(server_ip: str,
                               server_port: int,
                               img_bytes: int,
                               data_queue: asyncio.Queue,
                               client_connected: asyncio.Event,
                               start: asyncio.Event):
    try:
        img_server = await asyncio.start_server(
            lambda r, w:  receive_image_callback(r, w, img_bytes, data_queue, client_connected, start),
            server_ip, server_port)
        async with img_server:
            await img_server.serve_forever()
    except asyncio.CancelledError:
        pass


async def receive_task(server_ip: str,
                       server_port: int,
                       img_bytes: int,
                       data_queue: asyncio.Queue,
                       client_connected: asyncio.Event,
                       start: asyncio.Event,
                       finish: asyncio.Event):
    try:
        while not finish.is_set():
            while not finish.is_set() or not start.is_set():
                await asyncio.sleep(0.2)
            if finish.is_set():
                break
            elif start.is_set():
                img_server_task = asyncio.create_task(
                    receive_image_server(
                        server_ip, server_port, img_bytes, data_queue, client_connected, start))
                await client_connected.wait()
                img_server_task.cancel()

    except Exception as e:
        print(e)

    finally:
        print_terminal(0, "Image receiving task finished correctly.")


async def decode_task(current_res: dict,
                      img_bytes: int,
                      data_queue: asyncio.Queue,
                      image_queue: asyncio.Queue,
                      client_connected: asyncio.Event,
                      start: asyncio.Event,
                      finish: asyncio.Event):
    try:
        while not finish.is_set():
            count = 0
            mean_time = 0
            while not finish.is_set() or not client_connected.is_set():
                await asyncio.sleep(0.2)

            if finish.is_set():
                break

            while start.is_set() or not data_queue.empty():
                start_time = time.perf_counter_ns()
                image = np.empty((img_bytes,), dtype=np.uint8)
                data_received = 0
                try:
                    while data_received < img_bytes:
                        data = await asyncio.wait_for(data_queue.get(), LOOP_TIMEOUT)
                        if len(data) != 0:
                            image[data_received:len(data)] = np.frombuffer(data, np.uint8)
                            data_received += len(data)
                    image = image.view(np.uint16)
                    image_reshaped = image.reshape((4, current_res["band_height"], current_res["band_width"]))
                    await image_queue.put(image_reshaped)
                    mean_time += time.perf_counter_ns() - start_time
                    count += 1
                except asyncio.TimeoutError:
                    print_terminal(1, "Waiting for data in the image queue timed out.")

            if count != 0:
                mean_time /= count
                print_terminal(0, f"Mean time elapsed processing {count} images:  {mean_time / 1000000} ms")
    except Exception as e:
        raise e
    finally:
        print_terminal(0, "Image decoding task finished correctly.")


async def control_task(
        image_queue: asyncio.Queue,
        msg_queue: asyncio.Queue,
        process_msg_queue: asyncio.Queue,
        client_event: asyncio.Event,
        start_event: asyncio.Event,
        finish_event: asyncio.Event,
        output_folder: Path,
        capturing_folder: Path,
        image_display: ImageDisplay,
        save: bool,
        no_show: bool):
    try:
        save_count = 0
        while not finish_event.is_set():
            if not image_queue.empty():
                image = await image_queue.get()
                if not no_show:
                    image_display.show_frame("Arducam", image)

                if save:
                    filename = f"{save_count:08d}.raw"
                    file_path = capturing_folder.joinpath(filename)
                    image.tofile(file_path)
                    save_count += 1

            if not process_msg_queue.empty():
                current_msg = await process_msg_queue.get()
                if current_msg.key == "CLOSE":
                    finish_event.set()
                    await msg_queue.put(current_msg)
                    await asyncio.sleep(1)
                    client_event.clear()
                    start_event.clear()
                    return

                elif current_msg.key == "START":
                    start_event.set()
                    if save:
                        capturing_folder = generate_new_capturing_folder(output_folder)

                    await msg_queue.put(current_msg)

                elif current_msg.key == "STOP":
                    await msg_queue.put(current_msg)
                    # Wait for all the data to be saved in the file
                    if save:
                        while not image_queue.empty():
                            image = await image_queue.get()
                            filename = f"{save_count:08d}.raw"
                            file_path = capturing_folder.joinpath(filename)
                            image.tofile(file_path)
                            save_count += 1

                    save_count = 0

                elif current_msg.key == "EXPOSURE":
                    # Checks if image thread has been initialized or is currently receiving images
                    if start_event.is_set():
                        print_terminal(1, f"Can't set exposure while capturing.")
                    else:
                        print_terminal(0, f"Setting exposure to: {current_msg.value} us")
                        await msg_queue.put(current_msg)
            await asyncio.sleep(0.015)

    except Exception as e:
        raise e
    finally:
        print_terminal(0, "Control task finished correctly.")


async def configure_camera(reader, writer, resolution: str, configuration_complete: asyncio.Event):
    print_terminal(0, "Configuration client connected.")
    conf_message = ""
    if resolution == "LOW":
        conf_message = "--mode 1332:990:10:U --resolution LOW"
    elif resolution == "MEDIUM":
        conf_message = "--mode 2028:1520:12:U --resolution MEDIUM"
    elif resolution == "HIGH":
        conf_message = "--mode 4056:3040:12:U --resolution HIGH"

    writer.write(conf_message.encode('utf-8'))
    await writer.drain()

    writer.close()
    await writer.wait_closed()
    configuration_complete.set()


async def configure_camera_server(ip: str, port: int, resolution: str, configuration_complete: asyncio.Event):
    try:
        # Sending configuration to Raspberry
        print_terminal(0, "Waiting for connection to configure...")
        cnf_server = await asyncio.start_server(
            lambda w, r: configure_camera(w, r, resolution, configuration_complete), ip, port)
        async with cnf_server:
            await cnf_server.serve_forever()
    except asyncio.CancelledError:
        print_terminal(0, "Camera configured.")


async def main():
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

    data_queue = asyncio.Queue()
    image_queue = asyncio.Queue()
    msg_queue = asyncio.Queue()
    process_msg_queue = asyncio.Queue()
    client_event = asyncio.Event()
    start_event = asyncio.Event()
    finish_event = asyncio.Event()

    args = get_arguments()

    try:
        current_res = resolution_map[args.resolution]
    except KeyError:
        print("Input resolution not implemented")
        return

    # Bytes = height*with*bands*2 bytes each pixel
    image_bytes = current_res["band_width"] * current_res["band_height"] * 4 * 2

    config_complete = asyncio.Event()
    conf_task = asyncio.create_task(configure_camera_server(args.ip, TCP_CONF_PORT, args.resolution, config_complete))
    await config_complete.wait()
    conf_task.cancel()

    # Define default output or capturing folder to prevent errors and undefined behaviour
    output_folder = Path(args.output_folder)
    if not output_folder.is_dir():
        output_folder.mkdir()
    capturing_folder = output_folder

    image_display = ImageDisplay(0, current_res["width"], current_res["height"])

    try:
        tk_terminal = asyncio.create_task(async_terminal(user_action_map, process_msg_queue))
        tk_message = asyncio.create_task(message_server(args.ip, TCP_MSG_PORT, msg_queue, finish_event))
        tk_receive = asyncio.create_task(
            asyncio.shield(
                receive_task(args.ip, TCP_PORT, image_bytes, data_queue, client_event, start_event, finish_event)))
        tk_decode = asyncio.create_task(
            asyncio.shield(
                decode_task(
                    current_res, image_bytes, data_queue, image_queue, client_event, start_event, finish_event)))
        tk_control = asyncio.create_task(
            control_task(
                image_queue, msg_queue, process_msg_queue, client_event, start_event, finish_event,
                output_folder, capturing_folder, image_display, args.save, args.no_show)
        )
        await tk_control
        await finish_event.wait()
        tk_message.cancel()
        await tk_receive
        await tk_decode
        await tk_terminal

    except Exception as e:
        print(e)
    finally:
        print("Code finished")


if __name__ == '__main__':
    asyncio.run(main())
