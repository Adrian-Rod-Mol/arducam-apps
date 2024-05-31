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
    async_send_message
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


async def receive_task(server_ip: str,
                       server_port: int,
                       img_bytes: int,
                       data_queue: asyncio.Queue,
                       client_connected: threading.Event,
                       start: asyncio.Event,
                       finish: asyncio.Event):
    img_writer = None
    try:
        while not finish.is_set():
            _, _ = await asyncio.wait([finish.wait(), start.wait()], return_when=asyncio.FIRST_COMPLETED)
            if finish.is_set():
                break
            elif start.is_set():
                img_reader, img_writer = await asyncio.open_connection(server_ip, server_port, limit=img_bytes)
                client_connected.set()
                mean_time = 0
                count = 0

                while start.is_set():
                    start_time = time.perf_counter_ns()
                    bytes_to_receive = img_bytes
                    try:
                        while bytes_to_receive > 0:
                            data = await asyncio.wait_for(img_reader.read(bytes_to_receive), LOOP_TIMEOUT)
                            if data:
                                await data_queue.put(data)
                                bytes_to_receive -= len(data)
                        mean_time += time.perf_counter_ns() - start_time
                        count += 1

                    except asyncio.TimeoutError:
                        print_terminal(1, "Waiting for the server to send the image timed out.")

                img_writer.close()
                await img_writer.wait_closed()
                img_writer = None
                if count != 0:
                    mean_time /= count
                    print_terminal(0, f"Mean time elapsed receiving {count} images:  {mean_time / 1000000} ms")

    except Exception as e:
        print(e)

    finally:
        if img_writer is not None:
            img_writer.close()
            await img_writer.wait_closed()
        client_connected.clear()
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
            _, _ = await asyncio.wait([finish.wait(), client_connected.wait()], return_when=asyncio.FIRST_COMPLETED)
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
    # Sending configuration to Raspberry
    print_terminal(0, "Waiting for connection to configure...")
    cnf_reader, cnf_writer = await asyncio.open_connection(args.ip, TCP_CONF_PORT)
    print_terminal(0, "Configuration client connected.")

    conf_message = ""
    if args.resolution == "LOW":
        conf_message = "--mode 1332:990:10:U --resolution LOW"
    elif args.resolution == "MEDIUM":
        conf_message = "--mode 2028:1520:12:U --resolution MEDIUM"
    elif args.resolution == "HIGH":
        conf_message = "--mode 4056:3040:12:U --resolution HIGH"

    cnf_writer.write(conf_message.encode('utf-8'))
    await cnf_writer.drain()

    cnf_writer.close()
    await cnf_writer.wait_closed()

    print_terminal(0, "Waiting for client connection...")
    msg_reader, msg_writer = await asyncio.open_connection(args.ip, TCP_MSG_PORT)
    print_terminal(0, "A client has connected to the message queue.")

    # Define default output or capturing folder to prevent errors and undefined behaviour
    output_folder = Path(args.output_folder)
    if not output_folder.is_dir():
        output_folder.mkdir()
    capturing_folder = output_folder

    image_display = ImageDisplay(0, current_res["width"], current_res["height"])

    _tasks = [
        async_terminal(user_action_map, process_msg_queue),
        async_send_message(msg_writer, msg_queue, finish_event),
        receive_task(args.ip, TCP_PORT, image_bytes, data_queue, client_event, start_event, finish_event),
        decode_task(current_res, image_bytes, data_queue, image_queue, client_event, start_event, finish_event),
        control_task(image_queue, msg_queue, process_msg_queue,
                     client_event, start_event, finish_event,
                     output_folder, capturing_folder, image_display, args.save, args.no_show)
    ]
    try:
        async with asyncio.TaskGroup() as tg:
            tk_terminal = tg.create_task(async_terminal(user_action_map, process_msg_queue))
            tk_message = tg.create_task(async_send_message(msg_writer, msg_queue, finish_event))
            tk_receive = tg.create_task(
                asyncio.shield(
                    receive_task(args.ip, TCP_PORT, image_bytes, data_queue, client_event, start_event, finish_event)))
            tk_decode = tg.create_task(
                asyncio.shield(
                    decode_task(
                        current_res, image_bytes, data_queue, image_queue, client_event, start_event, finish_event)))
            tk_control = tg.create_task(
                control_task(
                    image_queue, msg_queue, process_msg_queue, client_event, start_event, finish_event,
                    output_folder, capturing_folder, image_display, args.save, args.no_show)
            )

    except KeyboardInterrupt:
        print("Code finished")
    except Exception as e:
        print(e)
    finally:
        msg_writer.close()
        await msg_writer.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())
