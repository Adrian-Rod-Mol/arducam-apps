import sys
from typing import List
import asyncio

TERMINAL_TIMEOUT = 3


class Message:

    def __init__(self, _need_value: bool, _key: str, _value: int = -1):
        self.need_value = _need_value
        self.key = _key
        self.value = _value

    def encode(self, encode_type: str):
        if self.need_value:
            message = f"{self.key} = {self.value}"
        else:
            message = f"{self.key}"
        return message.encode(encode_type)


class UserAction:

    def __init__(self, description: str, message: str, require_value: bool, _max: int = -1, _min: int = -1):
        self.description = description
        self.message = message
        self.require_value = require_value
        self.max = _max
        self.min = _min

    def check_value(self, value: int):
        return self.max >= value >= self.min

    def get_max_min(self):
        return f"MAX: {self.max}, MIN: {self.min}"

    def get_message(self, value: int = -1) -> Message:
        return Message(self.require_value, self.message, value)


def print_terminal(severity: int, s: str):
    if severity == 0:
        print(s)
    elif severity == 1:
        print("<--> ERROR: ", s)
    print("--> ", end="", flush=True)


async def ainput(string: str) -> str:
    await asyncio.get_event_loop().run_in_executor(
            None, lambda s=string: sys.stdout.write(s))
    await asyncio.sleep(0.2)
    return await asyncio.get_event_loop().run_in_executor(
            None, sys.stdin.readline)


async def read_int(string: str) -> int:
    str_index = await ainput(string)
    try:
        index = int(str_index) - 1
    except ValueError:
        index = -1
    return index


async def async_terminal(user_action_map: List[UserAction], process_msg_queue: asyncio.Queue):
    """Waits until an action is asked via terminal to the script. Check if the action is valid and put it in the process
    action queue to be processed and sent to the client.

    input:
        * user_action_map: list of available actions.
        * process_msg_queue: queue with the actions to be processed

    return:
        None"""
    stop_asking = False
    user_action_menu = ""
    for idx, action in enumerate(user_action_map):
        user_action_menu += f' {idx + 1}: {action.description}\n'
    print("==================")
    print(user_action_menu)
    while not stop_asking:
        index = await read_int("-->")
        if index < len(user_action_map):
            if user_action_map[index].require_value:
                value = await read_int("--> Insert the value to be set:\n--> ")
                if user_action_map[index].check_value(value):
                    await process_msg_queue.put(user_action_map[index].get_message(value))
                else:
                    print_terminal(1, "invalid value. Value should be between " +
                                   user_action_map[index].get_max_min())
            else:
                await process_msg_queue.put(user_action_map[index].get_message())

            if user_action_map[index].message == "CLOSE":
                stop_asking = True

        else:
            print_terminal(1, "invalid index, select one of the following:")
            print("==================")
            print(user_action_menu)
    print_terminal(0, "Message receiving thread finished correctly.")


async def async_send_message(msg_writer: asyncio.StreamWriter, msg_queue: asyncio.Queue, stop: asyncio.Event):
    """Send the messages contained in the msg_queue to the camera to control its behaviour.

    input:
        * msg_conn: asyncio writer that sends message to the camera via TCP
        * msg_queue: messages received from the terminal already processed.
        * stop: event that tells if the script should stop sending messages.

    return:
        None"""
    try:
        while not stop.is_set():
            msg = await msg_queue.get()
            encoded_message = msg.encode('utf-8')
            msg_writer.write(encoded_message)
            await msg_writer.drain()

    except Exception as e:
        print_terminal(1, "Something happened while trying to send a message to client")
        raise e

    finally:
        msg_writer.close()
        await msg_writer.wait_closed()
        print_terminal(0, "Message sender task finished correctly")
