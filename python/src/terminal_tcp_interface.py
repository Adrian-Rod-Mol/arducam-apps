import sys
from typing import List
import select
import queue
import socket
import threading


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


def terminal_input(user_action_map: List[UserAction], process_msg_queue: queue.Queue):
    """Waits until an action is asked via terminal to the script. Check if the action is valid and put it in the process
    action queue to be processed and sent to the client.

    input:
        * user_action_map: list of available actions.
        * process_msg_queue: queue with the actions to be processed
        * stop: event that tells if the loop should be stopped or not

    return:
        None"""
    stop_asking = False
    user_action_menu = ""
    for idx, action in enumerate(user_action_map):
        user_action_menu += f' {idx + 1}: {action.description}\n'
    print("==================")
    print(user_action_menu)
    print("-->", end=" ", flush=True)
    while not stop_asking:
        i, _, _ = select.select([sys.stdin], [], [], TERMINAL_TIMEOUT)
        if i:
            try:
                index = int(sys.stdin.readline().strip()) - 1
            except ValueError:
                index = 999
            if index < len(user_action_map):
                if user_action_map[index].require_value:
                    value = int(input("--> Insert the value to be set:\n--> "))
                    if user_action_map[index].check_value(value):
                        process_msg_queue.put(user_action_map[index].get_message(value))
                    else:
                        print_terminal(1, "invalid value. Value should be between " +
                                       user_action_map[index].get_max_min())
                else:
                    process_msg_queue.put(user_action_map[index].get_message())

                if user_action_map[index].message == "CLOSE":
                    stop_asking = True

            else:
                print_terminal(1, "invalid index, select one of the following:")
                print("==================")
                print(user_action_menu)
            print("-->", end=" ", flush=True)
    print_terminal(0, "Message receiving thread finished correctly.")
    return


def send_message(msg_conn: socket.socket, msg_queue: queue.Queue, stop: threading.Event):
    """Send the messages contained in the msg_queue to the camera to control its behaviour.

    input:
        * msg_conn: client socket in which the camera is listening for message.
        * msg_queue: messages received from the terminal already processed.
        * stop: event that tells if the script should stop sending messages.

    return:
        None"""
    try:
        while not stop.is_set():
            if not msg_queue.empty():
                msg = msg_queue.get()
                encoded_message = msg.encode('utf-8')
                msg_conn.send(encoded_message)
    except Exception as e:
        print_terminal(1, "Something happened while trying to send a message to client")
        print(e)

    finally:
        msg_conn.close()
        print_terminal(0, "Message sender thread finished correctly")
        return
