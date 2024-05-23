# rpicam-apps
This is a small suite of libcamera-based applications to drive the cameras on a Raspberry Pi platform.

>[!WARNING]
>These applications and libraries have been renamed from `libcamera-*` to `rpicam-*`. Symbolic links are installed to allow users to keep using the old application names, but these will be deprecated soon. Users are encouraged to adopt the new application and library names as soon as possible.

Build
-----
### To build the c++ code to capture in a Raspberry PI
> The standard used is the c++17 as it is the one used by the Raspberry Pi base apps.
These instructions has been obtained from the official Raspberry Pi documentation pages [here.](https://www.raspberrypi.com/documentation/computers/camera_software.html#building-libcamera-and-rpicam-apps)
1. Install the dependencies.
```shell
sudo apt install -y libcamera-dev libepoxy-dev libjpeg-dev libtiff5-dev libpng-dev

sudo apt install -y qtbase5-dev libqt5core5a libqt5gui5 libqt5widgets5

sudo apt install libavcodec-dev libavdevice-dev libavformat-dev libswresample-dev

sudo apt install -y cmake libboost-program-options-dev libdrm-dev libexif-dev

sudo apt install -y meson ninja-build
```
2. Configure the build using the following meson command.
```shell
meson setup build -Denable_libav=enabled -Denable_drm=enabled -Denable_egl=enabled -Denable_qt=enabled -Denable_opencv=disabled -Denable_tflite=disabled
```
3. Build the program.
```shell
meson compile -C build
```

### To setup the python control and visualization code.
> This code has been tested in python 3.8 and 3.10.
1. Install, if necessary, the Python virtual environment module.
```shell
sudo apt-get install python3-venv
```
2. Create a virtual enviroment and install the required libraries.
```shell
cd python
python3 -m venv venv --prompt arducam
source venv/bin/activate
pip install -r requirements.txt
```

License
-------

The source code is made available under the simplified [BSD 2-Clause license](https://spdx.org/licenses/BSD-2-Clause.html).

Status
------

[![ToT libcamera build/run test](https://github.com/raspberrypi/rpicam-apps/actions/workflows/rpicam-test.yml/badge.svg)](https://github.com/raspberrypi/rpicam-apps/actions/workflows/rpicam-test.yml)

Use Examples
------
### Capturing Code
* Save Images on a File
```shell
./build/apps/arducam-raw -t 2000 -o $HOME/outputs/residual_test/name%05d.raw --mode 2028:1080:10:U --shutter 1ms  --gain 1 --awbgains 1,1 --resolution MEDIUM --nopreview --verbose 2 --segment 1
```
* Send Images to a TCP server
```shell
./build/apps/arducam-raw -t 2000 -o tcp://10.42.0.1:32233 --mode 2028:1080:10:U --shutter 1ms  --gain 1 --awbgains 1,1 --resolution MEDIUM --nopreview --verbose 2
```

### Visualization code
* Start a server in a local repository to visualize the images.
```shell
python3 arducam_tcp.py --ip 127.0.0.1 --resolution MEDIUM
```
* Save the images on files whithout showing them.
```shell
python3 arducam_tcp.py --ip 127.0.0.1 --resolution MEDIUM --output_folder outputs --no_show
```