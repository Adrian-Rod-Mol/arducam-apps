# rpicam-apps
This is a small suite of libcamera-based applications to drive the cameras on a Raspberry Pi platform.

>[!WARNING]
>These applications and libraries have been renamed from `libcamera-*` to `rpicam-*`. Symbolic links are installed to allow users to keep using the old application names, but these will be deprecated soon. Users are encouraged to adopt the new application and library names as soon as possible.

Build
-----
For usage and build instructions, see the official Raspberry Pi documentation pages [here.](https://www.raspberrypi.com/documentation/computers/camera_software.html#building-libcamera-and-rpicam-apps)

License
-------

The source code is made available under the simplified [BSD 2-Clause license](https://spdx.org/licenses/BSD-2-Clause.html).

Status
------

[![ToT libcamera build/run test](https://github.com/raspberrypi/rpicam-apps/actions/workflows/rpicam-test.yml/badge.svg)](https://github.com/raspberrypi/rpicam-apps/actions/workflows/rpicam-test.yml)

Use Examples
------

* Save Images on a File
```shell
./build/apps/arducam-raw -t 2000 -o /home/armolina/outputs/residual_test/name%05d.raw --mode 2028:1080:10:U --shutter 1ms  --gain 1 --awbgains 1,1 --resolution MEDIUM --nopreview --verbose 2 --segment 1
```
* Send Images to a TCP server
```shell
./build/apps/arducam-raw -t 2000 -o tcp://10.42.0.1:32233 --mode 2028:1080:10:U --shutter 1ms  --gain 1 --awbgains 1,1 --resolution MEDIUM --nopreview --verbose 2
```