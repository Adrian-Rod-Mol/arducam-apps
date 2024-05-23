/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2020, Raspberry Pi (Trading) Ltd.
 * Modified by Adrián Rodríguez-Molina
 * arducam_raw.cpp - IMX477 stereo system raw video record app.
 */
#include <chrono>
#include <iostream>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>

#include "core/rpicam_encoder.hpp"
#include "encoder/arducam_encoder.hpp"
#include "output/output.hpp"

using namespace std::placeholders;

class ArducamRaw : public RPiCamEncoder
{
public:
	ArducamRaw() : RPiCamEncoder() {}

protected:
	// Force the use of "null" encoder.
	void createEncoder() { encoder_ = std::unique_ptr<Encoder>(new ArducamEncoder(GetOptions())); }
};

struct Message {
	Message() = default;
	Message(std::string raw_message) {
		auto position = raw_message.find(" = ");
		if (position != std::string::npos) {
			this->key = raw_message.substr(0, position);
			this->value = std::stoi(raw_message.substr(position + 3));
		} else {
			this->key = raw_message;
			this->value = 0;
		}
	}
	std::string key;
	unsigned int value;
};
static int connect_to_message_server(VideoOptions *options)
{
	char protocol[4];
	int start, end, a, b, c, d, port;
	if (sscanf(options->message_ip.c_str(), "%3s://%n%d.%d.%d.%d%n:%d", protocol, &start, &a, &b, &c, &d, &end, &port) != 6)
		throw std::runtime_error("bad network address " + options->message_ip);
	std::string address = options->message_ip.substr(start, end - start);
	if (strcmp(protocol, "tcp") == 0) {
		struct sockaddr_in msgServerAddr;
		msgServerAddr.sin_family = AF_INET;
		msgServerAddr.sin_port = htons(port);
		msgServerAddr.sin_addr.s_addr = inet_addr(address.c_str());

		int msg_socket = socket(AF_INET, SOCK_STREAM, 0);
		if (msg_socket  == -1) {
			throw std::runtime_error("message socket creation failed");
		}

		if (connect(msg_socket , (struct sockaddr*)&msgServerAddr, sizeof(msgServerAddr)) == -1) {
			close(msg_socket);
			throw std::runtime_error("connection to server failed");
		}
		return msg_socket;
	} else
		throw std::runtime_error("unrecognised network protocol " + options->message_ip);
}
static void receive_messages(int msg_socket, std::mutex& mtx, std::condition_variable& cv, std::queue<Message> &msg_queue, std::atomic<bool>& keep_process)
{
	while (keep_process) {
		char buffer[1024];
		int bytesReceived = recv(msg_socket, buffer, sizeof(buffer), 0);
		// This condition prevents the code to put in a queue empty messages sent by the server.
		if (bytesReceived >= 3) {
			buffer[bytesReceived] = '\0';
			auto message = Message(std::string(buffer));
			{
				std::lock_guard<std::mutex> lock(mtx);
				msg_queue.push(message);
			}
			cv.notify_one();
		} else {
			std::cerr << "Connection closed by server or error occurred\n";
			break;
		}
	}
}

static void capturing_control(VideoOptions *options, std::queue<Message> &msg_queue, std::mutex& msg_mtx, std::condition_variable& msg_cv, std::mutex& img_mtx, std::condition_variable& img_cv, bool& take_images, std::atomic<bool>& keep_process)
{
	while (keep_process) {
		{
			std::unique_lock<std::mutex> lock(msg_mtx);
			msg_cv.wait(lock, [&msg_queue]{ return !msg_queue.empty(); });
		}
		Message msg;
		{
			std::lock_guard <std::mutex> lock(msg_mtx);
			msg = msg_queue.front();
			msg_queue.pop();
		}
		LOG(2, "Received message: " << msg.key);
		if (msg.key == "CLOSE") {
			keep_process = false;
			{
				std::lock_guard<std::mutex> lock(img_mtx);
				take_images = false;
			}
			img_cv.notify_one();
			LOG(1, "Server closed connection.");
		} else if (msg.key == "START") {
			{
				std::lock_guard<std::mutex> lock(img_mtx);
				take_images = true;
			}
			img_cv.notify_one();
		} else if (msg.key == "STOP") {
			{
				std::lock_guard<std::mutex> lock(img_mtx);
				take_images = false;
			}
		} else if (msg.key == "EXPOSURE") {
			if (!take_images) {
				{
					std::lock_guard<std::mutex> lock(img_mtx);
					auto shutter_string = std::to_string(msg.value) + "us";
					options->shutter.set(shutter_string);
				}
			} else {
				LOG(1, "Can't change camera parameters while capturing.");
			}
		} else {
			LOG(1, "Unrecognized message: " << msg.key);
		}
	}
}
//// The main even loop for the application.
static void event_loop(ArducamRaw &app, std::mutex& img_mtx, std::condition_variable& img_cv, bool& take_images, std::atomic<bool>& keep_process)
{
	// The first time that the event loop is called, it should open the camera and configure the video streaming.
	// When the capturing process is restarted, it must avoid calling these two functions
	bool first = true;
	while(keep_process) {
		{
			std::unique_lock<std::mutex> lock(img_mtx);
			img_cv.wait(lock, [&take_images, &keep_process] { return take_images | !keep_process; });
		}
		if (!keep_process) return;
		VideoOptions const *options = app.GetOptions();
		if (first) {
			app.OpenCamera();
			app.ConfigureVideo(ArducamRaw::FLAG_VIDEO_RAW);
			first = false;
		}
		std::unique_ptr<Output> output = std::unique_ptr<Output>(Output::Create(options));
		app.SetEncodeOutputReadyCallback(std::bind(&Output::OutputReady, output.get(), _1, _2, _3, _4));
		app.SetMetadataReadyCallback(std::bind(&Output::MetadataReady, output.get(), _1));
		app.StartEncoder();
		app.StartCamera();
		auto start_time = std::chrono::high_resolution_clock::now();
		for (unsigned int count = 0; ; count++)
		{
			ArducamRaw::Msg msg = app.Wait();

			if (msg.type == RPiCamApp::MsgType::Timeout)
			{
				LOG_ERROR("ERROR: Device timeout detected, attempting a restart!!!");
				app.StopCamera();
				app.StartCamera();
				continue;
			}
			if (msg.type != ArducamRaw::MsgType::RequestComplete)
				throw std::runtime_error("unrecognised message!");
			if (count == 0)
			{
				libcamera::StreamConfiguration const &cfg = app.RawStream()->configuration();
				LOG(1, "Raw stream: " << cfg.size.width << "x" << cfg.size.height << " stride " << cfg.stride << " format "
									  << cfg.pixelFormat.toString());
			}

			LOG(2, "Viewfinder frame " << count);
			auto now = std::chrono::high_resolution_clock::now();
			if ((options->timeout && (now - start_time) > options->timeout.value) || !take_images)
			{
				app.StopCamera();
				app.StopEncoder();
				if (options->timeout && (now - start_time) > options->timeout.value) {
					return;
				}
				break;
			}

			app.EncodeBuffer(std::get<CompletedRequestPtr>(msg.payload), app.RawStream());
		}
	}
}

int main(int argc, char *argv[])
{
	try
	{
		std::mutex msg_mtx;
		std::condition_variable msg_cv;
		std::mutex img_mtx;
		std::condition_variable img_cv;
		std::queue<Message> msg_queue;

		bool take_images = false;
		std::atomic<bool> keep_process = true;

		ArducamRaw app;
		VideoOptions *options = app.GetOptions();
		if (options->Parse(argc, argv))
		{
			// Disable any codec (h.264/libav) based operations.
			options->codec = "yuv420";
			options->denoise = "cdn_off";
			options->nopreview = true;
			if (options->verbose >= 2)
			{
				options->Print();
			}
			if (options->message_ip) {
				auto msg_socket = connect_to_message_server(options);
				std::thread receiver_thread(receive_messages, msg_socket, std::ref(msg_mtx), std::ref(msg_cv), std::ref(msg_queue), std::ref(keep_process));
				{
					std::unique_lock<std::mutex> lock(msg_mtx);
					msg_cv.wait(lock, [&msg_queue]{ return !msg_queue.empty(); });
				}
				{
					std::lock_guard <std::mutex> lock(msg_mtx);
					auto resolution_message = msg_queue.front();
					msg_queue.pop();
					options->resolution_key = resolution_message.key;
				}
				std::thread control_thread(capturing_control, options, std::ref(msg_queue), std::ref(msg_mtx), std::ref(msg_cv), std::ref(img_mtx), std::ref(img_cv), std::ref(take_images), std::ref(keep_process));
			} else {
				{
					std::lock_guard<std::mutex> lock(img_mtx);
					take_images = true;
				}
				img_cv.notify_one();
			}
			
			event_loop(app, img_mtx, img_cv, take_images, keep_process);

			if (options->message_ip) {
				control_thread.join();
				receiver_thread.join();
			}
		}
	}
	catch (std::exception const &e)
	{
		LOG_ERROR("ERROR: *** " << e.what() << " ***");
		return -1;
	}
	return 0;
}
