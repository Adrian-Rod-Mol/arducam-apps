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

static void capturing_control(VideoOptions *options, std::mutex& start_mtx, std::condition_variable& start_cv, std::mutex& nxt_mtx, std::condition_variable& nxt_cv, bool& next, bool& start, std::atomic<bool>& keep_capturing)
{
	std::this_thread::sleep_for(std::chrono::milliseconds(3000));
	unsigned int base_exposure = 5000;
	for (int i = i; i <= 5; ++i) {
		std::cout << "Sending start capturing signal\n.";
		{
			std::lock_guard<std::mutex> lock(start_mtx);
			unsigned int current_exposure = base_exposure*i;
			auto shutter_string = std::to_string(current_exposure) + "us";
			options->shutter.set(shutter_string);
			start = true;
		}
		start_cv.notify_one();
		{
			std::unique_lock<std::mutex> lock(nxt_mtx);
			nxt_cv.wait(lock, [&next] { return next; });
			next = false;
		}
	}
	keep_capturing = false;
	start_cv.notify_one();
}
//// The main even loop for the application.
static void event_loop(ArducamRaw &app, std::mutex& start_mtx, std::condition_variable& start_cv, std::mutex& nxt_mtx, std::condition_variable& nxt_cv, bool& next, bool& start, std::atomic<bool>& keep_capturing)
{
	// The first time that the event loop is called, it should open the camera and configure the video streaming.
	// When the capturing process is restarted, it must avoid calling these two functions
	bool first = true;
	while(keep_capturing) {
		{
			std::unique_lock<std::mutex> lock(start_mtx);
			start_cv.wait(lock, [&start, &keep_capturing] { return start | !keep_capturing; });
			start = false;
		}
		if (!keep_capturing) return;
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
			if (options->timeout && (now - start_time) > options->timeout.value)
			{
				app.StopCamera();
				app.StopEncoder();
				break;
			}

			app.EncodeBuffer(std::get<CompletedRequestPtr>(msg.payload), app.RawStream());
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(2000));
		{
			std::lock_guard<std::mutex> lock(nxt_mtx);
			next = true;
		}
		nxt_cv.notify_one();
	}
}

int main(int argc, char *argv[])
{
	try
	{
		std::mutex start_mtx;
		std::condition_variable start_cv;
		std::mutex nxt_mtx;
		std::condition_variable nxt_cv;
		bool next;
		bool start;
		std::atomic<bool> keep_capturing;

		ArducamRaw app;

		VideoOptions *options = app.GetOptions();
		if (options->Parse(argc, argv))
		{
			// Disable any codec (h.264/libav) based operations.
			options->codec = "yuv420";
			options->denoise = "cdn_off";
			options->nopreview = true;
			if (options->verbose >= 2) {
				options->Print();

			}
			std::thread control_thread(capturing_control, options, std::ref(start_mtx), std::ref(start_cv), std::ref(nxt_mtx), std::ref(nxt_cv), std::ref(next), std::ref(start), std::ref(keep_capturing));
			event_loop(app, start_mtx, start_cv, nxt_mtx, nxt_cv, next, start, keep_capturing);
			control_thread.join();
		}
	}
	catch (std::exception const &e)
	{
		LOG_ERROR("ERROR: *** " << e.what() << " ***");
		return -1;
	}
	return 0;
}
