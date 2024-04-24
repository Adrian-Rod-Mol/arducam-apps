/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2020, Raspberry Pi (Trading) Ltd.
 * Modified by Adrián Rodríguez-Molina
 * arducam_raw.cpp - IMX477 stereo system raw video record app.
 */
#include <chrono>
#include <iostream>

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

// The main even loop for the application.

static void event_loop(ArducamRaw &app)
{
	LOG(1, "Logging works in the event loop");
	VideoOptions const *options = app.GetOptions();
	std::cerr << "Get options not working!" << std::endl;
	std::unique_ptr<Output> output = std::unique_ptr<Output>(Output::Create(options));
	LOG(1, "Pointer to output initialized correctly");
	app.SetEncodeOutputReadyCallback(std::bind(&Output::OutputReady, output.get(), _1, _2, _3, _4));
	app.SetMetadataReadyCallback(std::bind(&Output::MetadataReady, output.get(), _1));
	LOG(1, "Before opening the camera");
	app.OpenCamera();
	app.ConfigureVideo(ArducamRaw::FLAG_VIDEO_RAW);
	app.StartEncoder();
	app.StartCamera();
	LOG(1, "Camera and video started correctly");
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
			return;
		}

		app.EncodeBuffer(std::get<CompletedRequestPtr>(msg.payload), app.RawStream());
	}
}

int main(int argc, char *argv[])
{
	try
	{
		ArducamRaw app;

		VideoOptions *options = app.GetOptions();
		LOG(1, "Video options get initialized correctly");
		if (options->Parse(argc, argv))
		{
			// Disable any codec (h.264/libav) based operations.
			options->codec = "yuv420";
			options->denoise = "cdn_off";
			options->nopreview = true;
			if (options->verbose >= 2)
				options->Print();
			LOG(1, "The program enters the event loop");
			event_loop(app);
		}
	}
	catch (std::exception const &e)
	{
		LOG_ERROR("ERROR: *** " << e.what() << " ***");
		return -1;
	}
	return 0;
}
