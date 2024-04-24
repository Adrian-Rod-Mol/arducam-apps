/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2020, Raspberry Pi (Trading) Ltd.
 * Modified by Adrián Rodríguez-Molina. University of Las Palmas de Gran Canaria. armolina@iuma.ulpgc.es
 * arducam_encoder.cpp - Video encoder for Arducam IMX477 stereo system.
 */

#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

#include <stdlib.h>
#include <malloc.h>

#include "arducam_encoder.hpp"

// When read from the buffer, some pixels don't contain information. This is an empirical map
//  that pairs the memory size with the actual image size
static const std::map<std::string, ResolutionPairs> resolution_map_ = {
	{"LOW", ResolutionPairs(1344, 990, 1328, 990)},
	{"MEDIUM", ResolutionPairs(2032, 1080, 2024, 1080)}
};

ArducamEncoder::ArducamEncoder(const VideoOptions *options)
	: Encoder(options), abortEncode_(false), abortOutput_(false), index_(0)
{
	if (auto res_map = resolution_map_.find(options->resolution_key); res_map != resolution_map_.end()) {
		current_res_ = std::make_unique<ResolutionPairs>(res_map->second);
	}
	output_thread_ = std::thread(&ArducamEncoder::outputThread, this);
	for (int i = 0; i < NUM_ENC_THREADS; i++)
		encode_thread_[i] = std::thread(std::bind(&ArducamEncoder::encodeThread, this, i));
	LOG(2, "Opened ArducamEncoder");
}

ArducamEncoder::~ArducamEncoder()
{
	abortEncode_ = true;
	for (int i = 0; i < NUM_ENC_THREADS; i++)
		encode_thread_[i].join();
	abortOutput_ = true;
	output_thread_.join();
	LOG(2, "ArducamEncoder closed");
}

void ArducamEncoder::EncodeBuffer(int fd, size_t size, void *mem, StreamInfo const &info, int64_t timestamp_us)
{
	std::lock_guard<std::mutex> lock(encode_mutex_);
	EncodeItem item = { mem, info, timestamp_us, index_++ };
	encode_queue_.push(item);
	encode_cond_var_.notify_all();
}

void ArducamEncoder::encodeArducam(ArducamEncoder::EncodeItem &item, uint16_t *&encoded_buffer, size_t &buffer_len)
{
	const auto band_width = current_res_->imageWidth/2;
	const auto band_height = current_res_->imageHeight/2;

	auto img_ptr = reinterpret_cast<uint16_t*>(item.mem);
	auto input_image = std::vector<uint16_t>(img_ptr, img_ptr + current_res_->fileWidth*current_res_->fileHeight);

	auto input_band_1_it = input_image.begin();
	auto input_band_2_it = input_image.begin() + band_width;
	auto input_band_3_it = input_image.begin() + band_height*current_res_->fileWidth;
	auto input_band_4_it = input_image.begin() + band_height*current_res_->fileWidth + band_width;
	// The memory allocation and destruction is probably needed
	encoded_buffer = reinterpret_cast<uint16_t*>(malloc(current_res_->imageWidth*current_res_->imageHeight*sizeof(uint16_t)));
	auto encoded_image = std::vector<uint16_t>(current_res_->imageWidth*current_res_->imageHeight);

	auto out_band_1_it = encoded_image.begin();
	auto out_band_2_it = encoded_image.begin() + band_width*band_height;
	auto out_band_3_it = encoded_image.begin() + 2*band_width*band_height;
	auto out_band_4_it = encoded_image.begin() + 3*band_width*band_height;

	auto copyBandColumn = [](std::vector<unsigned short>::iterator &input_begin,
							 std::vector<unsigned short>::iterator input_end,
							 std::vector<unsigned short>::iterator &output_begin,
							 int band_width,
							 int file_width) {
		std::copy(input_begin, input_end, output_begin);
		std::advance(input_begin, file_width);
		std::advance(output_begin, band_width);
	};

	for (unsigned int i = 0; i < band_height; ++i) {
		copyBandColumn(input_band_1_it, (input_band_1_it + band_width), out_band_1_it, band_width, current_res_->fileWidth);
		copyBandColumn(input_band_2_it, (input_band_2_it + band_width), out_band_2_it, band_width, current_res_->fileWidth);
		copyBandColumn(input_band_3_it, (input_band_3_it + band_width), out_band_3_it, band_width, current_res_->fileWidth);
		copyBandColumn(input_band_4_it, (input_band_4_it + band_width), out_band_4_it, band_width, current_res_->fileWidth);
	}
	std::copy(encoded_image.begin(), encoded_image.end(), encoded_buffer);
	buffer_len = current_res_->imageWidth*current_res_->imageHeight*2;
}

void ArducamEncoder::encodeThread(int num)
{
	std::chrono::duration<double> encode_time(0);
	uint32_t frames = 0;

	EncodeItem encode_item;
	while (true)
	{
		{
			std::unique_lock<std::mutex> lock(encode_mutex_);
			while (true)
			{
				using namespace std::chrono_literals;
				if (abortEncode_ && encode_queue_.empty())
				{
					if (frames)
						LOG(2, "Encode " << frames << " frames, average time " << encode_time.count() * 1000 / frames
										 << "ms");
					// this is surely the place where the memory should be liberated
					return;
				}
				if (!encode_queue_.empty())
				{
					encode_item = encode_queue_.front();
					encode_queue_.pop();
					break;
				}
				else
					encode_cond_var_.wait_for(lock, 200ms);
			}
		}

		// Encode the buffer.
		uint16_t *encoded_buffer = nullptr;
		size_t buffer_len = 0;
		auto start_time = std::chrono::high_resolution_clock::now();
		encodeArducam(encode_item, encoded_buffer, buffer_len);
		encode_time += (std::chrono::high_resolution_clock::now() - start_time);
		frames++;
		// Don't return buffers until the output thread as that's where they're
		// in order again.

		// We push this encoded buffer to another thread so that our
		// application can take its time with the data without blocking the
		// encode process.
		OutputItem output_item = { encoded_buffer, buffer_len, encode_item.timestamp_us, encode_item.index };
		std::lock_guard<std::mutex> lock(output_mutex_);
		output_queue_[num].push(output_item);
		output_cond_var_.notify_one();
	}
}

void ArducamEncoder::outputThread()
{
	OutputItem item;
	uint64_t index = 0;
	while (true)
	{
		{
			std::unique_lock<std::mutex> lock(output_mutex_);
			while (true)
			{
				using namespace std::chrono_literals;
				// We look for the thread that's completed the frame we want next.
				// If we don't find it, we wait.
				//
				// Must also check for an abort signal, and if set, all queues must
				// be empty. This is done first to ensure all frame callbacks have
				// had a chance to run.
				bool abort = abortOutput_;
				for (auto &q : output_queue_)
				{
					if (abort && !q.empty())
						abort = false;

					if (!q.empty() && q.front().index == index)
					{
						item = q.front();
						q.pop();
						goto got_item;
					}
				}
				if (abort)
					return;

				output_cond_var_.wait_for(lock, 200ms);
			}
		}
	got_item:
		input_done_callback_(nullptr);
		std::cout << reinterpret_cast<uint16_t*>(item.mem)[0] << "\n";
		output_ready_callback_(item.mem, item.bytes_used, item.timestamp_us, true);
		free(item.mem);
		index++;
	}
}
