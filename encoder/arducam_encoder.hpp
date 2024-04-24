/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2020, Raspberry Pi (Trading) Ltd.
 *
 * mjpeg_encoder.hpp - mjpeg video encoder.
 */

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

#include "encoder.hpp"

struct ResolutionPairs {
	ResolutionPairs(unsigned int fWidth, unsigned int fHeight, unsigned int iWidth, unsigned int iHeight) :
		  fileWidth(fWidth), fileHeight(fHeight), imageWidth(iWidth), imageHeight(iHeight) {};
	const unsigned int fileWidth;
	const unsigned int fileHeight;
	const unsigned int imageWidth;
	const unsigned int imageHeight;
};
constexpr std::map<std::string, ResolutionPairs> resolution_map_ = {
	{"LOW", ResolutionPairs(1344, 990, 1328, 990)},
	{"MEDIUM", ResolutionPairs(2032, 1080, 2024, 1080)}
};

class ArducamEncoder : public Encoder
{
public:
	ArducamEncoder(VideoOptions const *options);
	~ArducamEncoder();
	// Encode the given buffer.
	void EncodeBuffer(int fd, size_t size, void *mem, StreamInfo const &info, int64_t timestamp_us) override;

private:
	// When read from the buffer, some pixels don't contain information. This is an empirical map
	// that pairs the memory size with the actual image size.
	std::unique_ptr<ResolutionPairs> current_res_;
	// How many threads to use. Whichever thread is idle will pick up the next frame.
	static const int NUM_ENC_THREADS = 4;

	// These threads do the actual encoding.
	void encodeThread(int num);

	// Handle the output buffers in another thread so as not to block the encoders. The
	// application can take its time, after which we return this buffer to the encoder for
	// re-use.
	void outputThread();

	bool abortEncode_;
	bool abortOutput_;
	uint64_t index_;

	struct EncodeItem
	{
		void *mem;
		StreamInfo info;
		int64_t timestamp_us;
		uint64_t index;
	};

	std::queue<EncodeItem> encode_queue_;
	std::mutex encode_mutex_;
	std::condition_variable encode_cond_var_;
	std::thread encode_thread_[NUM_ENC_THREADS];
	void encodeArducam(EncodeItem &item, uint16_t *&encoded_buffer, size_t &buffer_len);

	struct OutputItem
	{
		void *mem;
		size_t bytes_used;
		int64_t timestamp_us;
		uint64_t index;
	};
	std::queue<OutputItem> output_queue_[NUM_ENC_THREADS];
	std::mutex output_mutex_;
	std::condition_variable output_cond_var_;
	std::thread output_thread_;
};
