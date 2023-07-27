// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#pragma once

#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <set>

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/encodings/spherical_harmonics.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>
#include <json/json.hpp>

using namespace tcnn;
using precision_t = network_precision_t;

struct TINY_MLP {

	TINY_MLP(std::string configPath, int inputCh, int outputCh);

	void inference(float* inputDevice, float* outputDevice, int size);
	void loadWeights(std::string weightPath);
	void reset();

	nlohmann::json config;
	nlohmann::json encoding_opts, loss_opts, optimizer_opts, network_opts;

	std::shared_ptr<NetworkWithInputEncoding<precision_t>> network;
	std::shared_ptr<Optimizer<precision_t>> optimizer;
	std::shared_ptr<Loss<precision_t>> loss;
	std::shared_ptr<Trainer<float, precision_t, precision_t>> trainer;

	int inputCh, outputCh;
};

__global__
void getBatch(float* batchInput, float* batchOutput,
	float* inputBuffer, float* outputBuffer, int* indices,
	int dataInputCh, int dataOutputCh,
	int mlpInputCh, int mlpOutputCh);

struct DataLoader {
	
	DataLoader(std::string dataPath);
	void shuffleData();
	void populateBatch();

	float* batchInput;
	float* batchOutput;

	float* inputBuffer;
	float* outputBuffer;
	int* dataIndices;

	float angularx, angulary;
	float numPoints;
	size_t datasetSize;

	int dataInputCh, dataOutputCh;
	int mlpInputCh, mlpOutputCh;
};