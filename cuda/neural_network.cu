// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#include "neural_network.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

void TINY_MLP::reset()
{
    std::vector<precision_t> zeros(this->trainer->n_params(), 0.f);
    this->trainer->set_params(zeros.data(), this->trainer->n_params(), false);
}

void TINY_MLP::loadWeights(std::string weightPath)
{
    nlohmann::json weights;

    std::ifstream ip(weightPath);
    ip >> weights;
    ip.close();

    this->trainer->deserialize(weights);
}

TINY_MLP::TINY_MLP(std::string configPath, int inputCh, int outputCh)
{
    this->inputCh = inputCh;
    this->outputCh = outputCh;

    std::ifstream ip(configPath);
    ip >> this->config;
    ip.close();

    this->encoding_opts = this->config.value("encoding", nlohmann::json::object());
    this->loss_opts = this->config.value("loss", nlohmann::json::object());
    this->optimizer_opts = this->config.value("optimizer", nlohmann::json::object());
    this->network_opts = this->config.value("network", nlohmann::json::object());

    // Some things about network (eg. n_input, n_output) also hard-coded
    this->loss.reset(create_loss<precision_t>(this->loss_opts));
    this->optimizer.reset(create_optimizer<precision_t>(this->optimizer_opts));
    this->network.reset(new NetworkWithInputEncoding<precision_t>(inputCh, outputCh, this->encoding_opts, this->network_opts));

    this->trainer.reset(new Trainer<float, precision_t, precision_t>(this->network, 
        this->optimizer,
        this->loss));
}

void TINY_MLP::inference(float* inputDevice, float* outputDevice, int size)
{
    GPUMatrix<float> inputMatrix(inputDevice, this->inputCh, size);
    GPUMatrix<float> outputMatrix(outputDevice, this->outputCh, size);

    this->network->inference(inputMatrix, outputMatrix);
}

// ====================================================
// Data Loader
// ====================================================

__global__
void getBatch(float* batchInput, float* batchOutput,
    float* inputBuffer, float* outputBuffer, int* indices,
    int dataInputCh, int dataOutputCh,
    int mlpInputCh, int mlpOutputCh)
{
    int dataIdx = indices[blockDim.x * blockIdx.x + threadIdx.x];

    // Input
    batchInput[blockDim.x * blockIdx.x * mlpInputCh + threadIdx.x * mlpInputCh + 0] = inputBuffer[dataIdx * dataInputCh + 0];
    batchInput[blockDim.x * blockIdx.x * mlpInputCh + threadIdx.x * mlpInputCh + 1] = inputBuffer[dataIdx * dataInputCh + 1];
    batchInput[blockDim.x * blockIdx.x * mlpInputCh + threadIdx.x * mlpInputCh + 2] = inputBuffer[dataIdx * dataInputCh + 2];
    
    batchInput[blockDim.x * blockIdx.x * mlpInputCh + threadIdx.x * mlpInputCh + 3] = inputBuffer[dataIdx * dataInputCh + 3];
    batchInput[blockDim.x * blockIdx.x * mlpInputCh + threadIdx.x * mlpInputCh + 4] = inputBuffer[dataIdx * dataInputCh + 4];
    batchInput[blockDim.x * blockIdx.x * mlpInputCh + threadIdx.x * mlpInputCh + 5] = inputBuffer[dataIdx * dataInputCh + 5];
    
    // Normal
    // batchInput[blockDim.x * blockIdx.x * mlpInputCh + threadIdx.x * mlpInputCh + 6] = inputBuffer[dataIdx * dataInputCh + 6];
    // batchInput[blockDim.x * blockIdx.x * mlpInputCh + threadIdx.x * mlpInputCh + 7] = inputBuffer[dataIdx * dataInputCh + 7];
    // batchInput[blockDim.x * blockIdx.x * mlpInputCh + threadIdx.x * mlpInputCh + 8] = inputBuffer[dataIdx * dataInputCh + 8];

    // Tangent
    batchInput[blockDim.x * blockIdx.x * mlpInputCh + threadIdx.x * mlpInputCh + 6] = inputBuffer[dataIdx * dataInputCh + 9];
    batchInput[blockDim.x * blockIdx.x * mlpInputCh + threadIdx.x * mlpInputCh + 7] = inputBuffer[dataIdx * dataInputCh + 10];
    batchInput[blockDim.x * blockIdx.x * mlpInputCh + threadIdx.x * mlpInputCh + 8] = inputBuffer[dataIdx * dataInputCh + 11];
    
    // Output
    batchOutput[blockDim.x * blockIdx.x * mlpOutputCh + threadIdx.x * mlpOutputCh + 0] = outputBuffer[dataIdx * dataOutputCh + 0];
    batchOutput[blockDim.x * blockIdx.x * mlpOutputCh + threadIdx.x * mlpOutputCh + 1] = outputBuffer[dataIdx * dataOutputCh + 1];
    batchOutput[blockDim.x * blockIdx.x * mlpOutputCh + threadIdx.x * mlpOutputCh + 2] = outputBuffer[dataIdx * dataOutputCh + 2];
}

void DataLoader::populateBatch()
{
    getBatch<<<this->datasetSize / 128, 128>>>(this->batchInput, this->batchOutput,
        this->inputBuffer, this->outputBuffer, this->dataIndices, 
        this->dataInputCh, this->dataOutputCh,
        this->mlpInputCh, this->mlpOutputCh);
}

void DataLoader::shuffleData()
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    thrust::device_ptr<int> thrustPtr = thrust::device_pointer_cast(this->dataIndices);
    thrust::shuffle(thrustPtr, thrustPtr + this->datasetSize, thrust::default_random_engine(seed));
}

DataLoader::DataLoader(std::string dataPath)
{
    std::string metadataFile = dataPath + "/metadata.nlohmann::json";
    nlohmann::json metadata;
    std::ifstream stream(metadataFile.c_str());
    stream >> metadata;

    this->angularx = metadata["angular_resolution"][0];
    this->angulary = metadata["angular_\resolution"][1];
    this->numPoints = metadata["num_points"];
    this->datasetSize = this->angularx * this->angulary * this->numPoints;

    this->dataInputCh = 12;
    this->dataOutputCh = 4;

    // Hardcoded to: position, direction vector, normal/tangent
    this->mlpInputCh = 9;
    // Hardcoded to: RGB
    this->mlpOutputCh = 3;

    // Read input data, and transfer to GPU
    {
        std::string inputFile = dataPath + "/input.bin";

        std::vector<float> inputHost(this->datasetSize * this->dataInputCh * sizeof(float));
        std::ifstream input(inputFile.c_str(), std::ios::binary);
        input.read((char*)inputHost.data(), this->datasetSize * this->dataInputCh * sizeof(float));

        cudaMalloc(&this->inputBuffer, this->datasetSize * this->dataInputCh * sizeof(float));
        cudaMemcpy(this->inputBuffer, inputHost.data(), this->datasetSize * this->dataInputCh * sizeof(float), cudaMemcpyHostToDevice);
    
        input.close();
    }

    // Read output data, and transfer to GPU
    {
        std::string outputFile = dataPath + "/output.bin";

        std::vector<float> outputHost(this->datasetSize * this->dataOutputCh * sizeof(float));
        std::ifstream output(outputFile.c_str(), std::ios::binary);
        output.read((char*)outputHost.data(), this->datasetSize * this->dataOutputCh * sizeof(float));

        cudaMalloc(&this->outputBuffer, this->datasetSize * this->dataOutputCh * sizeof(float));
        cudaMemcpy(this->outputBuffer, outputHost.data(), this->datasetSize * this->dataOutputCh * sizeof(float), cudaMemcpyHostToDevice);
    
        output.close();
    }

    // Batch memory
    cudaMalloc(&this->batchInput, this->datasetSize * this->mlpInputCh * sizeof(float));
    cudaMalloc(&this->batchOutput, this->datasetSize * this->mlpOutputCh * sizeof(float));

    // Indices used for shuffling data
    cudaMalloc(&this->dataIndices, this->datasetSize * sizeof(int));
    thrust::sequence(thrust::device, this->dataIndices, this->dataIndices + this->datasetSize, 0);
}