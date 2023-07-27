// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#pragma once

#include "common.cuh"

enum HitInfo {
	ESCAPE = 0,
	LOCAL_SCATTER,
	GLOBAL_SCATTER
};

enum RenderPass {
	TRAIN_DATA_GEN = 0,
	G_BUFFER,
	RENDER
};

enum OutputType {
	COMBINED = 0,
	NRC_ONLY,
	PT_ONLY
};

struct GBuffer {
	bool hit, isSurface;
	vec3f p;
	vec3f shortPathColor;
};

struct LaunchParams {
	float4* ptAccumBuffer;
	float4* nnAccumBuffer;
	float4* finalAccumBuffer;
		 
	float4* ptAverageBuffer;
	float4* nnAverageBuffer;
	float4* finalAverageBuffer;

	int everyNth;
	int* trainIdxs;

	int accumId;

	GBuffer* gBuffer;

	OptixTraversableHandle world, bvhSurfaces;

	bool hasEnvLight, envPdfSampling;
	float envScale, envRotPhi;
	int envWidth, envHeight;
	cudaTextureObject_t env;
	cudaTextureObject_t conditionalCdf, conditionalPdf;
	cudaTextureObject_t marginalCdf, marginalPdf;

	bool hasDirectionalLights;
	DirectionalLight* dLights;
	int num_dlights;

	int num_total_lights;

	struct {
		vec3f pos;
		vec3f dir_00;
		vec3f dir_du;
		vec3f dir_dv;
	} camera;

	struct {
		vec3f sig_a;
		float beta_m, beta_n;
		float alpha;

		float R_G, TT_G, TRT_G, TRRT_G;
	} hairData;

	bool MIS;
	int pathV1, pathV2;

	// Properties of the scene
	vec3f maxBound, minBound;
	float sceneScale;

	// Denoiser
	float4* denoiserAlbedoInput;
	float4* denoiserNormalInput;

	// NRC variables
	int pass;

	float* nnFrameInput;
	float* nnFrameOutput;

	int mlpInputCh, mlpOutputCh;

	float* nnTrainInput;
	float* nnTrainOutput;

	float3* sampledPoints;
	float4* sampledSurfaceVector;
	float4* sampledParams;

	int* sceneIndices;

	float* omegaSamples;
	int* omegaIndices;

	int numTrainRecordsX, numTrainRecordsY, numSamples;

	// Debug
	int beta;
};

__constant__ LaunchParams optixLaunchParams;