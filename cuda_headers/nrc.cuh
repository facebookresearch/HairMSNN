// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#pragma once

#include "common.cuh"

#define DEBUG true
#define MAX_BOUNCES 40

enum RenderPass {
	G_BUFFER = 0,
	RENDER,
	RESET
};

struct GBuffer {
	bool hit = false;

	vec3f pathRadiance = vec3f(0.f);
	vec3f beta = vec3f(0.f);
	int bounces = 0;
};

struct TrainBuffer {
	vec3f vert[MAX_BOUNCES] = { vec3f(0.f) };
	vec3f wo[MAX_BOUNCES] = { vec3f(0.f) };
	vec3f n[MAX_BOUNCES] = { vec3f(0.f) };

	vec3f vertRadiance[MAX_BOUNCES] = {vec3f(0.f)};
	vec3f vertBeta[MAX_BOUNCES] = { vec3f(0.f) };
	vec4f color[MAX_BOUNCES] = { vec4f(0.f) };

	int bounces = 0;
	bool hit = false;
};

struct LaunchParams {
	float4* accumBuffer;
	float4* averageBuffer;
	int accumId;

	OptixTraversableHandle world;

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
	bool allUnbiased;

	// Debug
	bool showCache, showBounces;
	int bounceThreshold;
	cudaTextureObject_t colormap;

	// Properties of the scene
	vec3f maxBound, minBound;
	float sceneScale;

	// NRC variables
	int pass;

	GBuffer* gBuffer;
	TrainBuffer* tBuffer;

	float* nnFrameInput;
	float3* nnFrameOutput;

	int* trainIdxs;
	float* trainInput;
	float3* trainGT;

	int mlpInputCh, mlpOutputCh;

	int numTrainingRecords;
	int numTrainingPixels;
	int everyNth;	

	float c;
};

__constant__ LaunchParams optixLaunchParams;