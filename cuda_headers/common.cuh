// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#pragma once

#include "owl/owl.h"
#include "owl/common/math/LinearSpace.h"
#include "owl/common/math/vec.h"
#include "cuda_runtime.h"
#include "owl/common/math/random.h"

#include "random.cuh"
#include "curve_utils.cuh"

using namespace owl;

enum RayTypes {
	RADIANCE_RAY_TYPE = 0,
	SHADOW_RAY_TYPE,
	MULTISCATTER_RAY_TYPE,
	NUM_RAY_TYPES
};

#ifdef __CUDA_ARCH__
typedef RayT<RADIANCE_RAY_TYPE, NUM_RAY_TYPES> RadianceRay;
typedef RayT<SHADOW_RAY_TYPE, NUM_RAY_TYPES> ShadowRay;
typedef RayT<MULTISCATTER_RAY_TYPE, NUM_RAY_TYPES> MultiScatterRay;
#endif

struct HairInteraction {
	float R_G = 1.f, TT_G = 1.f, TRT_G = 1.f, TRRT_G = 1.f;

	vec3f curve_p = vec3f(0.f);
	float radius = 0.f;
	float v[3] = { 0.f };
	float s = 0.f;

	float sin2kAlpha[3] = { 0.f }, cos2kAlpha[3] = { 0.f };
};

// Interaction data
struct Interaction {
	bool hit = false;
	bool isSurface = true;
	vec3f Le = vec3f(0.f);

	vec3f p = vec3f(0.f);
	vec2f uv = vec2f(0.f);
	vec3f wo = vec3f(0.f), wi = vec3f(0.f);
	vec3f wo_local = vec3f(0.f), wi_local = vec3f(0.f);

	vec3f n = vec3f(0.f);
	vec3f t = vec3f(0.f);

	vec3f to_local[3], to_world[3];

	// General interaction
	vec3f color = vec3f(0.f);
	float alpha = 0.f;
	float beta_m = 0.f, beta_n = 0.f;

	// Hair interaction ptr
	HairInteraction hair;
};

// Direcitonal lights
struct DirectionalLight {
	vec3f from = vec3f(0.f);
	vec3f emit = vec3f(0.f);
};

// Common Optix structs
struct RayGenData {
	uint32_t* frameBuffer;
	vec2i frameBufferSize;
};

struct MissProgData {
	vec3f const_color = vec3f(0.f);
};

struct ShadowRayData {
	vec3f visibility = vec3f(0.f);
	bool isSurface = false;
};

struct MultiScatterRayData {
	bool surfaceHit = false;
	bool hairHit = false;
	int numHairHits = 0;
};

struct HairData {
	
};

struct TriangleMeshData {
	vec3f* vertex;
	vec3f* normal;
	vec3i* index;
	vec2f* texCoord;

	vec3f diffuse;
	bool hasDiffuseTexture;
	cudaTextureObject_t diffuse_texture;

	float alpha;
	bool hasAlphaTexture;
	cudaTextureObject_t alpha_texture;
};