// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#pragma once

#include "model.h"
#include "cyHairFile.h"
#include "json/json.hpp"

struct SceneCamera {
	vec3f from;
	vec3f at;
	vec3f up;
	float cosFovy;
};

struct Scene {
	nlohmann::json json;
	std::string jsonFilePath;

	// Scene contents
	Model* surface;
	bool has_surface = false;

	cyHairFile hair;
	HairModel hairModel;
	bool has_hair = false;

	// Camera
	SceneCamera camera;
	std::vector<SceneCamera> cameraPath;

	// Hair global material properties
	int bsdf = 0;

	vec3f sig_a = vec3f(0.06f, 0.1f, 0.2f);
	float beta_m = 0.3, beta_n = 0.3;
	float alpha = 0.03;
	float R_G = 1.f, TT_G = 1.f, TRT_G = 1.f, TRRT_G = 1.f;
	float pixarFactor = 1.f;

	// Environment light
	bool hasEnvLight = false;
	Texture env;
	float envScale = 0.f, envRotPhi = 0.f;
	
	// Directional lights
	bool hasDirectionalLights = false;
	std::vector<vec3f> dLightFrom;
	std::vector<vec3f> dLightEmit;

	// General integrator parameters
	bool MIS = true, envPdfSampling = true;
	bool denoise = false;

	int spp = 1, path_v1 = 0, path_v2 = 0;
	int imgWidth = 1280, imgHeight = 1280;
	std::string renderOutput = "";
	std::string renderStatsOutput = "";

	bool equalTime = false;
	float time = 0.f;

	// Dual scattering precomputation path
	std::string dualScatterPrecompPath = "";

	// NRC parameters
	std::string nrcConfig = "";
	std::string nrcConfig2 = "";
	std::string nrcWeights = "";
	bool nrcTrain = true;

	bool directLightVisibility = true;

	// Extract hair data
	void extractHairData();
	void loadYuefan(std::string filePath);
};

bool parseScene(std::string sceneFile, Scene &scene);
Scene parseScene_(std::string sceneFile);

void LoadCemYuksel(const char* filename, cyHairFile& hairfile);

void generateEnvSamplingTables(vec2i resolution, float* envMap, float** cPdf, float** cCdf, float** mPdf, float** mCdf);