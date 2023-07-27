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
#include <time.h>
#include <limits>
#include <random>

#ifdef _WIN32
#include <direct.h>
#define getCurrentDir_ _getcwd
#else
#include <unistd.h>
#define getCurrentDir_ getcwd
#endif

#include "common.h"
#include "owl/common/math/vec.h"
#include "owl/common/math/AffineSpace.h"

using namespace owl;

struct Texture {
    ~Texture()
    {
        if (pixel) delete[] pixel;
    }

    vec3f sample(vec2f uv);

    uint32_t* pixel{ nullptr };
    float* pixel_float{ nullptr };
    vec2i     resolution{ -1 };
};

/*! a simple indexed triangle mesh that our sample renderer will
    render */
struct TriangleMesh {
    
    std::vector<vec3f> vertex;
    std::vector<vec3f> normal;
    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;
    std::vector<float> triangleArea;

    vec4f sampleSurfaceParams(std::vector<Texture*>& textures, vec3f point, vec3i index);

    // material data:
    vec3f              diffuse;
    int                diffuseTextureID{ -1 };

    float              alpha;
    int                alphaTextureID{ -1 };

    float totalSurfaceArea = 0.f;
    vec3f minBound = vec3f(1e30f);
    vec3f maxBound = vec3f(-1e30f);
};

struct Model {
    ~Model()
    {
        for (auto mesh : meshes) delete mesh;
    }

    std::vector<TriangleMesh*> meshes;
    std::vector<Texture*>      textures;
    //! bounding box of all vertices in the model
    box3f bounds;

    float totalArea = 0.f;
};

struct HairModel {
    std::vector<vec3f> controlPoints;
    std::vector<float> widths;
    std::vector<int> segmentIndices;
    std::vector<int> numSgmentsInStrand;
    int numStrands;

    vec3f minBound = vec3f(0.f);
    vec3f maxBound = vec3f(0.f);
    float scale = 0.f;
};

Model* loadOBJ(const std::string& objFile);
bool loadEnvTexture(std::string& path, Texture* texture);

int saveBufferAsEXR(std::string path, float* hostBuffer, int width, int height, int numComponents);
float* loadBufferFromEXR(std::string path, int* width, int* height, int numComponents);

int saveEXR(std::string path, float* hostBuffer, int accumId, int width, int height);