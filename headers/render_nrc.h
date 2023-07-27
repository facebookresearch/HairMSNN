// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#include "owlViewer/OWLViewer.h"
#include "owl/owl.h"
#include "owl/common/math/random.h"

#include "scene.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"

#include "nrc.cuh"
#include "neural_network.cuh"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

// Compiled PTX code
extern "C" char nrc_ptx[];

struct RenderWindowNRC : public owl::viewer::OWLViewer
{
    RenderWindowNRC(Scene& scene, vec2i resolution, bool interactive);

    void initialize();
    
    void setAllUnbiasedPaths(bool val);
    void setNRCTrain(bool train);
    void showCacheVisualization(bool cache);
    void showBounceVisualization(bool vis, float thresh);
    void setBounceThresholdControl(float c);

    /*! gets called whenever the viewer needs us to re-render out widget */
    long long render() override;
    void drawUI() override;
    void denoise();

    // /*! window notifies us that we got resized. We HAVE to override
    //     this to know our actual render dimensions, and get pointer
    //     to the device frame buffer that the viewer cated for us */
    void resize(const vec2i& newSize) override;

    // /*! this function gets called whenever any camera manipulator
    //   updates the camera. gets called AFTER all values have been updated */
    void cameraChanged() override;

    void customKey(char key, const vec2i& pos) override;
    void screenShotEXR(std::string fname);

    void uploadLights();

    OWLContext context{ 0 };
    OWLModule module{ 0 };

    bool sbtDirty = true;
    bool progressive = false;

    OWLRayGen rayGen{ 0 };
    OWLMissProg missProg{ 0 };

    OWLGroup world;

    Scene currentScene;
    OWLParams launchParams;

    // Properties about the scene
    vec3f maxBound = vec3f(1e-30f), minBound = vec3f(1e30f);
    float sceneScale = 0.f;

    // Hair geometry and its parameters
    OWLGeom hairGeom;

    // Environment lights
    OWLTexture envTextureBuffer;
    int envWidth = 0.f, envHeight = 0.f;
    OWLTexture conditionalCdf, conditionalPdf;
    OWLTexture marginalCdf, marginalPdf;

    // Directional lights
    std::vector<DirectionalLight> dLightList;
    int num_dlights;
    OWLBuffer dLightsBuffer;

    // OptiX denoiser
    OptixDenoiser denoiser;
    void* denoiserScratch;
    size_t denoiserScratchSize;
    void* denoiserState;
    size_t denoiserStateSize;

    int tileSize = 64;
    vec2i numBlocksAndThreads;

    float4* denoisedBuffer;
    float4* denoiserAlbedoInput;
    float4* denoiserNormalInput;
    float* denoiserIntensity;

    // NRC
    bool showCache = false;
    bool showBounces = false;
    bool allUnbiased = false;
    int bounceThreshold = 0;

    OWLBuffer gBuffer{ 0 }, tBuffer{ 0 };
    OWLBuffer nnFrameInput{ 0 }, nnFrameOutput{ 0 };

    OWLBuffer trainIdxs{ 0 }, trainInput{ 0 }, trainGT{ 0 };

    TINY_MLP* mlp;
    int mlpInputCh = 9, mlpOutputCh = 3;

    int numTrainingRecords = 65536;
    int numTrainingPixels = 0;
    int numUnbiasedPixels = 0;
    int everyNth = 0;
    int nnFrameSize = 0;

    float c = 0.01; // NRC sect. 3.4, spread threshold multiplier

    float trainingLoss = 0.f;
};