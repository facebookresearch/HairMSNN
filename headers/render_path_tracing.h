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

#include "path_tracing.cuh"
#include <optix_stubs.h>

// Compiled PTX code
extern "C" char path_tracing_ptx[];

struct RenderWindowPT : public owl::viewer::OWLViewer
{
    RenderWindowPT(Scene& scene, vec2i resolution, bool interactive);

    void initialize();

    /*! gets called whenever the viewer needs us to re-render out widget */
    long long render() override;

    void drawUI() override;

    // /*! window notifies us that we got resized. We HAVE to override
    //     this to know our actual render dimensions, and get pointer
    //     to the device frame buffer that the viewer cated for us */
    void resize(const vec2i& newSize) override;

    // /*! this function gets called whenever any camera manipulator
    //   updates the camera. gets called AFTER all values have been updated */
    void cameraChanged() override;

    void customKey(char key, const vec2i& pos) override;
    void screenShotEXR(std::string fname);

    vec3f getSceneCg();

    void uploadLights();

    OWLContext context{ 0 };
    OWLModule module{ 0 };

    bool sbtDirty = true;
    bool progressive = true;

    OWLRayGen rayGen{ 0 };
    OWLMissProg missProg{ 0 };

    OWLGroup world;

    Scene currentScene;
    OWLParams launchParams;

    // Properties about the scene
    vec3f maxBound = vec3f(-1e30f), minBound = vec3f(1e30f);
    float sceneScale = 0.f;

    // Hair geometry and its parameters
    OWLGeom hairGeom;

    // Environment lights
    OWLTexture envTextureBuffer;
    OWLTexture conditionalPdf, conditionalCdf;
    OWLTexture marginalPdf, marginalCdf;
    int envWidth = 0.f, envHeight = 0.f;

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
};