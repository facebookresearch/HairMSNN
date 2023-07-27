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

#include "hair_msnn.cuh"
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
extern "C" char hair_msnn_ptx[];

struct RenderWindow_HairMSNN : public owl::viewer::OWLViewer
{
    RenderWindow_HairMSNN(Scene& scene, vec2i resolution, bool interactive);

    void initialize();

    vec3f getSceneCg();

    void fetchSceneSamples();
    
    void resetNetwork();
    void setNRCTrain(bool train);

    long long genTrainingData();
    long long train();

    /*! gets called whenever the viewer needs us to re-render out widget */
    long long render() override;
    void drawUI() override;
    void denoise();

    void setRenderComponents();

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

    OWLGroup world, bvhSurfaces;

    Scene currentScene;
    OWLParams launchParams;

    float4* ptAccumBuffer;
    float4* nnAccumBuffer;
    float4* finalAccumBuffer;
         
    float4* ptAverageBuffer;
    float4* nnAverageBuffer;
    float4* finalAverageBuffer;

    // Properties about the scene
    vec3f maxBound = vec3f(1e-30f), minBound = vec3f(1e30f);
    float sceneScale = 0.f;

    // Hair geometry and its parameters
    OWLGeom hairGeom;
    std::vector<float> segmentLengths;
    float totalSegmentsLength = 0.f;

    // Environment lights
    OWLTexture envTextureBuffer;
    int envWidth = 0.f, envHeight = 0.f;
    OWLTexture conditionalCdf, conditionalPdf;
    OWLTexture marginalCdf, marginalPdf;

    // Directional lights
    std::vector<DirectionalLight> dLightList;
    int num_dlights;
    OWLBuffer dLightsBuffer;

    // Debug
    int beta = 0;

    // NRC
    OWLBuffer sampledPointsBuf{ 0 }, sampledSurfaceVectorBuf{ 0 }, sampledParamsBuf{ 0 };
    OWLBuffer sceneIndicesBuf{ 0 };

    OWLBuffer omegaSamplesBuf{ 0 };
    OWLBuffer omegaIndicesBuf{ 0 };

    std::vector<vec3f> sampledPoints;
    std::vector<vec4f> sampledSurfaceVector, sampledParams;
    std::vector<vec3f> omegaSamples;

    int numTrainRecordsX = 128;
    int numTrainRecordsY = 128;
    int numTrainRecords = numTrainRecordsX * numTrainRecordsY;
    int numSamples = 1000000;
    int everyNth = 0;

    OWLBuffer trainIdxs{ 0 };

    bool showCache = false;
    bool saveComponents = false;

    OWLBuffer nnFrameInput{ 0 }, nnFrameOutput{ 0 };
    OWLBuffer nnTrainInput{ 0 }, nnTrainOutput{ 0 };

    OWLBuffer gBuffer{ 0 };

    TINY_MLP* mlp;
    float trainingLoss = 0.f;
    int mlpInputCh = 12, mlpOutputCh = 3;

    float trainTime = 0.f, inferTime = 0.f, renderTime = 0.f;
};