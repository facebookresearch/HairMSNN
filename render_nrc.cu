// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#include "render_nrc.h"

// ====================================================
// Custom functions
// ====================================================
void RenderWindowNRC::setAllUnbiasedPaths(bool val)
{
    this->allUnbiased = val;
    owlParamsSet1b(this->launchParams, "allUnbiased", val);
    this->cameraChanged();
}

void RenderWindowNRC::setNRCTrain(bool train)
{
    this->currentScene.nrcTrain = train;
}

void RenderWindowNRC::showCacheVisualization(bool cache)
{
    this->showCache = cache;
    owlParamsSet1b(this->launchParams, "showCache", cache);
    this->cameraChanged();
}

void RenderWindowNRC::showBounceVisualization(bool vis, float thresh)
{
    this->showBounces = vis;
    owlParamsSet1b(this->launchParams, "showBounces", vis);

    this->bounceThreshold = thresh;
    owlParamsSet1i(this->launchParams, "bounceThreshold", thresh);
    this->cameraChanged();
}

void RenderWindowNRC::setBounceThresholdControl(float c)
{
    this->c = c;
    owlParamsSet1f(this->launchParams, "c", c);
    this->cameraChanged();
}

void RenderWindowNRC::denoise()
{
    OptixDenoiserParams denoiserParams;
    denoiserParams.denoiseAlpha = 0;
    denoiserParams.hdrIntensity = (CUdeviceptr)this->denoiserIntensity;
    denoiserParams.blendFactor = 0.0f;

    OptixImage2D inputLayer[3];
    inputLayer[0].data = (CUdeviceptr)this->averageBuffer;
    inputLayer[0].width = this->fbSize.x;
    inputLayer[0].height = this->fbSize.y;
    inputLayer[0].rowStrideInBytes = this->fbSize.x * sizeof(float4);
    inputLayer[0].pixelStrideInBytes = sizeof(float4);
    inputLayer[0].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    inputLayer[1].data = (CUdeviceptr)this->denoiserAlbedoInput;
    inputLayer[1].width = this->fbSize.x;
    inputLayer[1].height = this->fbSize.y;
    inputLayer[1].rowStrideInBytes = this->fbSize.x * sizeof(float4);
    inputLayer[1].pixelStrideInBytes = sizeof(float4);
    inputLayer[1].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    inputLayer[2].data = (CUdeviceptr)this->denoiserNormalInput;
    inputLayer[2].width = this->fbSize.x;
    inputLayer[2].height = this->fbSize.y;
    inputLayer[2].rowStrideInBytes = this->fbSize.x * sizeof(float4);
    inputLayer[2].pixelStrideInBytes = sizeof(float4);
    inputLayer[2].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixImage2D outputLayer;
    outputLayer.data = (CUdeviceptr)this->denoisedBuffer;
    outputLayer.width = this->fbSize.x;
    outputLayer.height = this->fbSize.y;
    outputLayer.rowStrideInBytes = this->fbSize.x * sizeof(float4);
    outputLayer.pixelStrideInBytes = sizeof(float4);
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixDenoiserGuideLayer denoiserGuideLayer = {};
    denoiserGuideLayer.albedo = inputLayer[1];
    denoiserGuideLayer.normal = inputLayer[2];

    OptixDenoiserLayer denoiserLayer = {};
    denoiserLayer.input = inputLayer[0];
    denoiserLayer.output = outputLayer;

    optixDenoiserComputeIntensity(this->denoiser,
        /*stream*/0,
        &inputLayer[0],
        (CUdeviceptr)this->denoiserIntensity,
        (CUdeviceptr)this->denoiserScratch,
        this->denoiserScratchSize
    );

    optixDenoiserInvoke(this->denoiser,
        /*stream*/0,
        &denoiserParams,
        (CUdeviceptr)this->denoiserState,
        this->denoiserStateSize,
        &denoiserGuideLayer,
        &denoiserLayer, 1,
        /*inputOffsetX*/0,
        /*inputOffsetY*/0,
        (CUdeviceptr)this->denoiserScratch,
        this->denoiserScratchSize
    );
}

void RenderWindowNRC::initialize()
{
    vec2i frameRes(this->getWindowSize().x, this->getWindowSize().y);
    int frameSize = frameRes.x * frameRes.y;

    // ====================================================
    // Setup NRC variables & buffers
    // ====================================================

    // Number of training pixels depends on the number of bounces
    // This is because, for a pixel path, we will record radiance at each path vertex
    // This means, each pixel will generate B training data, where B is number of bounces
    // Since we have a upper limit on the number of training records, 
    // we scale the number training pixels like below.
    this->numTrainingPixels = std::ceil(this->numTrainingRecords / MAX_BOUNCES);

    // Every 16th pixel needs to be traced in an unbiased fashion. 
    // This ensures that the cache is not overly wrong.
    this->numUnbiasedPixels = std::ceil(this->numTrainingPixels / 16);

    // Every nth pixel is a training pixel
    this->everyNth = std::ceil(frameSize / this->numTrainingPixels);

    // This bit is tricky. We need to pass the cache hits for the current frame
    // and also cache hits for training suffixes (biased + unbiased)
    // for network evaluation. 
    // Now, TCNN requires that the batch size be a multiple of 128.
    // I am manually ensuring that the frame size is infact a multiple of 128
    // Thus, the following adds the extra space for cache hits, upper bounding such that the total is a multiple of 128.
    int remainder = this->numTrainingPixels % 128;
    this->nnFrameSize = frameSize + this->numTrainingPixels - remainder + 128;

    this->mlp = new TINY_MLP(this->currentScene.nrcConfig, this->mlpInputCh, this->mlpOutputCh);
    if(this->currentScene.nrcWeights != "")
        this->mlp->loadWeights(this->currentScene.nrcWeights);

    // The tBuffer (or TrainBuffer) is used to record training values at training pixel path vertices.
    // Recorded values include the position, normal, view vector, beta, direct lighting etc.
    this->tBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(TrainBuffer), this->numTrainingPixels, nullptr);

    // G-buffer in this case records accumulated radiance and betas before cache hit.
    // It thus needs to be as large as the frame itself.
    this->gBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(GBuffer), frameSize, nullptr);

    // Both NN input and output buffers are of nnFrameSize, which includes the cache hit for pixels
    // plus the seond cache hit for training pixels. (upper bounded to be a multiple of 128, a TCNN requirement)
    this->nnFrameInput = owlDeviceBufferCreate(context, OWL_FLOAT,
        this->nnFrameSize * this->mlpInputCh, nullptr);
    this->nnFrameOutput = owlDeviceBufferCreate(context, OWL_FLOAT3,
        this->nnFrameSize, nullptr);

    // Training indices ; fill with regular grid.
    // Shuffle each frame take random pixels as training data
    {
        int* idxs = (int*)malloc(this->numTrainingPixels * sizeof(int));
        thrust::sequence(thrust::host, idxs, idxs + this->numTrainingPixels, 0);

        this->trainIdxs = owlDeviceBufferCreate(context, OWL_INT, this->numTrainingPixels, idxs);

        free(idxs);
    }

    // All the training buffers are of the same size i.e. numTrainingRecords
    this->trainInput = owlDeviceBufferCreate(context, OWL_FLOAT, this->numTrainingRecords * this->mlpInputCh, nullptr);
    this->trainGT = owlDeviceBufferCreate(context, OWL_FLOAT3, this->numTrainingRecords, nullptr);

    // Accumulation buffer stores the non-averaged radiance in float32
    // Float buffer stores the averaged radiance in float32
    // Frame buffer (defined in OWL::Viewer) stores averaged radiance in uint8
    cudaMalloc(&this->accumBuffer, frameSize * sizeof(float4));
    cudaMalloc(&this->averageBuffer, frameSize * sizeof(float4));

    // ====================================================
    // Denoiser setup
    // ====================================================
    this->numBlocksAndThreads = vec2i(frameRes.y / this->tileSize, frameRes.x / this->tileSize);
    auto optixContext = owlContextGetOptixContext(context, 0);

    OptixDenoiserOptions denoiserOptions = {};
    optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiserOptions, &denoiser);

    OptixDenoiserSizes denoiserReturnSizes;
    optixDenoiserComputeMemoryResources(this->denoiser, frameRes.x, frameRes.y, &denoiserReturnSizes);

    this->denoiserScratchSize = std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes, denoiserReturnSizes.withoutOverlapScratchSizeInBytes);
    this->denoiserStateSize = denoiserReturnSizes.stateSizeInBytes;

    cudaMalloc(&this->denoiserScratch, this->denoiserScratchSize);
    cudaMalloc(&this->denoiserState, this->denoiserStateSize);

    cudaMalloc(&this->denoisedBuffer, frameSize * sizeof(float4));
    cudaMalloc(&this->denoiserAlbedoInput, frameSize * sizeof(float4));
    cudaMalloc(&this->denoiserNormalInput, frameSize * sizeof(float4));
    cudaMalloc(&this->denoiserIntensity, sizeof(float));

    optixDenoiserSetup(this->denoiser, 0,
        frameRes.x, frameRes.y,
        (CUdeviceptr)this->denoiserState,
        this->denoiserStateSize,
        (CUdeviceptr)this->denoiserScratch,
        this->denoiserScratchSize);

    // ====================================================
    // Environment lights setup
    // ====================================================
    if (this->currentScene.hasEnvLight) {

        int resx = this->currentScene.env.resolution.x;
        int resy = this->currentScene.env.resolution.y;
        vec2i resolution(resx, resy);
        float* envMap = this->currentScene.env.pixel_float;

        float* cPdf;
        float* cCdf;
        float* mPdf;
        float* mCdf;
        generateEnvSamplingTables(resolution, envMap, &cPdf, &cCdf, &mPdf, &mCdf);

        this->envTextureBuffer = owlTexture2DCreate(context,
            OWL_TEXEL_FORMAT_RGBA32F,
            resx, resy,
            envMap,
            OWL_TEXTURE_LINEAR,
            OWL_TEXTURE_CLAMP);
        this->envWidth = resx;
        this->envHeight = resy;

        this->conditionalPdf = owlTexture2DCreate(context,
            OWL_TEXEL_FORMAT_R32F,
            resx + 1, resy,
            cPdf,
            OWL_TEXTURE_NEAREST,
            OWL_TEXTURE_CLAMP);

        this->conditionalCdf = owlTexture2DCreate(context,
            OWL_TEXEL_FORMAT_R32F,
            resx + 1, resy,
            cCdf,
            OWL_TEXTURE_NEAREST,
            OWL_TEXTURE_CLAMP);

        this->marginalPdf = owlTexture2DCreate(context,
            OWL_TEXEL_FORMAT_R32F,
            resy + 1, 1.f,
            mPdf,
            OWL_TEXTURE_NEAREST,
            OWL_TEXTURE_CLAMP);

        this->marginalCdf = owlTexture2DCreate(context,
            OWL_TEXEL_FORMAT_R32F,
            resy + 1, 1.f,
            mCdf,
            OWL_TEXTURE_NEAREST,
            OWL_TEXTURE_CLAMP);

        free(cPdf); free(cCdf); free(mPdf); free(mCdf);
    }

    // ====================================================
    // Directional lights setup
    // ====================================================
    this->num_dlights = this->currentScene.dLightFrom.size();
    for (int i = 0; i < num_dlights; i++) {
        DirectionalLight dLight;
        dLight.from = this->currentScene.dLightFrom[i];
        dLight.emit = this->currentScene.dLightEmit[i];

        this->dLightList.push_back(dLight);
    }

    this->dLightsBuffer = owlDeviceBufferCreate(context,
        OWL_USER_TYPE(DirectionalLight), this->dLightList.size(), this->dLightList.data());

    // ====================================================
    // Vector of bottom level accel. structures (BLAS) 
    // This is then used to build the top level accel. structure (TLAS)
    // ====================================================
    std::vector<OWLGroup> blasList;

    // ====================================================
    // Surface geometry and data setup
    // ====================================================
    if (this->currentScene.has_surface) {
        auto surface = this->currentScene.surface;
        for (auto mesh : surface->meshes) {
            OWLVarDecl triangleGeomVars[] = {
                {"vertex", OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshData, vertex)},
                {"normal", OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshData, normal)},
                {"index", OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshData, index)},
                {"texCoord", OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshData, texCoord)},

                {"diffuse", OWL_FLOAT3, OWL_OFFSETOF(TriangleMeshData, diffuse)},
                {"diffuse_texture", OWL_TEXTURE, OWL_OFFSETOF(TriangleMeshData, diffuse_texture)},
                {"hasDiffuseTexture", OWL_BOOL, OWL_OFFSETOF(TriangleMeshData, hasDiffuseTexture)},

                {"alpha", OWL_FLOAT, OWL_OFFSETOF(TriangleMeshData, alpha)},
                {"alpha_texture", OWL_TEXTURE, OWL_OFFSETOF(TriangleMeshData, alpha_texture)},
                {"hasAlphaTexture", OWL_BOOL, OWL_OFFSETOF(TriangleMeshData, hasAlphaTexture)},

                {nullptr}
            };

            OWLGeomType triangleGeomType = owlGeomTypeCreate(context,
                OWL_GEOM_TRIANGLES,
                sizeof(TriangleMeshData),
                triangleGeomVars,
                -1);

            owlGeomTypeSetClosestHit(triangleGeomType, RADIANCE_RAY_TYPE, module, "triangleMeshCH");
            owlGeomTypeSetAnyHit(triangleGeomType, SHADOW_RAY_TYPE, module, "triangleMeshAHShadow");

            OWLGeom triangleGeom = owlGeomCreate(context, triangleGeomType);

            OWLBuffer vertexBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, mesh->vertex.size(), mesh->vertex.data());
            OWLBuffer vertexBuffer2 = owlDeviceBufferCreate(context, OWL_FLOAT3, mesh->vertex.size(), mesh->vertex.data());
            OWLBuffer normalBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, mesh->normal.size(), mesh->normal.data());
            OWLBuffer indexBuffer = owlDeviceBufferCreate(context, OWL_INT3, mesh->index.size(), mesh->index.data());
            OWLBuffer texCoordBuffer = owlDeviceBufferCreate(context, OWL_FLOAT2, mesh->texcoord.size(), mesh->texcoord.data());

            // Create CUDA buffers and upload them for diffuse and alpha textures
            if (mesh->diffuseTextureID != -1) {
                Texture* diffuseTexture = surface->textures[mesh->diffuseTextureID];
                OWLTexture diffuseTextureBuffer = owlTexture2DCreate(context,
                    OWL_TEXEL_FORMAT_RGBA8,
                    diffuseTexture->resolution.x,
                    diffuseTexture->resolution.y,
                    diffuseTexture->pixel,
                    OWL_TEXTURE_NEAREST,
                    OWL_TEXTURE_MIRROR);
                owlGeomSetTexture(triangleGeom, "diffuse_texture", diffuseTextureBuffer);
                owlGeomSet1b(triangleGeom, "hasDiffuseTexture", true);
            }
            else {
                owlGeomSet3f(triangleGeom, "diffuse", owl3f{ mesh->diffuse.x, mesh->diffuse.y, mesh->diffuse.z });
                owlGeomSet1b(triangleGeom, "hasDiffuseTexture", false);
            }

            if (mesh->alphaTextureID != -1) {
                Texture* alphaTexture = surface->textures[mesh->alphaTextureID];
                OWLTexture alphaTextureBuffer = owlTexture2DCreate(context,
                    OWL_TEXEL_FORMAT_RGBA8,
                    alphaTexture->resolution.x,
                    alphaTexture->resolution.y,
                    alphaTexture->pixel,
                    OWL_TEXTURE_NEAREST,
                    OWL_TEXTURE_MIRROR);
                owlGeomSetTexture(triangleGeom, "alpha_texture", alphaTextureBuffer);
                owlGeomSet1b(triangleGeom, "hasAlphaTexture", true);
            }
            else {
                owlGeomSet1f(triangleGeom, "alpha", mesh->alpha);
                owlGeomSet1b(triangleGeom, "hasAlphaTexture", false);
            }

            owlTrianglesSetVertices(triangleGeom, vertexBuffer,
                mesh->vertex.size(), sizeof(vec3f), 0);
            owlTrianglesSetIndices(triangleGeom, indexBuffer,
                mesh->index.size(), sizeof(vec3i), 0);

            owlGeomSetBuffer(triangleGeom, "vertex", vertexBuffer2);
            owlGeomSetBuffer(triangleGeom, "normal", normalBuffer);
            owlGeomSetBuffer(triangleGeom, "index", indexBuffer);
            owlGeomSetBuffer(triangleGeom, "texCoord", texCoordBuffer);

            OWLGroup triangleGroup = owlTrianglesGeomGroupCreate(context, 1, &triangleGeom);
            owlGroupBuildAccel(triangleGroup);

            this->minBound = min(this->minBound, mesh->minBound);
            this->maxBound = max(this->maxBound, mesh->maxBound);

            // Add to a list, which is later used to build the IAS (TLAS)
            blasList.push_back(triangleGroup);
        }

        this->sceneScale = length(this->maxBound - this->minBound);
    }


    // ====================================================
    // Hair setup (geometry and BSDF)
    // ====================================================
    OWLVarDecl hairGeomVars[] = {
        {nullptr}
    };

    OWLGeomType hairGeomType = owlGeomTypeCreate(context,
        OWL_GEOMETRY_CURVES,
        sizeof(HairData),
        hairGeomVars,
        -1);

    owlGeomTypeSetClosestHit(hairGeomType, RADIANCE_RAY_TYPE, module, "hairCH");
    owlGeomTypeSetAnyHit(hairGeomType, SHADOW_RAY_TYPE, module, "hairAHShadow");
    owlCurvesSetDegree(hairGeomType, 3, false);

    // ====================================================
    // Data setup (hair strands from .hair file)
    // ====================================================

    this->hairGeom = owlGeomCreate(context, hairGeomType);

    if (this->currentScene.has_hair) {
        this->minBound = min(this->currentScene.hairModel.minBound, minBound);
        this->maxBound = max(this->currentScene.hairModel.maxBound, maxBound);
        this->sceneScale = max(this->currentScene.hairModel.scale, this->sceneScale);

        OWLBuffer cpBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3,
            this->currentScene.hairModel.controlPoints.size(), this->currentScene.hairModel.controlPoints.data());
        OWLBuffer widthsBuffer = owlDeviceBufferCreate(context, OWL_FLOAT,
            this->currentScene.hairModel.widths.size(), this->currentScene.hairModel.widths.data());
        OWLBuffer indicesBuffer = owlDeviceBufferCreate(context, OWL_INT,
            this->currentScene.hairModel.segmentIndices.size(), this->currentScene.hairModel.segmentIndices.data());

        owlCurvesSetControlPoints(this->hairGeom, this->currentScene.hairModel.controlPoints.size(), cpBuffer, widthsBuffer);
        owlCurvesSetSegmentIndices(this->hairGeom, this->currentScene.hairModel.segmentIndices.size(), indicesBuffer);

        OWLGroup hairGroup = owlCurvesGeomGroupCreate(context, 1, &this->hairGeom);
        owlGroupBuildAccel(hairGroup);

        blasList.push_back(hairGroup);
    }

    // ====================================================
    // Build the TLAS (IAS)
    // ====================================================
    world = owlInstanceGroupCreate(context, blasList.size(), blasList.data());
    owlGroupBuildAccel(world);

    // ====================================================
    // Launch Parameters setup
    // ====================================================
    OWLVarDecl launchParamsDecl[] = {
        // NRC variables declaration
        {"allUnbiased", OWL_BOOL, OWL_OFFSETOF(LaunchParams, allUnbiased)},
        {"pass", OWL_INT, OWL_OFFSETOF(LaunchParams, pass)},

        {"gBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, gBuffer)},
        {"tBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, tBuffer)},

        {"nnFrameInput", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, nnFrameInput)},
        {"nnFrameOutput", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, nnFrameOutput)},

        {"trainIdxs", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, trainIdxs)},
        {"trainInput", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, trainInput)},
        {"trainGT", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, trainGT)},

        {"mlpInputCh", OWL_INT, OWL_OFFSETOF(LaunchParams, mlpInputCh)},
        {"mlpOutputCh", OWL_INT, OWL_OFFSETOF(LaunchParams, mlpOutputCh)},

        {"numTrainingRecords", OWL_INT, OWL_OFFSETOF(LaunchParams, numTrainingRecords)},
        {"numTrainingPixels", OWL_INT, OWL_OFFSETOF(LaunchParams, numTrainingPixels)},
        {"everyNth", OWL_INT, OWL_OFFSETOF(LaunchParams, everyNth)},

        {"c", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, c)},
        // Environment light parametsrs
        {"envRotPhi", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, envRotPhi)},
        {"envScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, envScale)},
        {"hasEnvLight", OWL_BOOL, OWL_OFFSETOF(LaunchParams, hasEnvLight)},
        {"env", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, env)},
        {"envWidth", OWL_INT, OWL_OFFSETOF(LaunchParams, envWidth)},
        {"envHeight", OWL_INT, OWL_OFFSETOF(LaunchParams, envHeight)},
        // Environment light PDF & CDF
        {"conditionalPdf", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, conditionalPdf)},
        {"conditionalCdf", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, conditionalCdf)},
        {"marginalPdf", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, marginalPdf)},
        {"marginalCdf", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, marginalCdf)},
        // Directional light parameters
        {"hasDirectionalLights", OWL_BOOL, OWL_OFFSETOF(LaunchParams, hasDirectionalLights)},
        {"dLights", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, dLights)},
        {"num_dlights", OWL_INT, OWL_OFFSETOF(LaunchParams, num_dlights)},
        // Global light parameters
        {"num_total_lights", OWL_INT, OWL_OFFSETOF(LaunchParams, num_total_lights)},
        // Integrator parameters
        {"MIS", OWL_BOOL, OWL_OFFSETOF(LaunchParams, MIS)},
        {"envPdfSampling", OWL_BOOL, OWL_OFFSETOF(LaunchParams, envPdfSampling)},
        // Scene properties
        {"maxBound", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, maxBound)},
        {"minBound", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, minBound)},
        {"sceneScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, sceneScale)},
        // Debug
        {"colormap", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, colormap)},
        {"showCache", OWL_BOOL, OWL_OFFSETOF(LaunchParams, showCache)},
        {"showBounces", OWL_BOOL, OWL_OFFSETOF(LaunchParams, showBounces)},
        {"bounceThreshold", OWL_INT, OWL_OFFSETOF(LaunchParams, bounceThreshold)},
        // Global Hair parameters
        {"sig_a", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, hairData.sig_a)},
        {"beta_m", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, hairData.beta_m)},
        {"beta_n", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, hairData.beta_n)},
        {"alpha", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, hairData.alpha)},
        {"R_G", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, hairData.R_G)},
        {"TT_G", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, hairData.TT_G)},
        {"TRT_G", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, hairData.TRT_G)},
        {"TRRT_G", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, hairData.TRRT_G)},
        // All other parameters
        {"accumBuffer", OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams, accumBuffer)},
        {"averageBuffer", OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams, averageBuffer)},
        {"accumId", OWL_INT, OWL_OFFSETOF(LaunchParams, accumId)},
        {"world", OWL_GROUP, OWL_OFFSETOF(LaunchParams, world)},
        {"camera.pos", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.pos)},
        {"camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_00)},
        {"camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_du)},
        {"camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_dv)},
        {nullptr}
    };

    this->launchParams = owlParamsCreate(context, sizeof(LaunchParams), launchParamsDecl, -1);

    // Global hair parameters
    owlParamsSet3f(this->launchParams, "sig_a", owl3f{ this->currentScene.sig_a.x, this->currentScene.sig_a.y, this->currentScene.sig_a.z });
    owlParamsSet1f(this->launchParams, "beta_m", this->currentScene.beta_m);
    owlParamsSet1f(this->launchParams, "beta_n", this->currentScene.beta_n);
    owlParamsSet1f(this->launchParams, "alpha", this->currentScene.alpha);

    owlParamsSet1f(this->launchParams, "R_G", this->currentScene.R_G);
    owlParamsSet1f(this->launchParams, "TT_G", this->currentScene.TT_G);
    owlParamsSet1f(this->launchParams, "TRT_G", this->currentScene.TRT_G);
    owlParamsSet1f(this->launchParams, "TRRT_G", this->currentScene.TRRT_G);

    // Upload NRC variables
    owlParamsSet1b(this->launchParams, "allUnbiased", this->allUnbiased);

    owlParamsSetBuffer(this->launchParams, "tBuffer", this->tBuffer);
    owlParamsSetBuffer(this->launchParams, "gBuffer", this->gBuffer);

    owlParamsSetBuffer(this->launchParams, "nnFrameInput", this->nnFrameInput);
    owlParamsSetBuffer(this->launchParams, "nnFrameOutput", this->nnFrameOutput);

    owlParamsSetBuffer(this->launchParams, "trainIdxs", this->trainIdxs);
    owlParamsSetBuffer(this->launchParams, "trainInput", this->trainInput);
    owlParamsSetBuffer(this->launchParams, "trainGT", this->trainGT);

    owlParamsSet1i(this->launchParams, "mlpInputCh", this->mlpInputCh);
    owlParamsSet1i(this->launchParams, "mlpOutputCh", this->mlpOutputCh);

    owlParamsSet1i(this->launchParams, "numTrainingRecords", this->numTrainingRecords);
    owlParamsSet1i(this->launchParams, "numTrainingPixels", this->numTrainingPixels);
    owlParamsSet1i(this->launchParams, "everyNth", this->everyNth);

    owlParamsSet1f(this->launchParams, "c", this->c);

    // Upload scene properties
    owlParamsSet3f(this->launchParams, "maxBound", owl3f{ this->maxBound.x, this->maxBound.y, this->maxBound.z });
    owlParamsSet3f(this->launchParams, "minBound", owl3f{ this->minBound.x, this->minBound.y, this->minBound.z });
    owlParamsSet1f(this->launchParams, "sceneScale", this->sceneScale);

    // Debug
    owlParamsSet1b(this->launchParams, "showCache", this->showCache);
    owlParamsSet1b(this->launchParams, "showBounces", this->showBounces);
    owlParamsSet1i(this->launchParams, "bounceThreshold", this->bounceThreshold);

    // Upload integrator parameters
    owlParamsSet1b(this->launchParams, "MIS", this->currentScene.MIS);
    owlParamsSet1i(this->launchParams, "accumId", this->accumId);
    owlParamsSet1ul(this->launchParams, "accumBuffer", (uint64_t)this->accumBuffer);
    owlParamsSet1ul(this->launchParams, "averageBuffer", (uint64_t)this->averageBuffer);

    int num_total_lights = 0;

    // Upload environment light
    owlParamsSet1f(this->launchParams, "envRotPhi", this->currentScene.envRotPhi);
    owlParamsSet1f(this->launchParams, "envScale", this->currentScene.envScale);
    owlParamsSet1i(this->launchParams, "envWidth", this->envWidth);
    owlParamsSet1i(this->launchParams, "envHeight", this->envHeight);
    owlParamsSet1b(this->launchParams, "envPdfSampling", this->currentScene.envPdfSampling);
    owlParamsSet1b(this->launchParams, "hasEnvLight", this->currentScene.hasEnvLight);
    if (this->currentScene.hasEnvLight) {
        owlParamsSetTexture(this->launchParams, "env", this->envTextureBuffer);

        owlParamsSetTexture(this->launchParams, "conditionalPdf", this->conditionalPdf);
        owlParamsSetTexture(this->launchParams, "conditionalCdf", this->conditionalCdf);
        owlParamsSetTexture(this->launchParams, "marginalPdf", this->marginalPdf);
        owlParamsSetTexture(this->launchParams, "marginalCdf", this->marginalCdf);

        num_total_lights += 1;
    }

    // Upload directional lights
    owlParamsSet1b(this->launchParams, "hasDirectionalLights", this->currentScene.hasDirectionalLights);
    owlParamsSetBuffer(this->launchParams, "dLights", dLightsBuffer);
    owlParamsSet1i(this->launchParams, "num_dlights", num_dlights);

    // Upload global light parameters
    num_total_lights += num_dlights;
    owlParamsSet1i(this->launchParams, "num_total_lights", num_total_lights);

    // ====================================================
    // Setup a generic miss program
    // ====================================================
    OWLVarDecl missProgVars[] = {
        {"const_color", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, const_color)},
        {nullptr}
    };

    missProg = owlMissProgCreate(context, module, "miss", sizeof(MissProgData), missProgVars, -1);

    // Set a constant background color in the miss program (black for now)
    owlMissProgSet3f(missProg, "const_color", owl3f{ 0.f, 0.f, 0.f });

    // ====================================================
    // Setup a pin-hole camera ray-gen program
    // ====================================================
    OWLVarDecl rayGenVars[] = {
        {"frameBuffer", OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData, frameBuffer)},
        {"frameBufferSize", OWL_INT2, OWL_OFFSETOF(RayGenData, frameBufferSize)},
        {nullptr}
    };

    // Create regular ray gen
    this->rayGen = owlRayGenCreate(context, module, "rayGenCam", sizeof(RayGenData), rayGenVars, -1);

    // Set the TLAS to be used
    owlParamsSetGroup(this->launchParams, "world", world);

    // ====================================================
    // Finally, build the programs, pipeline and SBT
    // ====================================================
    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);

    this->enableInspectMode(owl::box3f(this->minBound, this->maxBound));
    this->setWorldScale(this->sceneScale);
}

long long RenderWindowNRC::render()
{
    auto start = std::chrono::high_resolution_clock::now();

    if (sbtDirty) {
        owlBuildSBT(context);
        sbtDirty = false;
    }

    if (this->accumId < this->currentScene.spp || this->progressive) {
        owlParamsSet1i(this->launchParams, "accumId", this->accumId);

        // Shuffle, to choose random pixels as training points
        int* trainIdxsPtr = (int*)owlBufferGetPointer(this->trainIdxs, 0);
        thrust::device_ptr<int> thrustPtr = thrust::device_pointer_cast(trainIdxsPtr);
        thrust::shuffle(thrustPtr, thrustPtr + this->numTrainingPixels, thrust::default_random_engine());

        owlParamsSet1i(this->launchParams, "pass", G_BUFFER);
        owlLaunch2D(this->rayGen, this->fbSize.x, this->fbSize.y, this->launchParams);

        {
            float* networkInput = (float*)owlBufferGetPointer(this->nnFrameInput, 0);
            float* networkOutput = (float*)owlBufferGetPointer(this->nnFrameOutput, 0);
            this->mlp->inference(networkInput, networkOutput, this->nnFrameSize);
        }

        owlParamsSet1i(this->launchParams, "pass", RENDER);
        owlLaunch2D(this->rayGen, this->fbSize.x, this->fbSize.y, this->launchParams);

        if(!this->showCache && !this->showBounces && this->currentScene.nrcTrain)
        {
            float* trainInputPtr = (float*)owlBufferGetPointer(this->trainInput, 0);
            float* trainGTPtr = (float*)owlBufferGetPointer(this->trainGT, 0);

            GPUMatrix<float> trainInput(trainInputPtr,
                this->mlpInputCh,
                this->numTrainingRecords);
            GPUMatrix<float> trainGT(trainGTPtr,
                this->mlpOutputCh,
                this->numTrainingRecords);

            auto ctx = this->mlp->trainer->training_step(trainInput, trainGT);
            this->trainingLoss = mlp->trainer->loss(*ctx);
        }

        // Clear all training buffers
        owlBufferClear(this->nnFrameInput);
        owlBufferClear(this->nnFrameOutput);
        owlBufferClear(this->trainInput);
        owlBufferClear(this->trainGT);

        owlParamsSet1i(this->launchParams, "pass", RESET);
        owlLaunch2D(this->rayGen, this->fbSize.x, this->fbSize.y, this->launchParams);

        if (!this->showCache && !this->showBounces)
            accumId++;
    }

    auto finish = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
}

void RenderWindowNRC::resize(const vec2i& newSize)
{
    if (this->getWindowSize().x == 0 && this->getWindowSize().y == 0) {
        OWLViewer::resize(newSize);

        this->initialize();
        this->cameraChanged();
    }
}

void RenderWindowNRC::drawUI()
{
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    {
        ImGui::Begin("Controls", NULL, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
        ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        bool prog = this->progressive;
        ImGui::Checkbox("Progressive", &prog);
        if (prog != this->progressive) {
            this->progressive = prog;
            this->cameraChanged();
        }

        bool den = this->currentScene.denoise;
        ImGui::Checkbox("Denoise", &den);
        if (den != this->currentScene.denoise) {
            this->currentScene.denoise = den;
            this->cameraChanged();
        }

        if (!this->progressive) {
            int currentSpp = this->currentScene.spp;
            ImGui::SliderInt("SPP", &currentSpp, 0, 500);
            if (currentSpp != this->currentScene.spp) {
                this->currentScene.spp = currentSpp;
                this->cameraChanged();
            }
        }

        bool train = this->currentScene.nrcTrain;
        ImGui::Checkbox("Train NRC", &train);
        if (train != this->currentScene.nrcTrain) {
            this->setNRCTrain(train);
        }

        bool unbiased = this->allUnbiased;
        ImGui::Checkbox("Set all training paths - unbiased", &unbiased);
        if (unbiased != this->allUnbiased) {
            this->setAllUnbiasedPaths(unbiased);
        }

        float c_ = this->c;
        ImGui::SliderFloat("C", &c_, 0.f, 1.f);
        if (c_ != this->c) {
            this->setBounceThresholdControl(c_);
        }

        bool cache = this->showCache;
        ImGui::Checkbox("Show Cache", &cache);
        if (cache != this->showCache) {
            this->showCacheVisualization(cache);
        }

        bool bounces = this->showBounces;
        ImGui::Checkbox("Show Bounces", &bounces);
        if (bounces != this->showBounces) {
            this->showBounceVisualization(bounces, this->bounceThreshold);
        }

        int bt = this->bounceThreshold;
        ImGui::SliderInt("Bounce Vis. Thresh.", &bt, 0, MAX_BOUNCES);
        if (bt != this->bounceThreshold) {
            this->showBounceVisualization(bounces, bt);
        }

        // Sampling controls
        if (ImGui::CollapsingHeader("Sampling")) {
            bool currentMIS = this->currentScene.MIS;
            ImGui::Checkbox("MIS", &currentMIS);
            if (currentMIS != this->currentScene.MIS) {
                this->currentScene.MIS = currentMIS;
                owlParamsSet1b(this->launchParams, "MIS", currentMIS);
                this->cameraChanged();
            }

            bool currentEnvSampling = this->currentScene.envPdfSampling;
            ImGui::Checkbox("Env. PDF Samp.", &currentEnvSampling);
            if (currentEnvSampling != this->currentScene.envPdfSampling) {
                this->currentScene.envPdfSampling = currentEnvSampling;
                owlParamsSet1b(this->launchParams, "envPdfSampling", currentEnvSampling);
                this->cameraChanged();
            }
        }

        // All hair controls
        if (ImGui::CollapsingHeader("Hair")) {
            float cur_sig_a[3] = { this->currentScene.sig_a.x, this->currentScene.sig_a.y,
                                                this->currentScene.sig_a.z };
            ImGui::SliderFloat3("Sigma", cur_sig_a, 0.f, 10.f);
            if (this->currentScene.sig_a.x != cur_sig_a[0] ||
                this->currentScene.sig_a.y != cur_sig_a[1] ||
                this->currentScene.sig_a.z != cur_sig_a[2]) {
                this->currentScene.sig_a = vec3f(cur_sig_a[0], cur_sig_a[1], cur_sig_a[2]);
                owlParamsSet3f(this->launchParams, "sig_a", owl3f{ cur_sig_a[0], cur_sig_a[1],
                                                                cur_sig_a[2] });
                this->cameraChanged();
            }

            float cur_beta_m = this->currentScene.beta_m;
            ImGui::SliderFloat("beta_m", &cur_beta_m, 0.f, 1.f);
            if (cur_beta_m != this->currentScene.beta_m) {
                this->currentScene.beta_m = cur_beta_m;
                owlParamsSet1f(this->launchParams, "beta_m", cur_beta_m);
                this->cameraChanged();
            }

            float cur_beta_n = this->currentScene.beta_n;
            ImGui::SliderFloat("beta_n", &cur_beta_n, 0.f, 1.f);
            if (cur_beta_n != this->currentScene.beta_n) {
                this->currentScene.beta_n = cur_beta_n;
                owlParamsSet1f(this->launchParams, "beta_n", cur_beta_n);
                this->cameraChanged();
            }

            float cur_alpha = this->currentScene.alpha;
            ImGui::SliderFloat("alpha", &cur_alpha, 0.f, 20.f * 3.14159f / 180.f);
            if (cur_alpha != this->currentScene.alpha) {
                this->currentScene.alpha = cur_alpha;
                owlParamsSet1f(this->launchParams, "alpha", cur_alpha);
                this->cameraChanged();
            }

            float rg = this->currentScene.R_G;
            ImGui::SliderFloat("R Gain", &rg, 0.f, 1.f);
            if (rg != this->currentScene.R_G) {
                this->currentScene.R_G = rg;
                owlParamsSet1f(this->launchParams, "R_G", rg);
                this->cameraChanged();
            }

            float ttg = this->currentScene.TT_G;
            ImGui::SliderFloat("TT Gain", &ttg, 0.f, 1.f);
            if (ttg != this->currentScene.TT_G) {
                this->currentScene.TT_G = ttg;
                owlParamsSet1f(this->launchParams, "TT_G", ttg);
                this->cameraChanged();
            }

            float trtg = this->currentScene.TRT_G;
            ImGui::SliderFloat("TRT Gain", &trtg, 0.f, 1.f);
            if (trtg != this->currentScene.TRT_G) {
                this->currentScene.TRT_G = trtg;
                owlParamsSet1f(this->launchParams, "TRT_G", trtg);
                this->cameraChanged();
            }

            float trrtg = this->currentScene.TRRT_G;
            ImGui::SliderFloat("TRRT Gain", &trrtg, 0.f, 1.f);
            if (trrtg != this->currentScene.TRRT_G) {
                this->currentScene.TRRT_G = trrtg;
                owlParamsSet1f(this->launchParams, "TRRT_G", trrtg);
                this->cameraChanged();
            }
        }

        // All lighting controls
        if (ImGui::CollapsingHeader("Light")) {

            if (ImGui::TreeNode("Environment Light")) {
                // Controls for environment light
                if (this->currentScene.hasEnvLight) {
                    float scale = this->currentScene.envScale;
                    ImGui::SliderFloat("Radiance Scale", &scale, 0.f, 100.f);
                    if (scale != this->currentScene.envScale) {
                        this->currentScene.envScale = scale;
                        owlParamsSet1f(this->launchParams, "envScale", scale);
                        this->cameraChanged();
                    }

                    float rot = this->currentScene.envRotPhi;
                    ImGui::SliderFloat("Rotation", &rot, 0.f, 3.14159f * 2.f);
                    if (rot != this->currentScene.envRotPhi) {
                        this->currentScene.envRotPhi = rot;
                        owlParamsSet1f(this->launchParams, "envRotPhi", rot);
                        this->cameraChanged();
                    }
                }

                ImGui::TreePop();
            }

            if (ImGui::TreeNode("Directional Lights")) {
                // Controls for directional lights
                for (int i = 0; i < this->num_dlights; i++) {
                    float dLightFrom[3] = {
                        this->dLightList[i].from.x,
                        this->dLightList[i].from.y,
                        this->dLightList[i].from.z
                    };
                    std::string sname = "Dir " + std::to_string(i + 1);
                    ImGui::SliderFloat3(sname.c_str(), dLightFrom, -1.f, 1.f);

                    if (dLightFrom[0] != this->dLightList[i].from.x ||
                        dLightFrom[1] != this->dLightList[i].from.y ||
                        dLightFrom[2] != this->dLightList[i].from.z) {

                        this->dLightList[i].from = vec3f(dLightFrom[0], dLightFrom[1], dLightFrom[2]);
                        this->uploadLights();
                    }

                    float emit = this->dLightList[i].emit.x;
                    ImGui::SliderFloat("Emit", &emit, 0.f, 100.f);
                    if (emit != this->dLightList[i].emit.x) {
                        this->dLightList[i].emit = vec3f(emit);
                        this->uploadLights();
                    }
                }

                ImGui::TreePop();
            }
        }

        // Controls specific to path integrators
        if (ImGui::CollapsingHeader("Path tracing")) {
            std::string text = "Bounces set to " + std::to_string(MAX_BOUNCES);
            ImGui::Text(text.c_str());
        }

        if (ImGui::Button("Save PNG")) {
            this->screenShot(this->currentScene.renderOutput);
        }

        if (ImGui::Button("Save EXR")) {
            std::string exrFile = this->currentScene.renderOutput;
            size_t pos = exrFile.find(".png");
            exrFile.replace(pos, 4, ".exr");

            this->screenShotEXR(exrFile);
        }

        ImGui::End();
    }

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
}

// ====================================================
// Mostly constant functions
// ====================================================
void RenderWindowNRC::screenShotEXR(std::string fname)
{
    int bufferSize = this->fbSize.x * this->fbSize.y * 4 * sizeof(float);
    float* hostBuffer = (float*)malloc(bufferSize);
    cudaMemcpy(hostBuffer, this->averageBuffer, bufferSize, cudaMemcpyDeviceToHost);
    saveEXR(fname, hostBuffer, this->accumId, this->fbSize.x, this->fbSize.y);
    free(hostBuffer);
}

RenderWindowNRC::RenderWindowNRC(Scene& scene, vec2i resolution, bool interactive)
    : owl::viewer::OWLViewer("Real-time hair", resolution, interactive, false)
{
    this->currentScene = scene;

    this->camera.setOrientation(scene.camera.from,
        scene.camera.at,
        scene.camera.up,
        owl::viewer::toDegrees(acosf(scene.camera.cosFovy)));

    if(interactive)
        this->enableFlyMode();

    // Initialize IMGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(this->handle, true);
    ImGui_ImplOpenGL2_Init();

    // Context & Module creation, accumulation buffer initialize
    context = owlContextCreate(nullptr, 1);
    module = owlModuleCreate(context, nrc_ptx);

    owlContextSetRayTypeCount(context, NUM_RAY_TYPES);
    owlEnableCurves(context); // Required since hair strands are represented by curves
}

void RenderWindowNRC::customKey(char key, const vec2i& pos)
{
    if (key == '1' || key == '!') {
        this->camera.setOrientation(this->camera.getFrom(), vec3f(0.f), vec3f(0.f, 0.f, 1.f), this->camera.getFovyInDegrees());
        this->cameraChanged();
    }
    else if (key == 'R' || key == 'r') {
        SceneCamera cam;
        cam.from = this->camera.getFrom();
        cam.at = this->camera.getAt();
        cam.up = this->camera.getUp();
        cam.cosFovy = this->camera.getCosFovy();

        nlohmann::json oneCameraJson;
        std::vector<float> from, at, up;

        for (int i = 0; i < 3; i++) {
            from.push_back(cam.from[i]);
            at.push_back(cam.at[i]);
            up.push_back(cam.up[i]);
        }

        oneCameraJson["from"] = from;
        oneCameraJson["to"] = at;
        oneCameraJson["up"] = up;
        oneCameraJson["cos_fovy"] = cam.cosFovy;

        this->currentScene.json["camera"] = oneCameraJson;
        std::ofstream outputFile(this->currentScene.jsonFilePath);
        outputFile << std::setw(4) << this->currentScene.json << std::endl;
    }
}

void RenderWindowNRC::cameraChanged()
{
    // Reset accumulation buffer, to restart MC sampling
    this->accumId = 0;

    const vec3f lookFrom = camera.getFrom();
    const vec3f lookAt = camera.getAt();
    const vec3f lookUp = camera.getUp();
    const float cosFovy = camera.getCosFovy();

    // ----------- compute variable values  ------------------
    vec3f camera_pos = lookFrom;
    vec3f camera_d00
        = normalize(lookAt - lookFrom);
    float aspect = fbSize.x / float(fbSize.y);
    vec3f camera_ddu
        = cosFovy * aspect * normalize(cross(camera_d00, lookUp));
    vec3f camera_ddv
        = cosFovy * normalize(cross(camera_ddu, camera_d00));
    camera_d00 -= 0.5f * camera_ddu;
    camera_d00 -= 0.5f * camera_ddv;

    // ----------- set variables  ----------------------------
    owlRayGenSet1ul(this->rayGen, "frameBuffer", (uint64_t)this->fbPointer);
    owlRayGenSet2i(this->rayGen, "frameBufferSize", (const owl2i&)this->fbSize);

    owlParamsSet3f(this->launchParams, "camera.pos", (const owl3f&)camera_pos);
    owlParamsSet3f(this->launchParams, "camera.dir_00", (const owl3f&)camera_d00);
    owlParamsSet3f(this->launchParams, "camera.dir_du", (const owl3f&)camera_ddu);
    owlParamsSet3f(this->launchParams, "camera.dir_dv", (const owl3f&)camera_ddv);

    this->sbtDirty = true;
}

void RenderWindowNRC::uploadLights()
{
    // Directional lights
    this->dLightsBuffer = owlDeviceBufferCreate(context,
        OWL_USER_TYPE(DirectionalLight), this->dLightList.size(), this->dLightList.data());

    // Upload directional lights
    owlParamsSetBuffer(this->launchParams, "dLights", dLightsBuffer);
    owlParamsSet1i(this->launchParams, "num_dlights", num_dlights);

    this->cameraChanged();
}

// ====================================================
// Entry point 
// ====================================================
int main(int argc, char** argv)
{
    std::string renderer = "Neural Radiance Caching";

    std::string currentScene = "C:/Users/Projects/HairMSNN/scenes/curly/config.json";
    if (argc >= 2)
        currentScene = std::string(argv[1]);

    LOG("Loading scene " + currentScene);

    Scene scene;
    bool success = parseScene(currentScene, scene);
    if (!success) {
        LOG("Error loading scene");
        return -1;
    }

    vec2i resolution(scene.imgWidth, scene.imgHeight);
    RenderWindowNRC win(scene, resolution, true);
    win.setTitle(renderer);

    win.showAndRun();

    return 0;
}
