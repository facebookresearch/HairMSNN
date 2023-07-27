// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#include "render_path_tracing.h"
#include "utils.cuh"

// ====================================================
// Custom functions
// ====================================================
vec3f RenderWindowPT::getSceneCg()
{
    return (this->minBound + this->maxBound) / 2.f;
}

void RenderWindowPT::initialize()
{
    vec2i frameRes(this->getWindowSize().x, this->getWindowSize().y);
    int frameSize = frameRes.x * frameRes.y;

    // Accumulation buffer stores the non-averaged radiance in float32
    // Float buffer stores the averaged radiance in float32
    // Frame buffer (defined in OWL::Viewer) stores averaged radiance in uint8
    cudaMalloc(&this->accumBuffer, frameSize * sizeof(float4));
    cudaMalloc(&this->averageBuffer, frameSize * sizeof(float4));

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
            owlGeomTypeSetAnyHit(triangleGeomType, MULTISCATTER_RAY_TYPE, module, "triangleMeshAHMultiScatter");

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
    owlGeomTypeSetAnyHit(hairGeomType, MULTISCATTER_RAY_TYPE, module, "hairAHMultiScatter");
    owlCurvesSetDegree(hairGeomType, 3, true);

    // ====================================================
    // Data setup (hair strands from .hair file)
    // ====================================================

    this->hairGeom = owlGeomCreate(context, hairGeomType);

    if(this->currentScene.has_hair) {
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
        {"pathV1", OWL_INT, OWL_OFFSETOF(LaunchParams, pathV1)},
        {"pathV2", OWL_INT, OWL_OFFSETOF(LaunchParams, pathV2)},
        {"MIS", OWL_BOOL, OWL_OFFSETOF(LaunchParams, MIS)},
        {"envPdfSampling", OWL_BOOL, OWL_OFFSETOF(LaunchParams, envPdfSampling)},
        // Scene properties
        {"maxBound", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, maxBound)},
        {"minBound", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, minBound)},
        {"sceneScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, sceneScale)},
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

    // Upload scene properties
    owlParamsSet3f(this->launchParams, "maxBound", owl3f{ this->maxBound.x, this->maxBound.y, this->maxBound.z });
    owlParamsSet3f(this->launchParams, "minBound", owl3f{ this->minBound.x, this->minBound.y, this->minBound.z });
    owlParamsSet1f(this->launchParams, "sceneScale", this->sceneScale);

    // Upload integrator parameters
    owlParamsSet1b(this->launchParams, "MIS", this->currentScene.MIS);
    owlParamsSet1i(this->launchParams, "pathV1", this->currentScene.path_v1);
    owlParamsSet1i(this->launchParams, "pathV2", this->currentScene.path_v2);
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

long long RenderWindowPT::render()
{
    auto start = std::chrono::high_resolution_clock::now();

    if (sbtDirty) {
        owlBuildSBT(context);
        sbtDirty = false;
    }

    if (this->accumId < this->currentScene.spp || this->progressive) {
        owlParamsSet1i(this->launchParams, "accumId", this->accumId);
        owlLaunch2D(this->rayGen, this->fbSize.x, this->fbSize.y, this->launchParams);

        accumId++;
    }

    auto finish = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
}

void RenderWindowPT::resize(const vec2i& newSize)
{
    if (this->getWindowSize().x == 0 && this->getWindowSize().y == 0) {
        OWLViewer::resize(newSize);

        this->initialize();
        this->cameraChanged();
    }
}

void RenderWindowPT::drawUI()
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

        if (!this->progressive) {
            int currentSpp = this->currentScene.spp;
            ImGui::SliderInt("SPP", &currentSpp, 1, 100);
            if (currentSpp != this->currentScene.spp) {
                this->currentScene.spp = currentSpp;
                this->cameraChanged();
            }
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

                    float emit[3] = {
                        this->dLightList[i].emit.x,
                        this->dLightList[i].emit.y,
                        this->dLightList[i].emit.z
                    };
                    std::string ename = "Emit " + std::to_string(i + 1);
                    ImGui::SliderFloat3(ename.c_str(), emit, 0.f, 10.f);
                    if (emit[0] != this->dLightList[i].emit.x ||
                        emit[1] != this->dLightList[i].emit.y ||
                        emit[2] != this->dLightList[i].emit.z) {
                        this->dLightList[i].emit = vec3f(emit[0], emit[1], emit[2]);
                        this->uploadLights();
                    }
                }

                ImGui::TreePop();
            }
        }

        // Controls specific to path integrators
        if (ImGui::CollapsingHeader("Path tracing")) {
            int v2 = this->currentScene.path_v2;
            ImGui::SliderInt("Path Depth v2", &v2, 1, 50);

            int v1 = this->currentScene.path_v1;
            ImGui::SliderInt("Path Depth v1", &v1, 1, v2);

            if (v1 != this->currentScene.path_v1 || v2 != this->currentScene.path_v2) {
                this->currentScene.path_v1 = v1;
                this->currentScene.path_v2 = v2;

                owlParamsSet1i(this->launchParams, "pathV1", v1);
                owlParamsSet1i(this->launchParams, "pathV2", v2);

                this->cameraChanged();
            }
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
void RenderWindowPT::screenShotEXR(std::string fname)
{
    int bufferSize = this->fbSize.x * this->fbSize.y * 4 * sizeof(float);
    float* hostBuffer = (float*)malloc(bufferSize);
    cudaMemcpy(hostBuffer, this->averageBuffer, bufferSize, cudaMemcpyDeviceToHost);
    saveEXR(fname, hostBuffer, this->accumId, this->fbSize.x, this->fbSize.y);
    free(hostBuffer);
}

RenderWindowPT::RenderWindowPT(Scene& scene, vec2i resolution, bool interactive)
    : owl::viewer::OWLViewer("Real-time hair", resolution, interactive, false)
{
    this->currentScene = scene;

    this->camera.setOrientation(scene.camera.from,
        scene.camera.at,
        scene.camera.up,
        owl::viewer::toDegrees(acosf(scene.camera.cosFovy)));

    if (interactive)
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
    module = owlModuleCreate(context, path_tracing_ptx);

    owlContextSetRayTypeCount(context, NUM_RAY_TYPES);
    owlEnableCurves(context); // Required since hair strands are represented by curves
}

void RenderWindowPT::customKey(char key, const vec2i& pos)
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
    else if (key == 'P' || key == 'p') {
        SceneCamera cam;
        cam.from = this->camera.getFrom();
        cam.at = this->camera.getAt();
        cam.up = this->camera.getUp();
        cam.cosFovy = this->camera.getCosFovy();

        this->currentScene.cameraPath.push_back(cam);
    }
    else if (key == '[' || key == '{') {
        this->currentScene.cameraPath.clear();
    }
    else if (key == 'o' || key == 'O') {
        std::vector<nlohmann::json> cameraPathJson;
        
        for (auto cam : this->currentScene.cameraPath) {
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

            cameraPathJson.push_back(oneCameraJson);
        }

        this->currentScene.json["camera_path"] = cameraPathJson;
        std::ofstream outputFile(this->currentScene.jsonFilePath);
        outputFile << std::setw(4) << this->currentScene.json << std::endl;
    }
}

void RenderWindowPT::cameraChanged()
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

void RenderWindowPT::uploadLights()
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
    std::string renderer = "Path Tracing";

    std::string currentScene = "C:/Users/Projects/HairMSNN/scenes/straight/config.json";
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
    RenderWindowPT win(scene, resolution, true);
    win.setTitle(renderer);

    win.showAndRun();

    return 0;
}
