// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#include "hair_msnn.cuh"
#include "utils.cuh"

#include "curve_utils.cuh"
#include "disney_hair.cuh"
#include "frostbite_anisotropic.cuh"

#include "optix_common.cuh"

__device__
vec3f msnnNextPathVertex(Interaction& si, LCGRand& rng)
{
    vec2f rand1 = vec2f(lcg_randomf(rng), lcg_randomf(rng));

    float nextPdf = 1.f;
    vec3f nextBsdf(0.f);

    if (!si.isSurface) {
        vec4f rand2 = vec4f(rand1.x, rand1.y, lcg_randomf(rng), lcg_randomf(rng));
        nextBsdf = sample_disney_hair(si, rand2, &nextPdf);
    }
    else {
        si.wi_local = sample_GGX(rand1, si.alpha, si.wo_local,
            &nextPdf);
        si.wi = normalize(apply_mat(si.to_world, si.wi_local));

        nextBsdf = frostbite_GGX(si.wo_local, si.wi_local, si.color, si.alpha);
    }

    Interaction prevSi = si;

    si.hit = false;
    si.wo = -si.wi; // Direction is outward

    float wiDotN = dot(si.wi, si.n);
    bool lowerHemi = wiDotN < 0.f;
    vec3f nd = si.n;
    if (lowerHemi && !si.isSurface) {
        nd = -si.n;
        si.p = si.p + 2.f * si.hair.radius * nd;
    }

    vec3f mulFac(0.f);
    if (nextPdf == 0.f)
        mulFac = nextBsdf;
    else
        mulFac = nextBsdf / nextPdf;

    if (isnan(mulFac.x) || isnan(mulFac.y) || isnan(mulFac.z))
        mulFac = vec3f(1.f);

    RadianceRay nextRay;
    nextRay.origin = si.p + 1e-3f * nd;
    nextRay.direction = -si.wo; // On the other hand, ray direction should point correctly!
    owl::traceRay(optixLaunchParams.world, nextRay, si);

    return mulFac;
}

__device__
vec3f msnnTrainingPath(Interaction si, LCGRand& rng, vec3f& shortPathColor)
{
    vec3f beta(1.f), betaShortPath(1.f);
    vec3f final_color(0.f);
    float indirectFactor = 1.f;
    bool isPrevHair = !si.isSurface;

    final_color = directLighting(optixLaunchParams, si, rng);
    shortPathColor = final_color;

    int bounces = 1;
    for (bounces = 1; bounces < optixLaunchParams.pathV2; bounces++) {
        /* ================================================
        Next vertex in the path
        ================================================ */
        isPrevHair = !si.isSurface;

        vec3f mulFac(0.f);
        mulFac = msnnNextPathVertex(si, rng);

        // Update betas here
        beta = beta * mulFac;
        betaShortPath = betaShortPath * mulFac;

        /* ================================================
        Terminate if escaped
        ================================================ */
        if (si.hit == false) {
            break;
        }

        /* ================================================
        Direct lighting
        ================================================ */
        vec3f dl = directLighting(optixLaunchParams, si, rng);
        
        final_color += beta * dl;
        shortPathColor += indirectFactor * betaShortPath * dl;

        /* ================================================
        Russian Roulette path termination (from PBRT)
        ================================================ */
        float q = max(0.05f, 1.f - luminance(beta));
        float qShortPath = max(0.05f, 1.f - luminance(betaShortPath));
        float eps = lcg_randomf(rng);

        bool cond1 = eps < qShortPath;
        bool cond2 = bounces > optixLaunchParams.beta;

        if (cond1 || cond2) {
            betaShortPath = 0.f;
        }

        if (eps < q) {
            break;
        }

        beta = beta / (1.f - q);

        if(betaShortPath != vec3f(0.f))
            betaShortPath = betaShortPath / (1.f - qShortPath);
    }

    return final_color;
}

__device__
vec3f msnnPathTrace(Interaction si, LCGRand& rng, int v2Stop)
{
    vec3f beta(1.f);
    vec3f final_color(0.f);
    float indirectFactor = 1.f;
    bool isPrevHair = !si.isSurface;

    final_color = directLighting(optixLaunchParams, si, rng);

    int bounces = 1;
    for (bounces = 1; bounces <= v2Stop; bounces++) {
        /* ================================================
        Next vertex in the path
        ================================================ */
        isPrevHair = !si.isSurface;
        vec3f mulFac(0.f);
        mulFac = msnnNextPathVertex(si, rng);

        // Update beta with short path factor
        beta = beta * mulFac;

        /* ================================================
        Terminate if escaped
        ================================================ */
        if (si.hit == false) {
            break;
        }

        /* ================================================
        Direct lighting
        ================================================ */
        final_color += indirectFactor * beta * directLighting(optixLaunchParams, si, rng);

        /* ================================================
        Russian Roulette path termination (from PBRT)
        ================================================ */
        float q = max(0.05f, 1.f - luminance(beta));
        float eps = lcg_randomf(rng);

        bool cond1 = eps < q;
        bool cond2 = bounces > optixLaunchParams.beta;

        if (cond1 || cond2) {
            break;
        }

        beta = beta / (1.f - q);
    }

    return final_color;
}

OPTIX_RAYGEN_PROGRAM(rayGenCam)()
{
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const vec2i pixelId = owl::getLaunchIndex();
    int fbOfs = pixelId.x + self.frameBufferSize.x * pixelId.y;

    // Pseudo-random number generator
    LCGRand rng = get_rng(optixLaunchParams.accumId + 10007, make_uint2(pixelId.x, pixelId.y),
        make_uint2(self.frameBufferSize.x, self.frameBufferSize.y));

    int inputCh = optixLaunchParams.mlpInputCh;
    int outputCh = optixLaunchParams.mlpOutputCh;

    int pathV1 = optixLaunchParams.pathV1 - 1;
    int pathV2 = optixLaunchParams.pathV2 - 1;

    int trOfs = 0;
    bool isTrainingPixel = false;
    RadianceRay ray;
    if (optixLaunchParams.pass == G_BUFFER) {
        trOfs = fbOfs / optixLaunchParams.everyNth;
        int trainIdx = optixLaunchParams.trainIdxs[trOfs] % optixLaunchParams.everyNth;
        isTrainingPixel = fbOfs % optixLaunchParams.everyNth == trainIdx;

        // Shoot camera ray
        vec2f pixelOffset = vec2f(lcg_randomf(rng), lcg_randomf(rng));
        const vec2f screen = (vec2f(pixelId) + pixelOffset) / vec2f(self.frameBufferSize);

        ray.origin
            = optixLaunchParams.camera.pos;
        ray.direction
            = normalize(optixLaunchParams.camera.dir_00
                + screen.u * optixLaunchParams.camera.dir_du
                + screen.v * optixLaunchParams.camera.dir_dv);
    }
    else if (optixLaunchParams.pass == TRAIN_DATA_GEN) {
        trOfs = pixelId.x + optixLaunchParams.numTrainRecordsX * pixelId.y;
        isTrainingPixel = true;

        int sceneIdx = optixLaunchParams.sceneIndices[trOfs];
        vec3f sPoint = optixLaunchParams.sampledPoints[sceneIdx];

        ray.origin
            = optixLaunchParams.camera.pos;
        ray.direction
            = normalize(sPoint - ray.origin);
    }
    
    if (optixLaunchParams.pass == G_BUFFER || optixLaunchParams.pass == TRAIN_DATA_GEN) {
        Interaction si;
        si.hit = false;
        si.wo = -1.f * ray.direction;
        si.wi = ray.direction;
        owl::traceRay(optixLaunchParams.world, ray, si);

        vec3f color(0.f), shortPathColor(0.f);
        if (isTrainingPixel) {
            if (si.hit) {
                color = msnnTrainingPath(si, rng, shortPathColor);
            }

            if (isnan(color.x) || isnan(color.y) || isnan(color.z))
                color = vec3f(0.f);

            if (isinf(color.x) || isinf(color.y) || isinf(color.z))
                color = vec3f(1e5f);

            if (isnan(shortPathColor.x) || isnan(shortPathColor.y) || isnan(shortPathColor.z))
                shortPathColor = vec3f(0.01f);

            if (isinf(shortPathColor.x) || isinf(shortPathColor.y) || isinf(shortPathColor.z))
                shortPathColor = vec3f(1e5f);

            vec3f point = si.p / optixLaunchParams.sceneScale;
            optixLaunchParams.nnTrainInput[trOfs * inputCh + 0] = point.x;
            optixLaunchParams.nnTrainInput[trOfs * inputCh + 1] = point.y;
            optixLaunchParams.nnTrainInput[trOfs * inputCh + 2] = point.z;

            optixLaunchParams.nnTrainInput[trOfs * inputCh + 3] = si.wo.x;
            optixLaunchParams.nnTrainInput[trOfs * inputCh + 4] = si.wo.y;
            optixLaunchParams.nnTrainInput[trOfs * inputCh + 5] = si.wo.z;

            optixLaunchParams.nnTrainInput[trOfs * inputCh + 6] = si.t.x;
            optixLaunchParams.nnTrainInput[trOfs * inputCh + 7] = si.t.y;
            optixLaunchParams.nnTrainInput[trOfs * inputCh + 8] = si.t.z;

            optixLaunchParams.nnTrainOutput[trOfs * outputCh + 0] = color.x - shortPathColor.x;
            optixLaunchParams.nnTrainOutput[trOfs * outputCh + 1] = color.y - shortPathColor.y;
            optixLaunchParams.nnTrainOutput[trOfs * outputCh + 2] = color.z - shortPathColor.z;

            if (!si.hit) {
                color = si.Le;
            }
        }
        else {
            if (!si.hit) {
                color = si.Le;
            }
            else if (si.hit) {
                color = msnnPathTrace(si, rng, pathV2);
            }
        }

        if (optixLaunchParams.pass == G_BUFFER) {
            vec3f point = si.p / optixLaunchParams.sceneScale;

            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 0] = point.x;
            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 1] = point.y;
            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 2] = point.z;

            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 3] = si.wo.x;
            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 4] = si.wo.y;
            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 5] = si.wo.z;

            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 6] = si.t.x;
            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 7] = si.t.y;
            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 8] = si.t.z;

            GBuffer buf;
            buf.hit = si.hit;
            buf.isSurface = si.isSurface;
            buf.p = si.p;
            buf.shortPathColor = color;

            optixLaunchParams.gBuffer[fbOfs] = buf;
        }
    }
    else if (optixLaunchParams.pass == RENDER) {
        // Read GBuffer
        GBuffer gBuffer = optixLaunchParams.gBuffer[fbOfs];

        // NRC
        vec3f nnOutput(0.f);
        nnOutput.x = optixLaunchParams.nnFrameOutput[fbOfs * outputCh + 0];
        nnOutput.y = optixLaunchParams.nnFrameOutput[fbOfs * outputCh + 1];
        nnOutput.z = optixLaunchParams.nnFrameOutput[fbOfs * outputCh + 2];

        vec3f color(0.f);
        if (!gBuffer.hit || gBuffer.isSurface) {
            // Escaped or hit surface
            color = gBuffer.shortPathColor;
            nnOutput = color;
        }
        else {
            color = gBuffer.shortPathColor + nnOutput;
        }

        // Write final color
        if (optixLaunchParams.accumId > 0) {
            gBuffer.shortPathColor = gBuffer.shortPathColor + vec3f(optixLaunchParams.ptAccumBuffer[fbOfs]);
            nnOutput = nnOutput + vec3f(optixLaunchParams.nnAccumBuffer[fbOfs]);
            color = color + vec3f(optixLaunchParams.finalAccumBuffer[fbOfs]);
        }

        optixLaunchParams.ptAccumBuffer[fbOfs] = vec4f(gBuffer.shortPathColor, 1.f);
        optixLaunchParams.nnAccumBuffer[fbOfs] = vec4f(nnOutput, 1.f);
        optixLaunchParams.finalAccumBuffer[fbOfs] = vec4f(color, 1.f);

        gBuffer.shortPathColor = (1.f / (optixLaunchParams.accumId + 1)) * gBuffer.shortPathColor;
        nnOutput = (1.f / (optixLaunchParams.accumId + 1)) * nnOutput;
        color = (1.f / (optixLaunchParams.accumId + 1)) * color;

        optixLaunchParams.ptAverageBuffer[fbOfs] = vec4f(gBuffer.shortPathColor, 1.f);
        optixLaunchParams.nnAverageBuffer[fbOfs] = vec4f(nnOutput, 1.f);
        optixLaunchParams.finalAverageBuffer[fbOfs] = vec4f(color, 1.f);

        self.frameBuffer[fbOfs] = owl::make_rgba(vec3f(linear_to_srgb(color.x),
            linear_to_srgb(color.y),
            linear_to_srgb(color.z)));
    }
}