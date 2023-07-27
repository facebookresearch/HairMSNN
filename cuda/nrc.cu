// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#include "nrc.cuh"
#include "utils.cuh"

#include "curve_utils.cuh"
#include "disney_hair.cuh"
#include "frostbite_anisotropic.cuh"

#include "optix_common.cuh"
#include <optix_device.h>

__device__
float nextPathVertexNRC(Interaction& si, LCGRand& rng, vec3f& beta)
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

    // Update the beta (accumulation over path vertices)
    vec3f mulFac(0.f);
    if (nextPdf == 0.f)
        mulFac = nextBsdf;
    else
        mulFac = nextBsdf / nextPdf;

    if (isnan(mulFac.x) || isnan(mulFac.y) || isnan(mulFac.z))
        mulFac = vec3f(1.f);

    beta = beta * mulFac;

    si.hit = false;
    si.wo = -si.wi; // Direction is outward

    float wiDotN = dot(si.wi, si.n);
    bool lowerHemi = wiDotN < 0.f;
    vec3f nd = si.n;
    if (lowerHemi && !si.isSurface) {
        nd = -si.n;
        si.p = si.p + 2.f * si.hair.radius * nd;
    }

    RadianceRay nextRay;
    nextRay.origin = si.p + 1e-3f * nd;
    nextRay.direction = -si.wo; // On the other hand, ray direction should point correctly!
    owl::traceRay(optixLaunchParams.world, nextRay, si);

    return nextPdf;
}

__device__
void nrcGenerateTrainingData(int trOfs, GBuffer gBuf, bool isTrainingPixel, bool isUnbiasedPixel)
{
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const vec2i pixelId = owl::getLaunchIndex();
    int fbOfs = pixelId.x + self.frameBufferSize.x * pixelId.y;
    int frameSize = self.frameBufferSize.x * self.frameBufferSize.y;

    int inputCh = optixLaunchParams.mlpInputCh;
    int outputCh = optixLaunchParams.mlpOutputCh;

    if (isTrainingPixel && gBuf.hit == true) {
        vec3f cacheRadiance(0.f);
        if(optixLaunchParams.tBuffer[trOfs].hit == true && !isUnbiasedPixel)
            cacheRadiance = (vec3f)optixLaunchParams.nnFrameOutput[frameSize + trOfs];

        for (int bounce = 0; bounce < optixLaunchParams.tBuffer[trOfs].bounces; bounce++) {
            vec3f color(0.f), beta(1.f);

            for (int sub = bounce; sub < optixLaunchParams.tBuffer[trOfs].bounces; sub++) {
                color = color + beta * optixLaunchParams.tBuffer[trOfs].vertRadiance[sub];
                beta = beta * optixLaunchParams.tBuffer[trOfs].vertBeta[sub];
            }

            int bounceOffset = trOfs * inputCh * MAX_BOUNCES + bounce * inputCh;

            optixLaunchParams.trainInput[bounceOffset + 0] = optixLaunchParams.tBuffer[trOfs].vert[bounce].x;
            optixLaunchParams.trainInput[bounceOffset + 1] = optixLaunchParams.tBuffer[trOfs].vert[bounce].y;
            optixLaunchParams.trainInput[bounceOffset + 2] = optixLaunchParams.tBuffer[trOfs].vert[bounce].z;
            
            optixLaunchParams.trainInput[bounceOffset + 3] = optixLaunchParams.tBuffer[trOfs].wo[bounce].x;
            optixLaunchParams.trainInput[bounceOffset + 4] = optixLaunchParams.tBuffer[trOfs].wo[bounce].y;
            optixLaunchParams.trainInput[bounceOffset + 5] = optixLaunchParams.tBuffer[trOfs].wo[bounce].z;
            
            optixLaunchParams.trainInput[bounceOffset + 6] = optixLaunchParams.tBuffer[trOfs].n[bounce].x;
            optixLaunchParams.trainInput[bounceOffset + 7] = optixLaunchParams.tBuffer[trOfs].n[bounce].y;
            optixLaunchParams.trainInput[bounceOffset + 8] = optixLaunchParams.tBuffer[trOfs].n[bounce].z;

            vec3f trainingColor = color + beta * cacheRadiance;
            if (isnan(trainingColor.x) || isnan(trainingColor.y) || isnan(trainingColor.z))
                trainingColor = vec3f(0.f);

            optixLaunchParams.trainGT[trOfs * MAX_BOUNCES + bounce] = trainingColor;
        }
    }
    else if(isTrainingPixel && gBuf.hit == false) {
        for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
            int bounceOffset = trOfs * inputCh * MAX_BOUNCES + bounce * inputCh;

            optixLaunchParams.trainInput[bounceOffset + 0] = 0.f;
            optixLaunchParams.trainInput[bounceOffset + 1] = 0.f;
            optixLaunchParams.trainInput[bounceOffset + 2] = 0.f;

            optixLaunchParams.trainInput[bounceOffset + 3] = 0.f;
            optixLaunchParams.trainInput[bounceOffset + 4] = 0.f;
            optixLaunchParams.trainInput[bounceOffset + 5] = 0.f;

            optixLaunchParams.trainInput[bounceOffset + 6] = 0.f;
            optixLaunchParams.trainInput[bounceOffset + 7] = 0.f;
            optixLaunchParams.trainInput[bounceOffset + 8] = 0.f;

            optixLaunchParams.trainGT[trOfs * MAX_BOUNCES + bounce] = vec3f(0.f);
        }
    }
}

__device__
int nrcTracePaths(int trOfs, Interaction si, LCGRand &rng, bool isTrainingPixel, 
                        bool isUnbiasedPixel)
{
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const vec2i pixelId = owl::getLaunchIndex();
    int fbOfs = pixelId.x + self.frameBufferSize.x * pixelId.y;
    int frameSize = self.frameBufferSize.x * self.frameBufferSize.y;

    int inputCh = optixLaunchParams.mlpInputCh;
    int outputCh = optixLaunchParams.mlpOutputCh;

    bool isTrainingSuffix = false;
    bool recalculateA0 = false;
    
    // Initialize camera spread
    float c = optixLaunchParams.c;
    float a0 = pow(length(si.p - optixLaunchParams.camera.pos), 2) / (4.f * Pi) / abs(si.wo_local.z);

    // Records to keep track of
    vec3f prevPoint(0.f);
    float prevPdf = 0.f;
    float spread = 0.f;

    // Path tracing variables
    vec3f finalColor(0.f), beta(1.f);

    GBuffer buf;
    buf.hit = true;

    int bounce = 0;
    for (bounce; bounce < MAX_BOUNCES; bounce++) {

        // Direct lighting
        vec3f directLight = directLighting(optixLaunchParams, si, rng);
        vec3f thisBeta(1.f);
        finalColor = finalColor + beta * directLight;

        // Next vertex in the path
        float prevRoughness = 1.f;
        vec3f prevAlbedo(0.f);
        if (si.isSurface) {
            prevAlbedo = si.color;
            prevRoughness = si.alpha;
        }
        else {
            prevAlbedo = colorFromSiga(si.color, si.beta_n);
            prevRoughness = si.beta_n;
        }
        vec3f prevWo = si.wo;
        vec3f prevN = si.n;
        prevPoint = si.p;
        prevPdf = nextPathVertexNRC(si, rng, thisBeta);

        beta = beta * thisBeta;

        if (isTrainingPixel) {
            if (recalculateA0) {
                a0 = pow(length(si.p - prevPoint), 2) / (4.f * Pi) / abs(si.wo_local.z);
                recalculateA0 = false;
            }

            optixLaunchParams.tBuffer[trOfs].vert[bounce] = prevPoint / optixLaunchParams.sceneScale;
            optixLaunchParams.tBuffer[trOfs].wo[bounce] = prevWo;
            optixLaunchParams.tBuffer[trOfs].n[bounce] = prevN;

            optixLaunchParams.tBuffer[trOfs].vertRadiance[bounce] = directLight;
            optixLaunchParams.tBuffer[trOfs].vertBeta[bounce] = thisBeta;
            optixLaunchParams.tBuffer[trOfs].color[bounce] = vec4f(prevAlbedo, prevRoughness);
        }

        // Calculate spread starting from the second path vertex
        spread = spread + nrcSpread(si.p, prevPoint, abs(si.wo_local.z), prevPdf);
        bool spreadCond = spread * spread > c * a0;
        
        /* Break loop if path exited, or if its the last bounce */
        if (si.hit == false || bounce == MAX_BOUNCES - 1) {
            if (!isTrainingSuffix) {
                buf.pathRadiance = finalColor;
                buf.beta = vec3f(0.f);
                buf.bounces = bounce;
                optixLaunchParams.gBuffer[fbOfs] = buf;
            }

            optixLaunchParams.tBuffer[trOfs].bounces = bounce;
            optixLaunchParams.tBuffer[trOfs].hit = false;

            break;
        }
        /* Also break if spread is too much */
        else if (spreadCond && !isTrainingSuffix) {
            buf.pathRadiance = finalColor;
            buf.beta = beta;
            buf.bounces = bounce;
            optixLaunchParams.gBuffer[fbOfs] = buf;

            vec3f point = si.p / optixLaunchParams.sceneScale;

            float roughness = 1.f;
            vec3f albedo(0.f);
            if (si.isSurface) {
                albedo = si.color;
                roughness = si.alpha;
            }
            else {
                albedo = colorFromSiga(si.color, si.beta_n);
                roughness = si.beta_n;
            }

            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 0] = point.x;
            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 1] = point.y;
            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 2] = point.z;

            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 3] = si.wo.x;
            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 4] = si.wo.y;
            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 5] = si.wo.z;

            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 6] = si.n.x;
            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 7] = si.n.y;
            optixLaunchParams.nnFrameInput[fbOfs * inputCh + 8] = si.n.z;

            // optixLaunchParams.nnFrameInput[fbOfs * inputCh + 9] = roughness;
            // 
            // optixLaunchParams.nnFrameInput[fbOfs * inputCh + 10] = albedo.x;
            // optixLaunchParams.nnFrameInput[fbOfs * inputCh + 11] = albedo.y;
            // optixLaunchParams.nnFrameInput[fbOfs * inputCh + 12] = albedo.z;

            if (isTrainingPixel) {
                isTrainingSuffix = true;
                recalculateA0 = true;
                spread = 0.f;
            }
            else {
                break;
            }
        }
        else if (spreadCond && isTrainingSuffix) {
            vec3f point = si.p / optixLaunchParams.sceneScale;

            float roughness = 1.f;
            vec3f albedo(0.f);
            if (si.isSurface) {
                albedo = si.color;
                roughness = si.alpha;
            }
            else {
                albedo = colorFromSiga(si.color, si.beta_n);
                roughness = si.beta_n;
            }

            optixLaunchParams.nnFrameInput[frameSize * inputCh + trOfs * inputCh + 0] = point.x;
            optixLaunchParams.nnFrameInput[frameSize * inputCh + trOfs * inputCh + 1] = point.y;
            optixLaunchParams.nnFrameInput[frameSize * inputCh + trOfs * inputCh + 2] = point.z;
                                              
            optixLaunchParams.nnFrameInput[frameSize * inputCh + trOfs * inputCh + 3] = si.wo.x;
            optixLaunchParams.nnFrameInput[frameSize * inputCh + trOfs * inputCh + 4] = si.wo.y;
            optixLaunchParams.nnFrameInput[frameSize * inputCh + trOfs * inputCh + 5] = si.wo.z;
                                          
            optixLaunchParams.nnFrameInput[frameSize * inputCh + trOfs * inputCh + 6] = si.n.x;
            optixLaunchParams.nnFrameInput[frameSize * inputCh + trOfs * inputCh + 7] = si.n.y;
            optixLaunchParams.nnFrameInput[frameSize * inputCh + trOfs * inputCh + 8] = si.n.z;

            optixLaunchParams.tBuffer[trOfs].bounces = bounce;
            optixLaunchParams.tBuffer[trOfs].hit = true;

            if (!isUnbiasedPixel) {
                break;
            }
            else {
                c = 1e30f; // Make sure spread cond is always false
            }
        }
    }

    return bounce;
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

    int trOfs = fbOfs / optixLaunchParams.everyNth;
    int trainIdx = optixLaunchParams.trainIdxs[trOfs] % optixLaunchParams.everyNth;

    bool allUnbiased = optixLaunchParams.allUnbiased;
    bool isTrainingPixel = fbOfs % optixLaunchParams.everyNth == trainIdx;
    bool isUnbiasedPixel = (trOfs % 16 == 0 ? true : false || allUnbiased) && isTrainingPixel;

    if (optixLaunchParams.pass == G_BUFFER) {
        // Shoot camera ray
        vec2f pixelOffset = vec2f(lcg_randomf(rng), lcg_randomf(rng));
        const vec2f screen = (vec2f(pixelId) + pixelOffset) / vec2f(self.frameBufferSize);

        RadianceRay ray;
        ray.origin
            = optixLaunchParams.camera.pos;
        ray.direction
            = normalize(optixLaunchParams.camera.dir_00
                + screen.u * optixLaunchParams.camera.dir_du
                + screen.v * optixLaunchParams.camera.dir_dv);


        Interaction si;
        si.hit = false;
        si.wo = -1.f * ray.direction;
        si.wi = ray.direction;

        owl::traceRay(optixLaunchParams.world, ray, si);

        if (si.hit == false || (si.isSurface && si.wo_local.z < 0.f)) {
            GBuffer buf;
            buf.hit = false;
            buf.pathRadiance = si.Le;
            buf.beta = vec3f(0.f);

            optixLaunchParams.gBuffer[fbOfs] = buf;
        }
        else if (si.hit == true) {
            nrcTracePaths(trOfs, si, rng, isTrainingPixel, isUnbiasedPixel);
        }
    }
    else if(optixLaunchParams.pass == RENDER) {
        vec3f color(0.f);

        GBuffer buf = optixLaunchParams.gBuffer[fbOfs];
        nrcGenerateTrainingData(trOfs, buf, isTrainingPixel, isUnbiasedPixel);

        vec3f cacheRadiance = (vec3f)optixLaunchParams.nnFrameOutput[fbOfs];
        if (isnan(cacheRadiance.x) || isnan(cacheRadiance.y) || isnan(cacheRadiance.z))
            cacheRadiance = vec3f(0.f);

        color = buf.pathRadiance + buf.beta * owl::max(cacheRadiance, vec3f(0.f));
        writePixel(color, optixLaunchParams.accumId,
            self.frameBuffer,
            optixLaunchParams.accumBuffer,
            optixLaunchParams.averageBuffer,
            fbOfs);
    }
    else if (optixLaunchParams.pass == RESET) {
        GBuffer buf;
        optixLaunchParams.gBuffer[fbOfs] = buf;

        for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
            optixLaunchParams.tBuffer[trOfs].vert[bounce] = vec3f(0.f);
            optixLaunchParams.tBuffer[trOfs].wo[bounce] = vec3f(0.f);
            optixLaunchParams.tBuffer[trOfs].n[bounce] = vec3f(0.f);

            optixLaunchParams.tBuffer[trOfs].vertRadiance[bounce] = vec3f(0.f);
            optixLaunchParams.tBuffer[trOfs].vertBeta[bounce] = vec3f(0.f);
        }

        optixLaunchParams.tBuffer[trOfs].bounces = 0;
        optixLaunchParams.tBuffer[trOfs].hit = false;
    }
}