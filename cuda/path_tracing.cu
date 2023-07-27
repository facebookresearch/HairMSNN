// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#include "path_tracing.cuh"
#include "utils.cuh"

#include "curve_utils.cuh"
#include "disney_hair.cuh"
#include "frostbite_anisotropic.cuh"

#include "optix_common.cuh"

#include <optix_device.h>

OPTIX_RAYGEN_PROGRAM(rayGenCam)()
{
    // ---------------------
    // Path Tracing
    // ---------------------
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const vec2i pixelId = owl::getLaunchIndex();
    int fbOfs = pixelId.x + self.frameBufferSize.x * pixelId.y;
    
    // Pseudo-random number generator
    LCGRand rng = get_rng(optixLaunchParams.accumId + 10007, make_uint2(pixelId.x, pixelId.y),
        make_uint2(self.frameBufferSize.x, self.frameBufferSize.y));
    
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
    
    vec3f color(0.f, 0.f, 0.f);
    int v1Stop = optixLaunchParams.pathV1 - 1;
    int v2Stop = optixLaunchParams.pathV2 - 1;
    
    if (si.hit == false)
        color = si.Le;
    else {
        color = pathTrace(si, rng, v1Stop, v2Stop);
    }
    
    writePixel(color, optixLaunchParams.accumId,
        self.frameBuffer,
        optixLaunchParams.accumBuffer,
        optixLaunchParams.averageBuffer,
        fbOfs);
}