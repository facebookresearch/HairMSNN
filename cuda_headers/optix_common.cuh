// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#pragma once

__device__
vec3f getEnvironmentRadiance(LaunchParams &params, vec3f dir)
{
    float theta = sphericalTheta(dir);
    float phi = sphericalPhi(dir) + params.envRotPhi;

    if (phi > TWO_Pi)
        phi = phi - TWO_Pi;

    float x = phi / (2 * Pi);
    float y = theta / (Pi);

    vec3f rval =  params.envScale * (vec3f)tex2D<float4>(params.env, x, y);
    return rval;
}

__device__
float getEnvironmentPdf(LaunchParams& params, vec3f wi)
{
    float pdf = 1.f;

    if (params.envPdfSampling) {
        float width = (float)params.envWidth;
        float height = (float)params.envHeight;

        float theta = sphericalTheta(wi);
        float phi = sphericalPhi(wi) + params.envRotPhi;
        if (phi > TWO_Pi)
            phi = phi - TWO_Pi;

        float u = phi / (2.f * Pi);
        float v = theta / Pi;

        int index_u = clamp(int(u * params.envWidth), 0, params.envWidth - 1);
        int index_v = clamp(int(v * params.envHeight), 0, params.envHeight - 1);

        // pdfs in v direction
        int cdf_width = params.envWidth + 1;
        float cdf_last_u = tex2D<float>(params.conditionalPdf, 1.f, index_v / height);
        float cdf_last_v = tex2D<float>(params.marginalPdf, 1.f, 0.f);

        float sin_theta = sinf(theta);
        float denom = (2.f * Pi * Pi * sin_theta) * cdf_last_u * cdf_last_v;

        // pdfs in u direction
        float cdf_u = tex2D<float>(params.conditionalPdf, index_u / width, index_v / height);
        float cdf_v = tex2D<float>(params.marginalPdf, index_v / height, 0.f);
        pdf = denom ? (cdf_u * cdf_v) / denom : 0.f;
    }
    else {
        pdf = 1.f / (4.f * Pi);
    }

    return pdf;
}

__device__
vec3f sampleEnvironmentLight(LaunchParams& params, Interaction& si, LCGRand& rng, float* pdf)
{
    vec3f rval(0.f);

    if (params.envPdfSampling) {
        auto lower_bound = [](float u, cudaTextureObject_t cdf, float y, float size) {
            // This is basically std::lower_bound as used by PBRT
            int first = 0;
            int count = size;
            while (count > 0)
            {
                int step = count >> 1;
                int middle = first + step;
                if (tex2D<float>(cdf, middle / size, y) < u)
                {
                    first = middle + 1;
                    count -= step + 1;
                }
                else
                    count = step;
            }
            return max(0, first - 1);
        };

        vec2f u01 = vec2f(lcg_randomf(rng), lcg_randomf(rng));

        float width = (float)params.envWidth;
        float height = (float)params.envHeight;

        int index_v = lower_bound(u01.y, params.marginalCdf, 0.f, height);

        float cdf_v = tex2D<float>(params.marginalCdf, index_v / height, 0.f);
        float cdf_next_v = tex2D<float>(params.marginalCdf, (index_v + 1) / height, 0.f);
        float cdf_last_v = tex2D<float>(params.marginalCdf, 1.f, 0.f);

        // Importance-sampled v direction
        float dv = (cdf_next_v - u01.y) / (cdf_next_v - cdf_v);
        float v = (index_v + dv) / height;

        int cdf_width = width + 1;
        int index_u = lower_bound(u01.x, params.conditionalCdf, index_v / height, width);

        float cdf_u = tex2D<float>(params.conditionalCdf, index_u / width, index_v / height);
        float cdf_next_u = tex2D<float>(params.conditionalCdf, (index_u + 1) / width, index_v / height);
        float cdf_last_u = tex2D<float>(params.conditionalCdf, 1.f, index_v / height);

        // Importance-sampled u direction
        float du = (cdf_next_u - u01.x) / (cdf_next_u - cdf_u);
        float u = (index_u + du) / width;

        rval = params.envScale * (vec3f)tex2D<float4>(params.env, u, v);

        // Compute pdf
        cdf_last_u = tex2D<float>(params.conditionalPdf, 1.f, index_v / height);
        cdf_last_v = tex2D<float>(params.marginalPdf, 1.f, 0.f);

        cdf_u = tex2D<float>(params.conditionalPdf, index_u / width, index_v / height);
        cdf_v = tex2D<float>(params.marginalPdf, index_v / height, 0.f);

        float theta = Pi * v;
        float sin_theta = sinf(theta);
        float denom = (2.f * Pi * Pi * sin_theta) * cdf_last_u * cdf_last_v;
        *pdf = denom ? (cdf_u * cdf_v) / denom : 0.f;

        // Compute direction
        float phi = 2.f * Pi * u - params.envRotPhi;
        if (phi < 0)
            phi = TWO_Pi + phi;

        si.wi = vec3f(sin_theta * cosf(phi), sin_theta * sinf(phi), cosf(theta));
        si.wi_local = normalize(apply_mat(si.to_local, si.wi));
    }
    else {
        vec2f u01 = vec2f(lcg_randomf(rng), lcg_randomf(rng));
        si.wi = uniformSampleSphere(u01);
        si.wi_local = normalize(apply_mat(si.to_local, si.wi));

        *pdf = 1.f / (4.f * Pi);
        rval = getEnvironmentRadiance(params, si.wi);
    }

    return rval;
}

__device__
vec3f sampleBSDF(LaunchParams& params, Interaction& si, LCGRand& rng)
{
    vec2f rand1 = vec2f(lcg_randomf(rng), lcg_randomf(rng));

    vec3f emit = vec3f(0.f), bsdf = vec3f(0.f);
    vec3f final_color = vec3f(0.f);
    float lightPdf = 1.f, bsdfPdf = 1.f;

    if (!si.isSurface) {
        vec4f rand2 = vec4f(rand1.x, rand1.y, lcg_randomf(rng), lcg_randomf(rng));
        bsdf = sample_disney_hair(si, rand2, &bsdfPdf);
    }
    else {
        si.wi_local = sample_GGX(rand1, si.alpha, si.wo_local,
            &bsdfPdf);
        si.wi = normalize(apply_mat(si.to_world, si.wi_local));

        bsdf = frostbite_GGX(si.wo_local, si.wi_local, si.color, si.alpha);
    }

    float wiDotN = dot(si.wi, si.n);
    bool lowerHemi = wiDotN < 0.f;
    vec3f nd = si.n;

    if (si.isSurface && lowerHemi) {
        return final_color;
    }
    else if (!si.isSurface && lowerHemi) {
        nd = -si.n;
        si.p = si.p + 2.f * si.hair.radius * nd;
    }

    ShadowRay shadowRay;
    shadowRay.origin = si.p + 1e-3f * nd;
    shadowRay.direction = si.wi;
    ShadowRayData srd;
    srd.visibility = vec3f(1.f);
    owl::traceRay(params.world, shadowRay, srd);

    if (srd.visibility != vec3f(0.f)) {

        emit = getEnvironmentRadiance(params, si.wi);
        lightPdf = getEnvironmentPdf(params, si.wi) * 1.f / params.num_total_lights;

        if (params.MIS)
            final_color = bsdf * emit * powerHeuristic(1, bsdfPdf, 1, lightPdf) / bsdfPdf;
        else
            final_color = bsdf * emit / bsdfPdf;
    }

    return final_color;
}

__device__
vec3f sampleLights(LaunchParams& params, Interaction& si, LCGRand& rng, bool& isDelta)
{
    isDelta = false;

    vec3f emit = vec3f(0.f);
    vec3f final_color = vec3f(0.f);
    float lightPdf = 1.f, bsdfPdf = 1.f;

    float rand = lcg_randomf(rng);
    int selectedLight = floor(rand * params.num_total_lights);

    if (selectedLight >= params.num_dlights) {
        // Environment lighting
        selectedLight = selectedLight - params.num_dlights;
        lightPdf = lightPdf * 1.f / params.num_total_lights;

        float envPdf = 1.f;
        emit = sampleEnvironmentLight(params, si, rng, &envPdf);
        lightPdf = lightPdf * envPdf;
    }
    else {
        // Directional lighting
        selectedLight = selectedLight;
        lightPdf = lightPdf * 1.f / params.num_total_lights;

        DirectionalLight dLight = params.dLights[selectedLight];
        emit = dLight.emit;

        si.wi = normalize(dLight.from);
        si.wi_local = normalize(apply_mat(si.to_local, si.wi));

        isDelta = true;
    }

    float wiDotN = dot(si.wi, si.n);
    bool lowerHemi = wiDotN < 0.f;
    vec3f nd = si.n;

    if (si.isSurface && lowerHemi) {
        return final_color;
    }
    else if (!si.isSurface && lowerHemi) {
        nd = -si.n;
        si.p = si.p + 2.f * si.hair.radius * nd;
    }

    ShadowRay shadowRay;
    shadowRay.origin = si.p + 1e-3f * nd;
    shadowRay.direction = si.wi;
    ShadowRayData srd;
    srd.visibility = vec3f(1.f);
    owl::traceRay(params.world, shadowRay, srd);

    if (srd.visibility != vec3f(0.f)) {
        vec3f bsdf(0.f);

        if (!si.isSurface) {
            bsdf = disney_hair(si, &bsdfPdf);
        }
        else {
            bsdf = frostbite_GGX(si.wo_local, si.wi_local, si.color, si.alpha);
            bsdfPdf = pdf_GGX(si.alpha, si.wo_local, normalize(si.wo_local + si.wi_local));
        }

        if (params.MIS && !isDelta)
            final_color = bsdf * emit * powerHeuristic(1, lightPdf, 1, bsdfPdf) / lightPdf;
        else
            final_color = bsdf * emit / lightPdf;
    }

    return final_color;
}

__device__
vec3f directLighting(LaunchParams& params, Interaction si, LCGRand& rng)
{
    vec3f final_color(0.f);

    if (params.MIS) {
        // Perform Multiple Importance sampling b/w light and BSDF
        // First select a light source, and calculate MIS weighted radiance from it
        bool isDelta = false;
        vec3f lightSample = sampleLights(params, si, rng, isDelta);

        // Next, importance sample BRDF, only if the light is not a delta light (point, direction etc.)
        vec3f bsdfSample = vec3f(0.f);
        if (!isDelta)
            bsdfSample = sampleBSDF(params, si, rng);

        // Accumulate over path vertices
        vec3f misSample = lightSample + bsdfSample;
        final_color += misSample;
    }
    else {
        // Regular light sampling (Next event estimation)
        bool isDelta = false;
        final_color += sampleLights(params, si, rng, isDelta);
    }

    if (isnan(final_color.x) || isnan(final_color.y) || isnan(final_color.z))
        final_color = vec3f(0.f);

    return final_color;
}

__device__
vec3f nextPathVertex(LaunchParams& params, Interaction& si, LCGRand& rng, vec3f& beta)
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
    owl::traceRay(params.world, nextRay, si);

    return mulFac;
}

__device__
vec3f pathTrace(Interaction si, LCGRand& rng, int v1Stop, int v2Stop)
{
    if (v2Stop < v1Stop)
        return vec3f(0.f);

    vec3f beta(1.f);
    vec3f final_color(0.f);

    if (v1Stop == 0)
        final_color = directLighting(optixLaunchParams, si, rng);

    int bounces = 1;
    for (bounces = 1; bounces <= v2Stop; bounces++) {
        /* ================================================
        Next vertex in the path
        ================================================ */
        nextPathVertex(optixLaunchParams, si, rng, beta);

        /* ================================================
        Terminate if escaped
        ================================================ */
        if (si.hit == false) {
            break;
        }

        /* ================================================
        Direct lighting
        ================================================ */
        if (bounces >= v1Stop) {
            final_color += beta * directLighting(optixLaunchParams, si, rng);
        }

        /* ================================================
        Russian Roulette path termination (from PBRT)
        ================================================ */
        float q = max(0.05f, 1.f - luminance(beta));
        float eps = lcg_randomf(rng);

        if (eps < q) {
            break;
        }
        
        beta = beta / (1.f - q);
    }

    return final_color;
}

__inline__ __device__
vec3f densityTowardsLights(Interaction si, LCGRand& rng, bool evalForwardIntersections,
    vec3f* dir, float* weight, bool* surfaceHit, bool* hairHit, int* numHairHits)
{
    vec3f emit(0.f);
    float lightPdf = 1.f;

    float rand = lcg_randomf(rng);
    int selectedLight = floor(rand * optixLaunchParams.num_total_lights);

    if (selectedLight >= optixLaunchParams.num_dlights) {
        // Environment lighting
        selectedLight = selectedLight - optixLaunchParams.num_dlights;
        lightPdf = lightPdf * 1.f / optixLaunchParams.num_total_lights;

        float envPdf = 1.f;
        emit = sampleEnvironmentLight(optixLaunchParams, si, rng, &envPdf);
        lightPdf = lightPdf * envPdf;
    }
    else {
        // Directional lighting
        selectedLight = selectedLight;
        lightPdf = lightPdf * 1.f / optixLaunchParams.num_total_lights;

        DirectionalLight dLight = optixLaunchParams.dLights[selectedLight];
        emit = dLight.emit;

        si.wi = normalize(dLight.from);
        si.wi_local = normalize(apply_mat(si.to_local, si.wi));
    }

    *dir = si.wi;
    *weight = 1.f / lightPdf;

    float wiDotN = dot(si.wi, si.n);
    bool lowerHemi = wiDotN < 0.f;
    vec3f nd = si.n;

    if (lowerHemi) {
        nd = -si.n;
        si.p = si.p + 2.f * si.hair.radius * nd;
    }

    if (!evalForwardIntersections) {
        ShadowRay shadowRay;
        shadowRay.origin = si.p + 1e-3f * nd;
        shadowRay.direction = si.wi;
        ShadowRayData srd;
        srd.visibility = vec3f(1.f);
        owl::traceRay(optixLaunchParams.world, shadowRay, srd);

        *surfaceHit = srd.isSurface;
        *hairHit = srd.isSurface;
        *numHairHits = 0;
    }
    else {
        MultiScatterRay msRay;
        msRay.origin = si.p + 1e-3f * nd;
        msRay.direction = si.wi;

        MultiScatterRayData mrd;
        owl::traceRay(optixLaunchParams.world, msRay, mrd);

        *surfaceHit = mrd.surfaceHit;
        *hairHit = mrd.hairHit;
        *numHairHits = mrd.numHairHits;
    }

    return emit;
}

/* ==========================================================
ANY_HIT functions for MultiScatterRay
============================================================= */
OPTIX_ANY_HIT_PROGRAM(hairAHMultiScatter)()
{
    MultiScatterRayData& si = owl::getPRD<MultiScatterRayData>();
    si.hairHit = true;
    si.numHairHits = si.numHairHits + 1;

    optixIgnoreIntersection();
}

OPTIX_ANY_HIT_PROGRAM(triangleMeshAHMultiScatter)()
{
    MultiScatterRayData& si = owl::getPRD<MultiScatterRayData>();
    si.surfaceHit = true;
    si.numHairHits = 0;

    optixTerminateRay();
}

/* ==========================================================
ANY_HIT functions for ShadowRay
============================================================= */
OPTIX_ANY_HIT_PROGRAM(triangleMeshAHShadow)()
{
    ShadowRayData& si = owl::getPRD<ShadowRayData>();
    si.visibility = vec3f(0.f);
    si.isSurface = true;

    optixTerminateRay();
}

OPTIX_ANY_HIT_PROGRAM(hairAHShadow)()
{
    ShadowRayData& si = owl::getPRD<ShadowRayData>();
    si.visibility = vec3f(0.f);
    si.isSurface = false;

    optixTerminateRay();
}

/* ==========================================================
CLOSEST_HIT functions for RadianceRay
============================================================= */
OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshCH)()
{
    const TriangleMeshData& self = owl::getProgramData<TriangleMeshData>();
    const vec3i primitiveIndices = self.index[optixGetPrimitiveIndex()];

    Interaction& si = owl::getPRD<Interaction>();
    si.p = barycentricInterpolate(self.vertex, primitiveIndices);
    // si.p = (vec3f)optixGetWorldRayOrigin() + optixGetRayTmax() * normalize((vec3f)optixGetWorldRayDirection());
    si.uv = barycentricInterpolate(self.texCoord, primitiveIndices);
    si.n = normalize(barycentricInterpolate(self.normal, primitiveIndices));
    orthonormalBasis(si.n, si.to_local, si.to_world);

    si.t = si.to_local[0];
    si.wo_local = normalize(apply_mat(si.to_local, si.wo));

    si.color = self.diffuse;
    if (self.hasDiffuseTexture)
        si.color = (vec3f)tex2D<float4>(self.diffuse_texture, si.uv.x, si.uv.y);

    si.alpha = self.alpha;
    if (self.hasAlphaTexture) {
        vec3f alphaTex = tex2D<float4>(self.alpha_texture, si.uv.x, si.uv.y);
        si.alpha = 0.33f * (alphaTex.x + alphaTex.y + alphaTex.z);
    }
    si.alpha = clamp(si.alpha, 0.01f, 1.f);
    si.alpha = si.alpha * si.alpha;

    si.hit = true;
    si.isSurface = true;
}

OPTIX_CLOSEST_HIT_PROGRAM(hairCH)()
{
    const HairData& self = owl::getProgramData<HairData>();
    Interaction& si = owl::getPRD<Interaction>();

    unsigned int primIdx = optixGetPrimitiveIndex();
    computeCurveIntersection(optixGetPrimitiveIndex(),
        si.p,
        si.n,
        si.t,
        si.hair.curve_p,
        si.hair.radius);

    vec3f X = si.t;
    vec3f Y = normalize(cross(si.wo, X));
    vec3f Z = normalize(cross(X, Y));

    si.to_local[0] = X;
    si.to_local[1] = Y;
    si.to_local[2] = Z;

    si.to_world[0] = vec3f(X.x, Y.x, Z.x);
    si.to_world[1] = vec3f(X.y, Y.y, Z.y);
    si.to_world[2] = vec3f(X.z, Y.z, Z.z);

    si.wo_local = normalize(apply_mat(si.to_local, si.wo));

    si.color = optixLaunchParams.hairData.sig_a;
    si.beta_m = optixLaunchParams.hairData.beta_m;
    si.beta_n = optixLaunchParams.hairData.beta_n;
    si.alpha = optixLaunchParams.hairData.alpha;

    si.hair.R_G = optixLaunchParams.hairData.R_G;
    si.hair.TT_G = optixLaunchParams.hairData.TT_G;
    si.hair.TRT_G = optixLaunchParams.hairData.TRT_G;
    si.hair.TRRT_G = optixLaunchParams.hairData.TRRT_G;

    si.hit = true;
    si.isSurface = false;

    setupHairShading(si);
}

/* ==========================================================
Generic MISS function
============================================================= */
OPTIX_MISS_PROGRAM(miss)()
{
    const vec2i pixelId = owl::getLaunchIndex();
    const MissProgData& self = owl::getProgramData<MissProgData>();

    Interaction& si = owl::getPRD<Interaction>();
    si.hit = false;
    si.Le = vec3f(0.f);

    if (optixLaunchParams.hasEnvLight) {
        si.Le = getEnvironmentRadiance(optixLaunchParams, si.wi);
    }
}