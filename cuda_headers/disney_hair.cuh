// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#pragma once

__device__ const float Eta = 1.55f;
__device__ const float SqrtPiOver8 = 0.626657069f;

__device__ __forceinline__ float Sqr(float v) { return v * v; }

__device__ __forceinline__
float SafeASin(float x) {
    return asin(owl::clamp(x, -1.f, 1.f));
}

__device__ __forceinline__
float SafeSqrt(float x) {
    return owl::sqrt(owl::max(0.f, x));
}

__device__ __forceinline__
void setupHairShading(Interaction& si)
{
    si.hair.v[0] = Sqr(0.726f * si.beta_m + 0.812f * Sqr(si.beta_m) + 3.7f * pow(si.beta_m, 20.f));
    si.hair.v[1] = .25 * si.hair.v[0];
    si.hair.v[2] = 4 * si.hair.v[0];

    si.hair.s = SqrtPiOver8 *
        (0.265f * si.beta_n + 1.194f * Sqr(si.beta_n) + 5.372f * pow(si.beta_n, 22.f));

    si.hair.sin2kAlpha[0] = sin(si.alpha);
    si.hair.cos2kAlpha[0] = SafeSqrt(1 - Sqr(si.hair.sin2kAlpha[0]));
    for (int i = 1; i < 3; ++i) {
        si.hair.sin2kAlpha[i] = 2 * si.hair.cos2kAlpha[i - 1] * si.hair.sin2kAlpha[i - 1];
        si.hair.cos2kAlpha[i] = Sqr(si.hair.cos2kAlpha[i - 1]) - Sqr(si.hair.sin2kAlpha[i - 1]);
    }
}



__device__ __forceinline__
float Fresnel(float eta, float cosTheta) {
    float F0 = Sqr(1 - eta) / Sqr(1 + eta);
    float xd = (1 - cosTheta);
    
    return F0 + (1 - F0) * xd * xd * xd * xd * xd;
}

__device__ __forceinline__
float I0(float x) {
    float val = 0;
    float x2i = 1;
    int64_t ifact = 1;
    int i4 = 1;
    // I0(x) \approx Sum_i x^(2i) / (4^i (i!)^2)
    for (int i = 0; i < 10; ++i) {
        if (i > 1) ifact *= i;
        val += x2i / (i4 * Sqr(ifact));
        x2i *= x * x;
        i4 *= 4;
    }
    return val;
}

__device__ __forceinline__
float LogI0(float x) {
    if (x > 12)
        return x + 0.5 * (-log(2 * Pi) + log(1 / x) + 1 / (8 * x));
    else
        return log(I0(x));
}

__device__ __forceinline__
float Mp(float cosThetaI, float cosThetaO, float sinThetaI, float sinThetaO, float v) {
    float a = cosThetaI * cosThetaO / v;
    float b = sinThetaI * sinThetaO / v;
    float mp = 0.f;
    if(v <= 0.1f)
        mp = exp(LogI0(a) - b - 1 / v + 0.6931f + log(1 / (2 * v)));
    else
        mp = (exp(-b) * I0(a)) / (sinh(1 / v) * 2 * v);
    
    return mp;
}

__device__ __forceinline__
vec3f Ap0(float fresnel, vec3f& T) {
    return vec3f(fresnel);
}

__device__ __forceinline__
vec3f Ap1(float fresnel, vec3f& T) {
    return Sqr(1-fresnel) * T;
}

__device__ __forceinline__
vec3f Ap2(float fresnel, vec3f& T) {
    return Sqr(1 - fresnel) * T * T * fresnel;
}

__device__ __forceinline__
vec3f ApMax(float fresnel, vec3f& T) {
    vec3f ap2 = Ap2(fresnel, T);
    return ap2 * fresnel * T / (vec3f(1.f) - T * fresnel);
}

__device__ __forceinline__
float Phi(int p, float gammaO, float gammaT) {
    return 2 * p * gammaT - 2 * gammaO + p * Pi;
}

__device__ __forceinline__
float Logistic(float x, float s) {
    x = abs(x);
    return exp(-x / s) / (s * Sqr(1 + exp(-x / s)));
}

__device__ __forceinline__
float LogisticCDF(float x, float s) {
    return 1 / (1 + exp(-x / s));
}

__device__ __forceinline__
float TrimmedLogistic(float x, float s, float a, float b) {
    return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
}

__device__ __forceinline__
float Np(float phi, int p, float s, float gammaO, float gammaT) {
    float dphi = phi - Phi(p, gammaO, gammaT);
    // Remap _dphi_ to $[-\pi,\pi]$
    while (dphi > Pi) dphi -= 2 * Pi;
    while (dphi < -Pi) dphi += 2 * Pi;
    return TrimmedLogistic(dphi, s, -Pi, Pi);
}

// Sampling routines
__device__ __forceinline__
vec4f computeApPdf(Interaction& si, float sinThetaO, float cosThetaO, float cosGammaO, float eta, float h) {
    // Compute $\cos \thetat$ for refracted ray
    float sinThetaT = sinThetaO / eta;
    float cosThetaT = SafeSqrt(1 - Sqr(sinThetaT));

    // Compute $\gammat$ for refracted ray
    float etap = owl::sqrt(eta * eta - Sqr(sinThetaO)) / cosThetaO;
    float sinGammaT = h / etap;
    float cosGammaT = SafeSqrt(1 - Sqr(sinGammaT));

    // Compute the transmittance _T_ of a single path through the cylinder
    float fac = (2.f * cosGammaT / cosThetaT);
    float Tx = exp(-si.color.x * fac);
    float Ty = exp(-si.color.y * fac);
    float Tz = exp(-si.color.z * fac);
    vec3f T = vec3f(Tx, Ty, Tz);

    float fresnel = Fresnel(eta, cosThetaO * cosGammaO);

    vec3f apVec[4] = {vec3f(0.f)};
    apVec[0] = Ap0(fresnel, T);
    apVec[1] = Ap1(fresnel, T);
    apVec[2] = Ap2(fresnel, T);
    apVec[3] = ApMax(fresnel, T);

    vec4f ap;
    ap.x = (apVec[0].x + apVec[0].y + apVec[0].z) / 3.f;
    ap.y = (apVec[1].x + apVec[1].y + apVec[1].z) / 3.f;
    ap.z = (apVec[2].x + apVec[2].y + apVec[2].z) / 3.f;
    ap.w = (apVec[3].x + apVec[3].y + apVec[3].z) / 3.f;

    float sum = ap.x + ap.y + ap.z + ap.w;

    ap.x = ap.x / sum;
    ap.y = ap.y / sum;
    ap.z = ap.z / sum;
    ap.w = ap.w / sum;

    return ap;
}

/* returns f()*cos() */
__device__ __forceinline__
vec3f disney_hair(Interaction si, float* pdf) {
    vec3f wo = si.wo_local;
    vec3f wi = si.wi_local;
    vec3f n = normalize(apply_mat(si.to_local, si.n));

    // Compute hair coordinate system terms related to _wo_
    float sinThetaO = wo.x;
    float cosThetaO = SafeSqrt(1 - Sqr(sinThetaO));
    float phiO = atan2(wo.z, wo.y);

    // Compute hair coordinate system terms related to _wi_
    float sinThetaI = wi.x;
    float cosThetaI = SafeSqrt(1 - Sqr(sinThetaI));
    float phiI = atan2(wi.z, wi.y);

    // Compute $\cos \thetat$ for refracted ray
    float sinThetaT = sinThetaO / Eta;
    float cosThetaT = SafeSqrt(1 - Sqr(sinThetaT));

    // Compute $\gammat$ for refracted ray
    float h = dot(si.to_local[1], si.n);
    float gammaO = asin(h);
    float cosGammaO = cos(gammaO);

    float etap = owl::sqrt(Eta * Eta - Sqr(sinThetaO)) / cosThetaO;
    float sinGammaT = h / etap;
    float cosGammaT = SafeSqrt(1 - Sqr(sinGammaT));
    float gammaT = SafeASin(sinGammaT);

    float fac = (2.f * cosGammaT / cosThetaT);
    float Tx = exp(-si.color.x * fac);
    float Ty = exp(-si.color.y * fac);
    float Tz = exp(-si.color.z * fac);
    vec3f T = vec3f(Tx, Ty, Tz);

    float fresnel = Fresnel(Eta, cosThetaO * cosGammaO);

    float phi = phiI - phiO;
    vec3f f(0.f);

    float sinThetaOp = 0.f, cosThetaOp = 0.f;

    sinThetaOp = sinThetaO * si.hair.cos2kAlpha[1] - cosThetaO * si.hair.sin2kAlpha[1];
    cosThetaOp = cosThetaO * si.hair.cos2kAlpha[1] + sinThetaO * si.hair.sin2kAlpha[1];
    cosThetaOp = abs(cosThetaOp);
    
    float Mp0 = Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, si.hair.v[0]);
    float Np0 = Np(phi, 0, si.hair.s, gammaO, gammaT);
    f += si.hair.R_G * Mp0 * Ap0(fresnel, T) * Np0;

    sinThetaOp = sinThetaO * si.hair.cos2kAlpha[0] + cosThetaO * si.hair.sin2kAlpha[0];
    cosThetaOp = cosThetaO * si.hair.cos2kAlpha[0] - sinThetaO * si.hair.sin2kAlpha[0];
    cosThetaOp = abs(cosThetaOp);

    float Mp1 = Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, si.hair.v[1]);
    float Np1 = Np(phi, 1, si.hair.s, gammaO, gammaT);
    f += si.hair.TT_G * Mp1  * Ap1(fresnel, T)  * Np1;

    sinThetaOp = sinThetaO * si.hair.cos2kAlpha[2] + cosThetaO * si.hair.sin2kAlpha[2];
    cosThetaOp = cosThetaO * si.hair.cos2kAlpha[2] - sinThetaO * si.hair.sin2kAlpha[2];
    cosThetaOp = abs(cosThetaOp);

    float Mp2 = Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, si.hair.v[2]);
    float Np2 = Np(phi, 2, si.hair.s, gammaO, gammaT);
    f += si.hair.TRT_G * Mp2 * Ap2(fresnel, T) * Np2;

    float MpMax = Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, si.hair.v[2]);
    float NpMax = 1.f / (2.f * Pi);
    f += si.hair.TRRT_G * MpMax * ApMax(fresnel, T) * NpMax;

    // Compute the BSDF pdf
    *pdf = 0.f;
    vec4f apPdf = computeApPdf(si, sinThetaO, cosThetaO, cosGammaO, Eta, h);

    (*pdf) = (*pdf) + Mp0 * apPdf[0] * Np0;
    (*pdf) = (*pdf) + Mp1 * apPdf[1] * Np1;
    (*pdf) = (*pdf) + Mp2 * apPdf[2] * Np2;
    (*pdf) = (*pdf) + MpMax * apPdf[3] * NpMax;

    return f;
}

__device__ __forceinline__ 
float SampleTrimmedLogistic(float u, float s, float a, float b) {
    float k = LogisticCDF(b, s) - LogisticCDF(a, s);
    float x = -s * log(1 / (u * k + LogisticCDF(a, s)) - 1);

    return clamp(x, a, b);
}

__device__ __forceinline__
vec3f sample_disney_hair(Interaction& si, vec4f& rand, float *pdf) {

    vec3f wo = si.wo_local;
    vec3f n = normalize(apply_mat(si.to_local, si.n));

    // Compute hair coordinate system terms related to _wo_
    float sinThetaO = wo.x;
    float cosThetaO = SafeSqrt(1 - Sqr(sinThetaO));
    float phiO = atan2(wo.z, wo.y);

    // Compute $\gammat$ for refracted ray
    // vec3f wop(0.f, wo.y, wo.z);
    // wop = normalize(wop);
    // float cosGammaO = dot(wop, n);
    // float gammaO = acos(cosGammaO);
    // float h = sin(gammaO);
    float h = dot(si.to_local[1], si.n);
    float gammaO = asin(h);
    float cosGammaO = cos(gammaO);

    vec4f apPdf = computeApPdf(si, sinThetaO, cosThetaO, cosGammaO, Eta, h);

    // Choose p for Ap (absorption term)
    int p = 0;
    float eps1 = rand.x;
    for (p = 0; p < 3; ++p) {
        if (eps1 < apPdf[p]) break;
        eps1 -= apPdf[p];
    }

    // Rotate $\sin \thetao$ and $\cos \thetao$ to account for hair scale tilt
    float sinThetaOp = 0.f, cosThetaOp = 0.f;
    if (p == 0) {
        sinThetaOp = sinThetaO * si.hair.cos2kAlpha[1] - cosThetaO * si.hair.sin2kAlpha[1];
        cosThetaOp = cosThetaO * si.hair.cos2kAlpha[1] + sinThetaO * si.hair.sin2kAlpha[1];
    }
    else if (p == 1) {
        sinThetaOp = sinThetaO * si.hair.cos2kAlpha[0] + cosThetaO * si.hair.sin2kAlpha[0];
        cosThetaOp = cosThetaO * si.hair.cos2kAlpha[0] - sinThetaO * si.hair.sin2kAlpha[0];
    }
    else if (p == 2) {
        sinThetaOp = sinThetaO * si.hair.cos2kAlpha[2] + cosThetaO * si.hair.sin2kAlpha[2];
        cosThetaOp = cosThetaO * si.hair.cos2kAlpha[2] - sinThetaO * si.hair.sin2kAlpha[2];
    }
    else {
        sinThetaOp = sinThetaO;
        cosThetaOp = cosThetaO;
    }

    // Sample $M_p$ to compute $\thetai$
    float eps2 = owl::max(rand.y, 1e-5f);
    float cosTheta =
        1.f + si.hair.v[p] * log(eps2 + (1.f - eps2) * exp(-2.f / si.hair.v[p]));
    float sinTheta = SafeSqrt(1 - Sqr(cosTheta));
    float cosPhi = cos(2 * Pi * rand.z);
    float sinThetaI = -cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp;
    float cosThetaI = SafeSqrt(1 - Sqr(sinThetaI));

    // Sample $N_p$ to compute $\Delta\phi$

    // Compute $\gammat$ for refracted ray
    float etap = owl::sqrt(Eta * Eta - Sqr(sinThetaO)) / cosThetaO;
    float sinGammaT = h / etap;
    float gammaT = SafeASin(sinGammaT);
    float dphi;
    if (p < 3)
        dphi = Phi(p, gammaO, gammaT) + SampleTrimmedLogistic(rand.w, si.hair.s, -Pi, Pi);
    else
        dphi = 2 * Pi * rand.w;

    // Compute _wi_ from sampled hair scattering angles
    float phiI = phiO + dphi;
    si.wi_local = normalize(vec3f(sinThetaI, cosThetaI * cos(phiI), cosThetaI * sin(phiI)));
    si.wi = normalize(apply_mat(si.to_world, si.wi_local));

    // // Compute PDF for sampled hair scattering direction _wi_
    // (*pdf) = 0.f;
    // for (int p = 0; p < 3; ++p) {
    //     // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
    //     float sinThetaOp = 0.f, cosThetaOp = 0.f;
    //     if (p == 0) {
    //         sinThetaOp = sinThetaO * si.cos2kAlpha[1] - cosThetaO * si.sin2kAlpha[1];
    //         cosThetaOp = cosThetaO * si.cos2kAlpha[1] + sinThetaO * si.sin2kAlpha[1];
    //     }
    // 
    //     // Handle remainder of $p$ values for hair scale tilt
    //     else if (p == 1) {
    //         sinThetaOp = sinThetaO * si.cos2kAlpha[0] + cosThetaO * si.sin2kAlpha[0];
    //         cosThetaOp = cosThetaO * si.cos2kAlpha[0] - sinThetaO * si.sin2kAlpha[0];
    //     }
    //     else if (p == 2) {
    //         sinThetaOp = sinThetaO * si.cos2kAlpha[2] + cosThetaO * si.sin2kAlpha[2];
    //         cosThetaOp = cosThetaO * si.cos2kAlpha[2] - sinThetaO * si.sin2kAlpha[2];
    //     }
    //     else {
    //         sinThetaOp = sinThetaO;
    //         cosThetaOp = cosThetaO;
    //     }
    // 
    //     // Handle out-of-range $\cos \thetao$ from scale adjustment
    //     cosThetaOp = abs(cosThetaOp);
    //     *pdf += Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, si.v[p]) *
    //         apPdf[p] * Np(dphi, p, si.s, gammaO, gammaT);
    // }
    // 
    // *pdf += Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, si.v[2]) *
    //     apPdf[3] * (1 / (2 * Pi));

    return disney_hair(si, pdf);
}