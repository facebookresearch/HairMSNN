// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#pragma once

#define Pi 3.1415926f
#define TWO_Pi 2.f * 3.14159f

__device__
void writePixel(vec3f& color, int accumId,
    uint32_t* frameBuffer, 
    float4* accumBuffer,
    float4* averageBuffer,
    int offset)
{
    if (isnan(color.x) || isnan(color.y) || isnan(color.z))
        color = vec3f(0.f);

    // Write the calculated radiance
    if (accumId > 0) {
        color = color + vec3f(accumBuffer[offset]);
    }

    accumBuffer[offset] = vec4f(color, 1.f);
    color = (1.f / (accumId + 1)) * color;

    averageBuffer[offset] = vec4f(color, 1.0);
    frameBuffer[offset] = owl::make_rgba(vec3f(linear_to_srgb(color.x),
        linear_to_srgb(color.y),
        linear_to_srgb(color.z)));
}

__device__ __forceinline__
vec3f colorFromSiga(vec3f siga, float beta_n)
{
    float beta_n2 = beta_n * beta_n;
    float beta_n4 = beta_n2 * beta_n2;
    float beta_n3 = beta_n2 * beta_n;
    float beta_n5 = beta_n4 * beta_n;

    float denom = (5.969f - 0.215f * beta_n + 2.532f * beta_n2 -
        10.73f * beta_n3 + 5.574f * beta_n4 +
        0.245f * beta_n5);

    vec3f color(0.f);

    color.x = exp(-owl::sqrt(siga.x) * denom);
    color.y = exp(-owl::sqrt(siga.y) * denom);
    color.z = exp(-owl::sqrt(siga.z) * denom);

    return color;
}

__device__ __forceinline__
vec3f sigaFromColor(vec3f color, float beta_n)
{
    float beta_n2 = beta_n * beta_n;
    float beta_n4 = beta_n2 * beta_n2;
    float beta_n3 = beta_n2 * beta_n;
    float beta_n5 = beta_n4 * beta_n;

    float denom = (5.969f - 0.215f * beta_n + 2.532f * beta_n2 -
        10.73f * beta_n3 + 5.574f * beta_n4 +
        0.245f * beta_n5);

    vec3f siga(0.f);

    siga.x = pow(log(color.x) / denom, 2.f);
    siga.y = pow(log(color.y) / denom, 2.f);
    siga.z = pow(log(color.z) / denom, 2.f);

    return siga;
}

__device__
float luminance(vec3f rgb)
{
    return 0.212671f * rgb[0] + 0.715160f * rgb[1] + 0.072169f * rgb[2];
}

__device__
float nrcSpread(vec3f x1, vec3f x2, float abscos, float pdf)
{
    float lsq = pow(length(x2 - x1), 2);
    return owl::sqrt(lsq / pdf / abscos);
}

__device__
vec3f barycentricInterpolate(vec3f* tex, vec3i index)
{
    float u = optixGetTriangleBarycentrics().x;
    float v = optixGetTriangleBarycentrics().y;

    return (1.f - u - v) * tex[index.x]
        + u * tex[index.y]
        + v * tex[index.z];
}

__device__
vec2f barycentricInterpolate(vec2f* tex, vec3i index)
{
    float u = optixGetTriangleBarycentrics().x;
    float v = optixGetTriangleBarycentrics().y;

    return (1.f - u - v) * tex[index.x]
        + u * tex[index.y]
        + v * tex[index.z];
}


__device__ __host__
vec3f uniformSampleSphere(vec2f rand) {
    float z = 1 - 2 * rand.x;
    float r = owl::sqrt(owl::max(0.f, 1.f - z * z));
    float phi = 2 * Pi * rand.y;
    return normalize(vec3f(r * cos(phi), r * sin(phi), z));
}

__device__
vec3f uniformSampleHemisphere(vec2f rand)
{
    float z = rand.x;
    float r = owl::sqrt(owl::max(0.f, 1.f - z * z));
    float phi = 2.f * Pi * rand.y;

    return normalize(vec3f(r * cos(phi), r * sin(phi), z));
}

__device__
vec2f ConcentricSampleDisk(vec2f rand) {
    // Map uniform random numbers to $[-1,1]^2$
    vec2f uOffset = 2.f * rand - vec2f(1, 1);

    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0) return vec2f(0, 0);

    // Apply concentric mapping to point
    float theta, r;
    if (owl::abs(uOffset.x) > owl::abs(uOffset.y)) {
        r = uOffset.x;
        theta = Pi / 4.f * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = Pi / 2.f - Pi / 4.f * (uOffset.x / uOffset.y);
    }
    return r * vec2f(owl::cos(theta), owl::sin(theta));
}

__device__
vec3f CosineSampleHemisphere(vec2f rand) {
    vec2f d = ConcentricSampleDisk(rand);
    float z = owl::sqrt(owl::max(0.f, 1.f - d.x * d.x - d.y * d.y));
    return normalize(vec3f(d.x, d.y, z));
}

__device__
vec3f apply_mat(vec3f mat[3], vec3f v)
{
    vec3f result(dot(mat[0], v), dot(mat[1], v), dot(mat[2], v));
    return result;
}

__device__
void matrixInverse(vec3f m[3], vec3f minv[3]) {
    int indxc[3], indxr[3];
    int ipiv[3] = { 0, 0, 0 };

    minv[0] = m[0];
    minv[1] = m[1];
    minv[2] = m[2];

    for (int i = 0; i < 3; i++) {
        int irow = 0, icol = 0;
        float big = 0.f;
        // Choose pivot
        for (int j = 0; j < 3; j++) {
            if (ipiv[j] != 1) {
                for (int k = 0; k < 3; k++) {
                    if (ipiv[k] == 0) {
                        if (abs(minv[j][k]) >= big) {
                            big = abs(minv[j][k]);
                            irow = j;
                            icol = k;
                        }
                    }
                }
            }
        }
        ++ipiv[icol];
        // Swap rows _irow_ and _icol_ for pivot
        if (irow != icol) {
            for (int k = 0; k < 3; ++k) {
                float temp = minv[irow][k];
                minv[irow][k] = minv[icol][k];
                minv[icol][k] = temp;
            }
        }
        indxr[i] = irow;
        indxc[i] = icol;

        // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
        float pivinv = 1.f / minv[icol][icol];
        minv[icol][icol] = 1.f;
        for (int j = 0; j < 3; j++) minv[icol][j] *= pivinv;

        // Subtract this row from others to zero out their columns
        for (int j = 0; j < 3; j++) {
            if (j != icol) {
                float save = minv[j][icol];
                minv[j][icol] = 0.f;
                for (int k = 0; k < 3; k++) minv[j][k] -= minv[icol][k] * save;
            }
        }
    }
    // Swap columns to reflect permutation
    for (int j = 2; j >= 0; j--) {
        if (indxr[j] != indxc[j]) {
            for (int k = 0; k < 3; k++) {
                float temp = minv[k][indxr[j]];
                minv[k][indxr[j]] = minv[k][indxc[j]];
                minv[k][indxc[j]] = temp;
            }
        }
    }
}

__device__
void orthonormalBasis(vec3f n, vec3f mat[3], vec3f invmat[3])
{
    vec3f c1, c2, c3;
    if (n.z < -0.999999f)
    {
        c1 = vec3f(0, -1, 0);
        c2 = vec3f(-1, 0, 0);
    }
    else
    {
        float a = 1. / (1. + n.z);
        float b = -n.x * n.y * a;
        c1 = normalize(vec3f(1. - n.x * n.x * a, b, -n.x));
        c2 = normalize(vec3f(b, 1. - n.y * n.y * a, -n.y));
    }
    c3 = n;

    mat[0] = c1;
    mat[1] = c2;
    mat[2] = c3;

    invmat[0] = vec3f(c1.x, c2.x, c3.x);
    invmat[1] = vec3f(c1.y, c2.y, c3.y);
    invmat[2] = vec3f(c1.z, c2.z, c3.z);

    // matrixInverse(mat, invmat);
}

__device__
vec3f samplePointOnTriangle(vec3f v1, vec3f v2, vec3f v3,
    float u1, float u2)
{
    float su1 = owl::sqrt(u1);
    return (1 - su1) * v1 + su1 * ((1 - u2) * v2 + u2 * v3);
}

__device__
float sphericalPhi(vec3f v) {
    float p = atan2(v.y, v.x);
    return (p < 0) ? (p + 2 * Pi) : p;
}

__device__
float sphericalTheta(vec3f p) {
    return acos(p.z);
}

__device__
vec3f sphericalDirection(float r, float theta, float phi) {
    return vec3f(r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta));
}

__device__
float balanceHeuristic(int nf, float fPdf, int ng, float gPdf) {
    return (nf * fPdf) / (nf * fPdf + ng * gPdf);
}

__device__
float powerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf, g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}