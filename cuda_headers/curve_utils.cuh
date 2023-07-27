// ======================================================================== //
// Copyright (c) Meta Platforms, Inc. and affiliates.                       //
//                                                                          //
// This source code is licensed under the MIT license found in the          //
// LICENSE file in the root directory of this source tree.                  //
// ======================================================================== //

#pragma once

using namespace owl;

struct LinearBSplineSegment
{
    __device__ __forceinline__ LinearBSplineSegment() {}
    __device__ __forceinline__ LinearBSplineSegment(const vec4f* q) { initialize(q); }

    __device__ __forceinline__ void initialize(const vec4f* q)
    {
        p[0] = q[0];
        p[1] = q[1] - q[0];  // pre-transform p[] for fast evaluation
    }

    __device__ __forceinline__ float radius(const float& u) const { return p[0].w + p[1].w * u; }

    __device__ __forceinline__ vec3f position3(float u) const { return (vec3f&)p[0] + u * (vec3f&)p[1]; }
    __device__ __forceinline__ vec4f position4(float u) const { return p[0] + u * p[1]; }

    __device__ __forceinline__ float min_radius(float u1, float u2) const
    {
        return fminf(radius(u1), radius(u2));
    }

    __device__ __forceinline__ float max_radius(float u1, float u2) const
    {
        if (!p[1].w)
            return p[0].w;  // a quick bypass for constant width
        return fmaxf(radius(u1), radius(u2));
    }

    __device__ __forceinline__ vec3f velocity3(float u) const { return (vec3f&)p[1]; }
    __device__ __forceinline__ vec4f velocity4(float u) const { return p[1]; }

    __device__ __forceinline__ vec3f acceleration3(float u) const { return vec3f(0.f); }
    __device__ __forceinline__ vec4f acceleration4(float u) const { return vec4f(0.f); }

    __device__ __forceinline__ float derivative_of_radius(float u) const { return p[1].w; }

    vec4f p[2];  // pre-transformed "control points" for fast evaluation
};

//
// Third order polynomial interpolator
//
// Storing {p0, p1, p2, p3} for evaluation:
//     P(u) = p0 * u^3 + p1 * u^2 + p2 * u + p3
//
struct CubicInterpolator
{
    __device__ __host__
    __forceinline__ CubicInterpolator() {}

    __device__ __host__
    __forceinline__ void initializeFromCatrom(const vec4f* q)
    {
        // Catrom-to-Poly = Matrix([[-1/2, 3/2, -3/2,  1/2],
        //                          [1,   -5/2,    2, -1/2],
        //                          [-1/2,   0,  1/2,    0],
        //                          [0,      1,    0,    0]])
        p[0] = (-1.0f * q[0] + (3.0f) * q[1] + (-3.0f) * q[2] + (1.0f) * q[3]) / 2.0f;
        p[1] = (2.0f * q[0] + (-5.0f) * q[1] + (4.0f) * q[2] + (-1.0f) * q[3]) / 2.0f;
        p[2] = (-1.0f * q[0] + (1.0f) * q[2]) / 2.0f;
        p[3] = ((2.0f) * q[1]) / 2.0f;
    }

    __device__ __host__
    __forceinline__ vec4f position4(float u) const
    {
        return (((p[0] * u) + p[1]) * u + p[2]) * u + p[3]; // Horner scheme
    }

    __device__ __host__
    __forceinline__ vec3f position3(float u) const
    {
        // rely on compiler and inlining for dead code removal
        return vec3f(position4(u));
    }
    __device__ __host__
    __forceinline__ float radius(float u) const
    {
        return position4(u).w;
    }

    __device__ __host__
    __forceinline__ vec4f velocity4(float u) const
    {
        // adjust u to avoid problems with tripple knots.
        if (u == 0)
            u = 0.000001f;
        if (u == 1)
            u = 0.999999f;
        return ((3.0f * p[0] * u) + 2.0f * p[1]) * u + p[2];
    }

    __device__ __host__
    __forceinline__ vec3f velocity3(float u) const
    {
        return vec3f(velocity4(u));
    }

    __device__ __host__
    __forceinline__ float derivative_of_radius(float u) const
    {
        return velocity4(u).w;
    }

    __device__ __host__
    __forceinline__ vec4f acceleration4(float u) const
    {
        return 6.0f * p[0] * u + 2.0f * p[1]; // Horner scheme
    }

    __device__ __host__
    __forceinline__ vec3f acceleration3(float u) const
    {
        return vec3f(acceleration4(u));
    }

    vec4f p[4];
};

/*! stolen from optixHair sample in OptiX 7.4 SDK */
// Get curve hit-point in world coordinates.
static __forceinline__ __device__ vec3f getHitPoint()
{
    const float  t = optixGetRayTmax();
    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 rayDirection = optixGetWorldRayDirection();

    return (vec3f)rayOrigin + t * (vec3f)rayDirection;
}

// Compute curve primitive surface normal in object space.
//
// Template parameters:
//   CurveType - A B-Spline evaluator class.
//
// Parameters:
//   bc - A B-Spline evaluator object.
//   u  - segment parameter of hit-point.
//   ps - hit-point on curve's surface in object space; usually
//        computed like this.
//        float3 ps = ray_orig + t_hit * ray_dir;
//        the resulting point is slightly offset away from the
//        surface. For this reason (Warning!) ps gets modified by this
//        method, projecting it onto the surface
//        in case it is not already on it. (See also inline
//        comments.)
//
template <typename CurveType>
__device__ __forceinline__ vec3f surfaceNormal(const CurveType& bc, float u, vec3f& ps)
{
    vec3f normal;
    if (u == 0.0f)
    {
        normal = -bc.velocity3(0);  // special handling for flat endcaps
    }
    else if (u == 1.0f)
    {
        normal = bc.velocity3(1);   // special handling for flat endcaps
    }
    else
    {
        // ps is a point that is near the curve's offset surface,
        // usually ray.origin + ray.direction * rayt.
        // We will push it exactly to the surface by projecting it to the plane(p,d).
        // The function derivation:
        // we (implicitly) transform the curve into coordinate system
        // {p, o1 = normalize(ps - p), o2 = normalize(curve'(t)), o3 = o1 x o2} in which
        // curve'(t) = (0, length(d), 0); ps = (r, 0, 0);
        vec4f p4 = bc.position4(u);
        vec3f p = vec3f(p4);
        float  r = p4.w;  // == length(ps - p) if ps is already on the surface
        vec4f d4 = bc.velocity4(u);
        vec3f d = vec3f(d4);
        float  dr = d4.w;
        float  dd = dot(d, d);

        vec3f o1 = ps - p;               // dot(modified_o1, d) == 0 by design:
        o1 -= (dot(o1, d) / dd) * d;  // first, project ps to the plane(p,d)
        o1 *= r / length(o1);           // and then drop it to the surface
        ps = p + o1;                      // fine-tuning the hit point

        dd -= dot(bc.acceleration3(u), o1);
        normal = dd * o1 - (dr * r) * d;
    }
    return normalize(normal);
}

// Compute curve primitive tangent in object space.
//
// Template parameters:
//   CurveType - A B-Spline evaluator class.
//
// Parameters:
//   bc - A B-Spline evaluator object.
//   u  - segment parameter of tangent location on curve.
//
template <typename CurveType>
__device__ __host__
__forceinline__ vec3f curveTangent(const CurveType& bc, float u)
{
    vec3f tangent = bc.velocity3(u);
    return normalize(tangent);
}

/*! compute normal - stolen from optixHair sample in OptiX 7.4 SDK */
// Compute surface normal of quadratic pimitive in world space.
static __forceinline__ __device__
void computeCurveIntersection(const int primitiveIndex, vec3f& p, vec3f& n, vec3f& t, vec3f& c, float& radius)
{
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int           gasSbtIndex = optixGetSbtGASIndex();

    float u = optixGetCurveParameter();

#define CURVE_CATMULL_ROM
#ifdef CURVE_CATMULL_ROM
    vec4f controlPoints[4];
    CubicInterpolator interpolator;
    optixGetCatmullRomVertexData(gas, primitiveIndex, gasSbtIndex, 0.0f, (float4*)controlPoints);
    interpolator.initializeFromCatrom(controlPoints);

    c = interpolator.position3(u);

    p = getHitPoint();
    radius = interpolator.radius(u);

    n = surfaceNormal(interpolator, u, p);
    n = normalize(n);

    t = curveTangent(interpolator, u);
#else
    vec4f controlPoints[2];
    optixGetLinearCurveVertexData(gas, primitiveIndex, gasSbtIndex, 0.0f, (float4*)controlPoints);
    LinearBSplineSegment interpolator(controlPoints);

    p = getHitPoint();
    radius = interpolator.radius(u);

    n = surfaceNormal(interpolator, u, p);
    n = normalize(n);

    t = (vec3f)(controlPoints[1] - controlPoints[0]);
    t = normalize(t);
#endif
#undef CURVE_CATMULL_ROM
}