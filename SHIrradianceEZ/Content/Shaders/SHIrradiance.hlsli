//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#ifndef PI
#define PI 3.1415926535897
#endif
#define SH_NUM_COEFF (SH_ORDER * SH_ORDER)

#define c1 g_sh_c1
#define c2 g_sh_c2
#define c3 g_sh_c3
#define c4 g_sh_c4

// irradiance = max(0.0, \
// (c1 * (x * x - y * y)) * shCoeffs[8]												// c1.L22.(x^2 - y^2)
// + (c3 * (3.0 * z * z - 1.0)) * shCoeffs[6]										// c3.L20.(3.z^2 - 1)
// + c4 * shCoeffs[0]																// c4.L00 
// + 2.0 * c1 * (shCoeffs[4] * x * y + shCoeffs[7] * x * z + shCoeffs[5] * y * z)	// 2.c1.(L2-2.xy + L21.xz + L2-1.yz)
// + 2.0 * c2 * (shCoeffs[3] * x + shCoeffs[1] * y + shCoeffs[2] * z));				// 2.c2.(L11.x + L1-1.y + L10.z)
#define EVALUATE_SH_IRRADIANCE(irradiance, shCoeffs, norm) \
{ \
	const float x = -norm.x;	\
	const float y = -norm.y;	\
	const float z = norm.z;		\
	\
	irradiance = max(0.0, \
	(c1 * (x * x - y * y)) * shCoeffs[8]											\
	+ (c3 * (3.0 * z * z - 1.0)) * shCoeffs[6]										\
	+ c4 * shCoeffs[0]																\
	+ 2.0 * c1 * (shCoeffs[4] * x * y + shCoeffs[7] * x * z + shCoeffs[5] * y * z)	\
	+ 2.0 * c2 * (shCoeffs[3] * x + shCoeffs[1] * y + shCoeffs[2] * z));			\
	\
	irradiance /= PI; \
}

static const float g_sh_c1 = 0.42904276540489171563379376569857; // 4 * A2.Y22 = 1/16 * sqrt(15.PI)
static const float g_sh_c2 = 0.51166335397324424423977581244463; // 0.5 * A1.Y10 = 1/2 * sqrt(PI/3)
static const float g_sh_c3 = 0.24770795610037568833406429782001; // A2.Y20 = 1/16 * sqrt(5.PI)
static const float g_sh_c4 = 0.88622692545275801364908374167057; // A0.Y00 = 1/2 * sqrt(PI)

//--------------------------------------------------------------------------------------
// Load spherical harmonics
//--------------------------------------------------------------------------------------
void LoadSH(out float3 shCoeffs[SH_NUM_COEFF], StructuredBuffer<float3> roSHCoeffs)
{
	[unroll]
	for (uint i = 0; i < SH_NUM_COEFF; ++i) shCoeffs[i] = roSHCoeffs[i];
}

//--------------------------------------------------------------------------------------
// Evaluate irradiance using spherical harmonics
//--------------------------------------------------------------------------------------
float3 EvaluateSHIrradiance(float3 shCoeffs[SH_NUM_COEFF], float3 norm)
{
	float3 irradiance;
	EVALUATE_SH_IRRADIANCE(irradiance, shCoeffs, norm);

	return irradiance;
}

float3 EvaluateSHIrradiance(StructuredBuffer<float3> roSHCoeffs, float3 norm)
{
	float3 irradiance;
	EVALUATE_SH_IRRADIANCE(irradiance, roSHCoeffs, norm);

	return irradiance;
}

#undef c4
#undef c3
#undef c2
#undef c1
