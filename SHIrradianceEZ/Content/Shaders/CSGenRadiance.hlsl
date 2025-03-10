//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "CubeMap.hlsli"

//--------------------------------------------------------------------------------------
// Constant buffer
//--------------------------------------------------------------------------------------
cbuffer cbPerFrame : register (b1)
{
	float g_blend;
};

//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
TextureCube<float3>			g_txSources[2];
RWTexture2DArray<float3>	g_rwDest;

//--------------------------------------------------------------------------------------
// Texture sampler
//--------------------------------------------------------------------------------------
SamplerState	g_smpLinear;

//--------------------------------------------------------------------------------------
// Compute shader
//--------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	const float3 uv = GetCubeTexcoord(DTid, g_rwDest);
	const float3 source1 = g_txSources[0].SampleLevel(g_smpLinear, uv, 0.0);
	const float3 source2 = g_txSources[1].SampleLevel(g_smpLinear, uv, 0.0);

	g_rwDest[DTid] = lerp(source1, source2, g_blend);
}
