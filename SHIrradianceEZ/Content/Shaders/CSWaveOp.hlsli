//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Group-shared memory
//--------------------------------------------------------------------------------------
#if !defined(__SHADER_TARGET_MAJOR) || __SHADER_TARGET_MAJOR <= 5 || SH_GROUP_SIZE > SH_WAVE_SIZE
groupshared float4 g_smem[SH_WAVE_SIZE];
#endif

#if defined(__SHADER_TARGET_MAJOR) && __SHADER_TARGET_MAJOR > 5
//--------------------------------------------------------------------------------------
// Native WaveActiveSum for SM6+
//--------------------------------------------------------------------------------------
float WaveLaneSum(uint laneId, float expr)
{
	return WaveActiveSum(expr);
}

float2 WaveLaneSum(uint laneId, float2 expr)
{
	return WaveActiveSum(expr);
}

float3 WaveLaneSum(uint laneId, float3 expr)
{
	return WaveActiveSum(expr);
}

float4 WaveLaneSum(uint laneId, float4 expr)
{
	return WaveActiveSum(expr);
}
#else
//--------------------------------------------------------------------------------------
// WaveActiveSum emulation for SM6-
//--------------------------------------------------------------------------------------
float WaveLaneSum(uint laneId, float expr)
{
	const uint waveBits = firstbithigh(SH_WAVE_SIZE);

	g_smem[laneId].x = expr;
	GroupMemoryBarrierWithGroupSync();

	[unroll]
	for (uint i = 0; i < waveBits; ++i)
	{
		const uint s = 1 << i;
		g_smem[laneId].x += g_smem[laneId + s].x;
		GroupMemoryBarrierWithGroupSync();
	}

	return g_smem[0].x;
}

float2 WaveLaneSum(uint laneId, float2 expr)
{
	const uint waveBits = firstbithigh(SH_WAVE_SIZE);

	g_smem[laneId].xy = expr;
	GroupMemoryBarrierWithGroupSync();

	[unroll]
	for (uint i = 0; i < waveBits; ++i)
	{
		const uint s = 1 << i;
		g_smem[laneId].xy += g_smem[laneId + s].xy;
		GroupMemoryBarrierWithGroupSync();
	}

	return g_smem[0].xy;
}

float3 WaveLaneSum(uint laneId, float3 expr)
{
	const uint waveBits = firstbithigh(SH_WAVE_SIZE);

	g_smem[laneId].xyz = expr;
	GroupMemoryBarrierWithGroupSync();

	[unroll]
	for (uint i = 0; i < waveBits; ++i)
	{
		const uint s = 1 << i;
		g_smem[laneId].xyz += g_smem[laneId + s].xyz;
		GroupMemoryBarrierWithGroupSync();
	}

	return g_smem[0].xyz;
}

float4 WaveLaneSum(uint laneId, float4 expr)
{
	const uint waveBits = firstbithigh(SH_WAVE_SIZE);

	g_smem[laneId] = expr;
	GroupMemoryBarrierWithGroupSync();

	[unroll]
	for (uint i = 0; i < waveBits; ++i)
	{
		const uint s = 1 << i;
		g_smem[laneId] += g_smem[laneId + s];
		GroupMemoryBarrierWithGroupSync();
	}

	return g_smem[0];
}

uint WaveGetLaneCount()
{
	return SH_WAVE_SIZE;
}
#endif
