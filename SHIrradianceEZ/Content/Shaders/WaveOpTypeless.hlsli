//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#ifndef T
#define T float4
#endif

#ifndef SWZ
#define SWZ xyzw
#endif

#ifndef _BROAD_CAST_
#define _BROAD_CAST_ 0
#endif

#if _BROAD_CAST_
#define WAVE_LANES_SUM WaveLanesSumBroadCast
#else
#define WAVE_LANES_SUM WaveLanesSum
#endif

//--------------------------------------------------------------------------------------
// WaveActiveSum emulation for SM6-
//--------------------------------------------------------------------------------------
T WAVE_LANES_SUM(uint laneId, T expr)
{
	const uint waveBits = firstbithigh(SH_WAVE_SIZE);
	g_smem[laneId].SWZ = expr;

	[unroll]
	for (uint i = 0; i < waveBits; ++i)
	{
		const uint s = 1 << i;
		GroupMemoryBarrierWithGroupSync();
		g_smem[laneId].SWZ += laneId % s ? 0.0 : g_smem[laneId + s].SWZ;
	}

#if _BROAD_CAST_
	GroupMemoryBarrierWithGroupSync();
#endif

	return g_smem[0].SWZ;
}

#undef WAVE_LANES_SUM
