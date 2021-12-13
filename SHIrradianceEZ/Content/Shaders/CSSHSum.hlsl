//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "SHMath.hlsli"

cbuffer cb
{
	uint g_order;
	uint g_pixelCount;
};

//--------------------------------------------------------------------------------------
// Buffers
//--------------------------------------------------------------------------------------
RWStructuredBuffer<float3> g_rwSHBuff;
RWStructuredBuffer<float> g_rwWeight;
StructuredBuffer<float3> g_roSHBuff;
StructuredBuffer<float> g_roWeight;

groupshared float3 g_smem[32];

//--------------------------------------------------------------------------------------
// Compute shader
//--------------------------------------------------------------------------------------
[numthreads(1024, 1, 1)]
void main(uint2 DTid : SV_DispatchThreadID, uint GTid : SV_GroupThreadID, uint Gid : SV_GroupID)
{
	const uint n = g_order * g_order;
	float sumSH = 0.0, weight = 0.0;

	if (DTid.x < g_pixelCount)
	{
		float3 sh = g_roSHBuff[GetLocation(n, DTid)];
		sh = WaveActiveSum(sh);
		if (WaveIsFirstLane()) g_smem[GTid / WaveGetLaneCount()] = sh;

		GroupMemoryBarrierWithGroupSync();

		if (GTid < WaveGetLaneCount())
		{
			sh = g_smem[GTid];
			sh = WaveActiveSum(sh);
			if (GTid == 0) g_rwSHBuff[GetLocation(n, uint2(Gid, DTid.y))] = sh;
		}

		GroupMemoryBarrierWithGroupSync();

		if (DTid.y == 0)
		{
			float wt = g_roWeight[DTid.x];
			wt = WaveActiveSum(wt);
			if (WaveIsFirstLane()) g_smem[GTid / WaveGetLaneCount()].x = wt;
			
			GroupMemoryBarrierWithGroupSync();

			if (GTid < WaveGetLaneCount())
			{
				wt = g_smem[GTid].x;
				wt = WaveActiveSum(wt);
				if (GTid == 0) g_rwWeight[Gid] = wt;
			}
		}
	}
}
