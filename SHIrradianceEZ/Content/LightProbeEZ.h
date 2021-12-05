//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#pragma once

#include "DXFramework.h"
#include "Helper/XUSG-EZ.h"

class LightProbeEZ
{
public:
	LightProbeEZ(const XUSG::Device::sptr& device);
	virtual ~LightProbeEZ();

	bool Init(XUSG::CommandList* pCommandList, uint32_t width, uint32_t height,
		std::vector<XUSG::Resource::uptr>& uploaders, const std::wstring pFileNames[],
		uint32_t numFiles, bool typedUAV);

	void UpdateFrame(double time, uint8_t frameIndex);
	void Process(XUSG::EZ::CommandList* pCommandList, uint8_t frameIndex);

	XUSG::ShaderResource* GetRadiance() const;
	XUSG::StructuredBuffer::sptr GetSH() const;

	static const uint8_t FrameCount = 3;
	static const uint8_t CubeMapFaceCount = 6;

protected:
	enum ShaderIndex : uint8_t
	{
		VS_SCREEN_QUAD,
		PS_GEN_RADIANCE,
		CS_GEN_RADIANCE,
		CS_SH_CUBE_MAP,
		CS_SH_SUM,
		CS_SH_NORMALIZE,

		NUM_SHADER
	};

	bool createShaders();

	void generateRadianceGraphics(XUSG::EZ::CommandList* pCommandList, uint8_t frameIndex);
	void generateRadianceCompute(XUSG::EZ::CommandList* pCommandList, uint8_t frameIndex);
	void shCubeMap(XUSG::EZ::CommandList* pCommandList, uint8_t order);
	void shSum(XUSG::EZ::CommandList* pCommandList, uint8_t order, uint8_t frameIndex);
	void shNormalize(XUSG::EZ::CommandList* pCommandList, uint8_t order);

	XUSG::Device::sptr m_device;

	XUSG::ShaderPool::uptr m_shaderPool;
	XUSG::Blob m_shaders[NUM_SHADER];

	std::vector<XUSG::Texture::sptr> m_sources;
	XUSG::RenderTarget::uptr	m_radiance;

	XUSG::StructuredBuffer::sptr m_coeffSH[2];
	XUSG::StructuredBuffer::uptr m_weightSH[2];

	XUSG::ConstantBuffer::uptr m_cbPerFrame;
	XUSG::ConstantBuffer::uptr m_cbCubeMapSlices;
	XUSG::ConstantBuffer::uptr m_cbSHCubeMap;
	XUSG::ConstantBuffer::uptr m_cbSHSums;

	uint32_t	m_inputProbeIdx;
	uint32_t	m_numSHTexels;
	uint8_t		m_shBufferParity;
};
