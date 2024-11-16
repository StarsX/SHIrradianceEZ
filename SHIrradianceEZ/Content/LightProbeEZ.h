//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#pragma once

#include "Helper/XUSG-EZ.h"

class LightProbeEZ
{
public:
	LightProbeEZ();
	virtual ~LightProbeEZ();

	bool Init(XUSG::CommandList* pCommandList, std::vector<XUSG::Resource::uptr>& uploaders,
		const std::wstring pFileNames[], uint32_t numFiles);

	void UpdateFrame(double time, uint8_t frameIndex);
	void Process(XUSG::EZ::CommandList* pCommandList, uint8_t frameIndex);

	XUSG::Texture::sptr GetRadiance() const;
	XUSG::StructuredBuffer::sptr GetSH() const;

	static const uint8_t FrameCount = 3;
	static const uint8_t CubeMapFaceCount = 6;

protected:
	enum ShaderIndex : uint8_t
	{
		CS_RADIANCE_GEN,
		CS_SH_CUBE_MAP,
		CS_SH_SUM,
		CS_SH_NORMALIZE,

		NUM_SHADER
	};

	bool createShaders();

	void generateRadiance(XUSG::EZ::CommandList* pCommandList, uint8_t frameIndex);
	void shCubeMap(XUSG::EZ::CommandList* pCommandList, uint8_t order);
	void shSum(XUSG::EZ::CommandList* pCommandList, uint8_t order, uint8_t frameIndex);
	void shNormalize(XUSG::EZ::CommandList* pCommandList, uint8_t order);

	XUSG::ShaderLib::uptr m_shaderLib;
	XUSG::Blob m_shaders[NUM_SHADER];

	std::vector<XUSG::Texture::sptr> m_sources;
	XUSG::RenderTarget::sptr	m_radiance;

	XUSG::StructuredBuffer::sptr m_coeffSH[2];
	XUSG::StructuredBuffer::uptr m_weightSH[2];

	XUSG::ConstantBuffer::uptr m_cbPerFrame;

	uint32_t	m_inputProbeIdx;
	uint32_t	m_numSHTexels;
	uint8_t		m_shBufferParity;
};
