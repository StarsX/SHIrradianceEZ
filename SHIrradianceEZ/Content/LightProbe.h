//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#pragma once

#include "Helper/XUSG-EZ.h"

class LightProbe
{
public:
	LightProbe();
	virtual ~LightProbe();

	bool Init(XUSG::CommandList* pCommandList, const XUSG::DescriptorTableLib::sptr& descriptorTableLib,
		std::vector<XUSG::Resource::uptr>& uploaders, const std::wstring pFileNames[], uint32_t numFiles);
	bool CreateDescriptorTables(XUSG::Device* pDevice);

	void UpdateFrame(double time, uint8_t frameIndex);
	void Process(XUSG::CommandList* pCommandList, uint8_t frameIndex);

	XUSG::ShaderResource* GetRadiance() const;
	XUSG::StructuredBuffer::sptr GetSH() const;

	static const uint8_t FrameCount = 3;
	static const uint8_t CubeMapFaceCount = 6;

protected:
	enum PipelineIndex : uint8_t
	{
		RADIANCE_GEN,
		SH_CUBE_MAP,
		SH_SUM,
		SH_NORMALIZE,

		NUM_PIPELINE
	};

	enum SrvTableIndex : uint8_t
	{
		SRV_TABLE_INPUT,
		SRV_TABLE_RADIANCE,

		NUM_SRV
	};

	bool createPipelineLayouts();
	bool createPipelines(XUSG::Format rtFormat);
	bool createDescriptorTables();

	void generateRadiance(XUSG::CommandList* pCommandList, uint8_t frameIndex);
	void shCubeMap(XUSG::CommandList* pCommandList, uint8_t order);
	void shSum(XUSG::CommandList* pCommandList, uint8_t order);
	void shNormalize(XUSG::CommandList* pCommandList, uint8_t order);

	XUSG::ShaderLib::uptr				m_shaderLib;
	XUSG::Graphics::PipelineLib::uptr	m_graphicsPipelineLib;
	XUSG::Compute::PipelineLib::uptr	m_computePipelineLib;
	XUSG::PipelineLayoutLib::uptr		m_pipelineLayoutLib;
	XUSG::DescriptorTableLib::sptr		m_descriptorTableLib;

	XUSG::PipelineLayout	m_pipelineLayouts[NUM_PIPELINE];
	XUSG::Pipeline			m_pipelines[NUM_PIPELINE];

	std::vector<XUSG::DescriptorTable> m_srvTables[NUM_SRV];
	XUSG::DescriptorTable	m_uavTable;
	XUSG::DescriptorTable	m_samplerTable;

	std::vector<XUSG::Texture::sptr> m_sources;
	XUSG::RenderTarget::uptr	m_radiance;

	XUSG::StructuredBuffer::sptr m_coeffSH[2];
	XUSG::StructuredBuffer::uptr m_weightSH[2];

	XUSG::ConstantBuffer::uptr	m_cbPerFrame;

	uint32_t				m_inputProbeIdx;
	uint32_t				m_numSHTexels;
	uint8_t					m_shBufferParity;
};
