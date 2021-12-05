//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#pragma once

#include "Helper/XUSG-EZ.h"

class RendererEZ
{
public:
	RendererEZ(const XUSG::Device::sptr& device);
	virtual ~RendererEZ();

	bool Init(XUSG::CommandList* pCommandList, uint32_t width, uint32_t height,
		std::vector<XUSG::Resource::uptr>& uploaders, const char* fileName,
		const DirectX::XMFLOAT4& posScale = DirectX::XMFLOAT4(0.0f, 0.0f, 0.0f, 1.0f));

	void SetLightProbe(const XUSG::Texture::sptr& radiance);
	void SetLightProbesSH(const XUSG::StructuredBuffer::sptr& coeffSH);
	void UpdateFrame(uint8_t frameIndex, DirectX::CXMVECTOR eyePt,
		DirectX::CXMMATRIX viewProj, float glossy, bool isPaused);
	void Render(XUSG::EZ::CommandList* pCommandList, uint8_t frameIndex, bool needClear = false);
	void Postprocess(XUSG::EZ::CommandList* pCommandList, XUSG::RenderTarget* pRenderTarget);

	static const uint8_t FrameCount = 3;

protected:
	enum ShaderIndex : uint8_t
	{
		VS_BASE_PASS,
		VS_SCREEN_QUAD,
		PS_BASE_PASS,
		PS_ENVIRONMENT,
		PS_POSTPROCESS,
		CS_TEMPORAL_AA,

		NUM_SHADER
	};

	enum RenderTargetIndex : uint8_t
	{
		RT_COLOR,
		RT_VELOCITY,

		NUM_RENDER_TARGET
	};

	enum OutputView : uint8_t
	{
		UAV_PP_TAA,
		UAV_PP_TAA1,

		NUM_OUTPUT_VIEW
	};

	bool createVB(XUSG::CommandList* pCommandList, uint32_t numVert,
		uint32_t stride, const uint8_t* pData, std::vector<XUSG::Resource::uptr>& uploaders);
	bool createIB(XUSG::CommandList* pCommandList, uint32_t numIndices,
		const uint32_t* pData, std::vector<XUSG::Resource::uptr>& uploaders);
	bool createShaders();

	void createInputLayout();
	void render(XUSG::EZ::CommandList* pCommandList, uint8_t frameIndex, bool needClear);
	void environment(XUSG::EZ::CommandList* pCommandList, uint8_t frameIndex);
	void temporalAA(XUSG::EZ::CommandList* pCommandList);

	XUSG::Device::sptr m_device;

	uint32_t	m_numIndices;
	uint8_t		m_frameParity;

	DirectX::XMUINT2	m_viewport;
	DirectX::XMFLOAT4	m_posScale;
	DirectX::XMFLOAT4X4	m_worldViewProj;

	XUSG::InputLayout m_inputLayout;

	XUSG::Texture::sptr m_radiance;
	XUSG::StructuredBuffer::sptr m_coeffSH;

	XUSG::VertexBuffer::uptr	m_vertexBuffer;
	XUSG::IndexBuffer::uptr		m_indexBuffer;

	XUSG::RenderTarget::uptr	m_renderTargets[NUM_RENDER_TARGET];
	XUSG::Texture2D::uptr		m_outputViews[NUM_OUTPUT_VIEW];
	XUSG::DepthStencil::uptr	m_depth;

	XUSG::ConstantBuffer::uptr	m_cbBasePass;
	XUSG::ConstantBuffer::uptr	m_cbPerFrame;

	XUSG::ShaderPool::uptr		m_shaderPool;
	XUSG::Blob m_shaders[NUM_SHADER];
};
