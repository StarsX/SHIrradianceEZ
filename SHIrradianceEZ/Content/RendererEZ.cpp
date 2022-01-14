//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "DXFrameworkHelper.h"
#include "Optional/XUSGObjLoader.h"
#include "RendererEZ.h"

using namespace std;
using namespace DirectX;
using namespace XUSG;

struct CBBasePass
{
	XMFLOAT4X4	WorldViewProj;
	XMFLOAT4X4	WorldViewProjPrev;
	XMFLOAT3X4	World;
	XMFLOAT2	ProjBias;
};

struct CBPerFrame
{
	XMFLOAT4	EyePtGlossy;
	XMFLOAT4X4	ScreenToWorld;
};

RendererEZ::RendererEZ() :
	m_frameParity(0)
{
	m_shaderPool = ShaderPool::MakeUnique();
}

RendererEZ::~RendererEZ()
{
}

bool RendererEZ::Init(CommandList* pCommandList, uint32_t width, uint32_t height,
	vector<Resource::uptr>& uploaders, const char* fileName, const XMFLOAT4& posScale)
{
	const auto pDevice = pCommandList->GetDevice();
	m_viewport = XMUINT2(width, height);
	m_posScale = posScale;

	// Load inputs
	ObjLoader objLoader;
	if (!objLoader.Import(fileName, true, true)) return false;
	N_RETURN(createVB(pCommandList, objLoader.GetNumVertices(), objLoader.GetVertexStride(), objLoader.GetVertices(), uploaders), false);
	N_RETURN(createIB(pCommandList, objLoader.GetNumIndices(), objLoader.GetIndices(), uploaders), false);

	// Create output views
	// Render targets
	for (auto& renderTarget : m_renderTargets) renderTarget = RenderTarget::MakeUnique();
	m_renderTargets[RT_COLOR]->Create(pDevice, width, height, Format::R16G16B16A16_FLOAT,
		1, ResourceFlag::NONE, 1, 1, nullptr, false, MemoryFlag::NONE, L"Color");
	m_renderTargets[RT_VELOCITY]->Create(pDevice, width, height, Format::R16G16_FLOAT,
		1, ResourceFlag::NONE, 1, 1, nullptr, false, MemoryFlag::NONE, L"Velocity");

	m_depth = DepthStencil::MakeUnique();
	m_depth->Create(pDevice, width, height, Format::D24_UNORM_S8_UINT,
		ResourceFlag::DENY_SHADER_RESOURCE, 1, 1, 1, 1.0f, 0, false,
		MemoryFlag::NONE, L"Depth");

	// Temporal AA
	for (auto& outView : m_outputViews) outView = Texture2D::MakeUnique();
	m_outputViews[UAV_PP_TAA]->Create(pDevice, width, height, Format::R16G16B16A16_FLOAT, 1,
		ResourceFlag::ALLOW_UNORDERED_ACCESS, 1, 1, false, MemoryFlag::NONE, L"TemporalAAOut0");
	m_outputViews[UAV_PP_TAA1]->Create(pDevice, width, height, Format::R16G16B16A16_FLOAT, 1,
		ResourceFlag::ALLOW_UNORDERED_ACCESS, 1, 1, false, MemoryFlag::NONE, L"TemporalAAOut1");

	// Create constant buffers
	m_cbBasePass = ConstantBuffer::MakeUnique();
	N_RETURN(m_cbBasePass->Create(pDevice, sizeof(CBBasePass[FrameCount]), FrameCount,
		nullptr, MemoryType::UPLOAD, MemoryFlag::NONE, L"CBBasePass"), false);

	m_cbPerFrame = ConstantBuffer::MakeUnique();
	N_RETURN(m_cbPerFrame->Create(pDevice, sizeof(CBPerFrame[FrameCount]), FrameCount,
		nullptr, MemoryType::UPLOAD, MemoryFlag::NONE, L"CBPerFrame"), false);

	// Create shaders and input layout
	N_RETURN(createShaders(), false);
	createInputLayout();

	return true;
}

void RendererEZ::SetLightProbe(const Texture::sptr& radiance)
{
	m_radiance = radiance;
}

void RendererEZ::SetLightProbesSH(const StructuredBuffer::sptr& coeffSH)
{
	m_coeffSH = coeffSH;
}

static const XMFLOAT2& IncrementalHalton()
{
	static auto haltonBase = XMUINT2(0, 0);
	static auto halton = XMFLOAT2(0.0f, 0.0f);

	// Base 2
	{
		// Bottom bit always changes, higher bits
		// Change less frequently.
		auto change = 0.5f;
		auto oldBase = haltonBase.x++;
		auto diff = haltonBase.x ^ oldBase;

		// Diff will be of the form 0*1+, i.e. one bits up until the last carry.
		// Expected iterations = 1 + 0.5 + 0.25 + ... = 2
		do
		{
			halton.x += (oldBase & 1) ? -change : change;
			change *= 0.5f;

			diff = diff >> 1;
			oldBase = oldBase >> 1;
		} while (diff);
	}

	// Base 3
	{
		const auto oneThird = 1.0f / 3.0f;
		auto mask = 0x3u;	// Also the max base 3 digit
		auto add = 0x1u;	// Amount to add to force carry once digit == 3
		auto change = oneThird;
		++haltonBase.y;

		// Expected iterations: 1.5
		while (true)
		{
			if ((haltonBase.y & mask) == mask)
			{
				haltonBase.y += add;	// Force carry into next 2-bit digit
				halton.y -= 2 * change;

				mask = mask << 2;
				add = add << 2;

				change *= oneThird;
			}
			else
			{
				halton.y += change;	// We know digit n has gone from a to a + 1
				break;
			}
		};
	}

	return halton;
}

void RendererEZ::UpdateFrame(uint8_t frameIndex, CXMVECTOR eyePt, CXMMATRIX viewProj, float glossy, bool isPaused)
{
	{
		static auto angle = 0.0f;
		angle += !isPaused ? 0.1f * XM_PI / 180.0f : 0.0f;
		const auto rot = XMMatrixRotationY(angle);

		const auto world = XMMatrixScaling(m_posScale.w, m_posScale.w, m_posScale.w) * rot *
			XMMatrixTranslation(m_posScale.x, m_posScale.y, m_posScale.z);

		const auto halton = IncrementalHalton();
		XMFLOAT2 jitter =
		{
			(halton.x * 2.0f - 1.0f) / m_viewport.x,
			(halton.y * 2.0f - 1.0f) / m_viewport.y
		};

		const auto pCbData = reinterpret_cast<CBBasePass*>(m_cbBasePass->Map(frameIndex));
		pCbData->ProjBias = jitter;
		pCbData->WorldViewProjPrev = m_worldViewProj;
		XMStoreFloat4x4(&pCbData->WorldViewProj, XMMatrixTranspose(world * viewProj));
		XMStoreFloat3x4(&pCbData->World, world);
		m_worldViewProj = pCbData->WorldViewProj;
	}

	{
		const auto pCbData = reinterpret_cast<CBPerFrame*>(m_cbPerFrame->Map(frameIndex));
		const auto projToWorld = XMMatrixInverse(nullptr, viewProj);
		XMStoreFloat4x4(&pCbData->ScreenToWorld, XMMatrixTranspose(projToWorld));
		XMStoreFloat4(&pCbData->EyePtGlossy, eyePt);
		pCbData->EyePtGlossy.w = glossy;
	}

	m_frameParity = !m_frameParity;
}

void RendererEZ::Render(EZ::CommandList* pCommandList, uint8_t frameIndex, bool needClear)
{
	render(pCommandList, frameIndex, needClear);
	environment(pCommandList, frameIndex);
	temporalAA(pCommandList);
}

void RendererEZ::Postprocess(EZ::CommandList* pCommandList, RenderTarget* pRenderTarget)
{
	// Set pipeline state
	const auto state = pCommandList->GetGraphicsPipelineState();
	state->SetShader(Shader::Stage::VS, m_shaders[VS_SCREEN_QUAD]);
	state->SetShader(Shader::Stage::PS, m_shaders[PS_POSTPROCESS]);
	state->IASetPrimitiveTopologyType(PrimitiveTopologyType::TRIANGLE);
	state->OMSetNumRenderTargets(1);
	state->OMSetRTVFormat(0, pRenderTarget->GetFormat());
	pCommandList->DSSetState(Graphics::DEPTH_STENCIL_NONE);

	// Set render target
	const auto rtv = EZ::GetRTV(pRenderTarget);
	pCommandList->OMSetRenderTargets(1, &rtv);

	// Set SRV
	const auto srv = EZ::GetSRV(m_outputViews[UAV_PP_TAA + m_frameParity].get());
	pCommandList->SetGraphicsResources(Shader::Stage::PS, DescriptorType::SRV, 0, 1, &srv);

	// Set viewport
	Viewport viewport(0.0f, 0.0f, static_cast<float>(m_viewport.x), static_cast<float>(m_viewport.y));
	RectRange scissorRect(0, 0, m_viewport.x, m_viewport.y);
	pCommandList->RSSetViewports(1, &viewport);
	pCommandList->RSSetScissorRects(1, &scissorRect);

	pCommandList->IASetPrimitiveTopology(PrimitiveTopology::TRIANGLELIST);
	pCommandList->Draw(3, 1, 0, 0);
}

bool RendererEZ::createVB(CommandList* pCommandList, uint32_t numVert,
	uint32_t stride, const uint8_t* pData, vector<Resource::uptr>& uploaders)
{
	m_vertexBuffer = VertexBuffer::MakeUnique();
	N_RETURN(m_vertexBuffer->Create(pCommandList->GetDevice(), numVert, stride,
		ResourceFlag::NONE, MemoryType::DEFAULT, 1, nullptr, 1, nullptr,
		1, nullptr, MemoryFlag::NONE, L"MeshVB"), false);
	uploaders.emplace_back(Resource::MakeUnique());

	return m_vertexBuffer->Upload(pCommandList, uploaders.back().get(), pData, stride * numVert);
}

bool RendererEZ::createIB(CommandList* pCommandList, uint32_t numIndices,
	const uint32_t* pData, vector<Resource::uptr>& uploaders)
{
	m_numIndices = numIndices;

	const uint32_t byteWidth = sizeof(uint32_t) * numIndices;
	m_indexBuffer = IndexBuffer::MakeUnique();
	N_RETURN(m_indexBuffer->Create(pCommandList->GetDevice(), byteWidth, Format::R32_UINT, ResourceFlag::NONE,
		MemoryType::DEFAULT, 1, nullptr, 1, nullptr, 1, nullptr, MemoryFlag::NONE, L"MeshIB"), false);
	uploaders.emplace_back(Resource::MakeUnique());

	return m_indexBuffer->Upload(pCommandList, uploaders.back().get(), pData, byteWidth);
}

bool RendererEZ::createShaders()
{
	auto vsIndex = 0u;
	auto psIndex = 0u;
	auto csIndex = 0u;

	N_RETURN(m_shaderPool->CreateShader(Shader::Stage::VS, vsIndex, L"VSBasePass.cso"), false);
	m_shaders[VS_BASE_PASS] = m_shaderPool->GetShader(Shader::Stage::VS, vsIndex++);

	N_RETURN(m_shaderPool->CreateShader(Shader::Stage::VS, vsIndex, L"VSScreenQuad.cso"), false);
	m_shaders[VS_SCREEN_QUAD] = m_shaderPool->GetShader(Shader::Stage::VS, vsIndex++);

	N_RETURN(m_shaderPool->CreateShader(Shader::Stage::PS, psIndex, L"PSBasePassSH.cso"), false);
	m_shaders[PS_BASE_PASS] = m_shaderPool->GetShader(Shader::Stage::PS, psIndex++);

	N_RETURN(m_shaderPool->CreateShader(Shader::Stage::PS, psIndex, L"PSEnvironment.cso"), false);
	m_shaders[PS_ENVIRONMENT] = m_shaderPool->GetShader(Shader::Stage::PS, psIndex++);

	N_RETURN(m_shaderPool->CreateShader(Shader::Stage::PS, psIndex, L"PSPostprocess.cso"), false);
	m_shaders[PS_POSTPROCESS] = m_shaderPool->GetShader(Shader::Stage::PS, psIndex++);

	N_RETURN(m_shaderPool->CreateShader(Shader::Stage::CS, csIndex, L"CSTemporalAA.cso"), false);
	m_shaders[CS_TEMPORAL_AA] = m_shaderPool->GetShader(Shader::Stage::CS, csIndex++);

	return true;
}

void RendererEZ::createInputLayout()
{
	// Define the vertex input layout.
	const InputElement inputElements[] =
	{
		{ "POSITION",	0, Format::R32G32B32_FLOAT, 0, 0,								InputClassification::PER_VERTEX_DATA, 0 },
		{ "NORMAL",		0, Format::R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,	InputClassification::PER_VERTEX_DATA, 0 }
	};

	m_inputLayout.resize(static_cast<uint32_t>(size(inputElements)));
	memcpy(m_inputLayout.data(), inputElements, sizeof(inputElements));
}

void RendererEZ::render(EZ::CommandList* pCommandList, uint8_t frameIndex, bool needClear)
{
	// Set pipeline state
	const auto state = pCommandList->GetGraphicsPipelineState();
	state->IASetInputLayout(&m_inputLayout);
	state->SetShader(Shader::Stage::VS, m_shaders[VS_BASE_PASS]);
	state->SetShader(Shader::Stage::PS, m_shaders[PS_BASE_PASS]);
	state->OMSetNumRenderTargets(NUM_RENDER_TARGET);
	state->OMSetRTVFormat(RT_COLOR, Format::R16G16B16A16_FLOAT);
	state->OMSetRTVFormat(RT_VELOCITY, Format::R16G16_FLOAT);
	state->OMSetDSVFormat(Format::D24_UNORM_S8_UINT);
	pCommandList->DSSetState(Graphics::DEFAULT_LESS);

	// Set render targets
	EZ::ResourceView rtvs[] =
	{
		EZ::GetRTV(m_renderTargets[RT_COLOR].get()),
		EZ::GetRTV(m_renderTargets[RT_VELOCITY].get()),
	};
	auto dsv = EZ::GetDSV(m_depth.get());
	pCommandList->OMSetRenderTargets(static_cast<uint32_t>(size(rtvs)), rtvs, &dsv);

	// Clear render target
	const float clearColor[4] = { 0.2f, 0.2f, 0.7f, 0.0f };
	const float clearColorNull[4] = {};
	if (needClear) pCommandList->ClearRenderTargetView(rtvs[RT_COLOR], clearColor);
	pCommandList->ClearRenderTargetView(rtvs[RT_VELOCITY], clearColorNull);
	pCommandList->ClearDepthStencilView(dsv, ClearFlag::DEPTH, 1.0f);

	// Set viewport
	Viewport viewport(0.0f, 0.0f, static_cast<float>(m_viewport.x), static_cast<float>(m_viewport.y));
	RectRange scissorRect(0, 0, m_viewport.x, m_viewport.y);
	pCommandList->RSSetViewports(1, &viewport);
	pCommandList->RSSetScissorRects(1, &scissorRect);

	// Set IA
	pCommandList->IASetPrimitiveTopology(PrimitiveTopology::TRIANGLELIST);
	pCommandList->IASetVertexBuffers(0, 1, &m_vertexBuffer->GetVBV());
	pCommandList->IASetIndexBuffer(m_indexBuffer->GetIBV());

	// Set CBVs
	const auto cbvBasePass = EZ::GetCBV(m_cbBasePass.get(), frameIndex);
	pCommandList->SetGraphicsResources(Shader::Stage::VS, DescriptorType::CBV, 0, 1, &cbvBasePass);

	const auto cbvPerFrame = EZ::GetCBV(m_cbPerFrame.get(), frameIndex);
	pCommandList->SetGraphicsResources(Shader::Stage::PS, DescriptorType::CBV, 0, 1, & cbvPerFrame);

	// Set SRVs
	const EZ::ResourceView srvs[] =
	{
		EZ::GetSRV(m_radiance.get()),
		EZ::GetSRV(m_coeffSH.get())
	};
	pCommandList->SetGraphicsResources(Shader::Stage::PS, DescriptorType::SRV, 0,
		static_cast<uint32_t>(size(srvs)), srvs);

	// Set sampler
	const auto sampler = SamplerPreset::ANISOTROPIC_WRAP;
	pCommandList->SetGraphicsSamplerStates(0, 1, &sampler);

	pCommandList->DrawIndexed(m_numIndices, 1, 0, 0, 0);
}

void RendererEZ::environment(EZ::CommandList* pCommandList, uint8_t frameIndex)
{
	// Set pipeline state
	const auto state = pCommandList->GetGraphicsPipelineState();
	state->SetShader(Shader::Stage::VS, m_shaders[VS_SCREEN_QUAD]);
	state->SetShader(Shader::Stage::PS, m_shaders[PS_ENVIRONMENT]);
	state->OMSetNumRenderTargets(1);
	state->OMSetRTVFormat(RT_COLOR, Format::R16G16B16A16_FLOAT);
	state->OMSetRTVFormat(RT_VELOCITY, Format::UNKNOWN);
	state->OMSetDSVFormat(Format::D24_UNORM_S8_UINT);
	pCommandList->DSSetState(Graphics::DEPTH_READ_LESS_EQUAL);

	// Set render target
	const auto rtv = EZ::GetRTV(m_renderTargets[RT_COLOR].get());
	auto dsv = EZ::GetDSV(m_depth.get());
	pCommandList->OMSetRenderTargets(1, &rtv, &dsv);

	// Set CBV
	const auto cbv = EZ::GetCBV(m_cbPerFrame.get(), frameIndex);
	pCommandList->SetGraphicsResources(Shader::Stage::PS, DescriptorType::CBV, 0, 1, &cbv);

	// Set SRVs
	const EZ::ResourceView srvs[] =
	{
		EZ::GetSRV(m_radiance.get()),
		EZ::GetSRV(m_coeffSH.get())
	};
	pCommandList->SetGraphicsResources(Shader::Stage::PS, DescriptorType::SRV, 0,
		static_cast<uint32_t>(size(srvs)), srvs);

	// Set sampler
	const auto sampler = SamplerPreset::ANISOTROPIC_WRAP;
	pCommandList->SetGraphicsSamplerStates(0, 1, &sampler);

	pCommandList->IASetPrimitiveTopology(PrimitiveTopology::TRIANGLELIST);
	pCommandList->Draw(3, 1, 0, 0);
}

void RendererEZ::temporalAA(EZ::CommandList* pCommandList)
{
	// Set pipeline state
	const auto state = pCommandList->GetComputePipelineState();
	state->SetShader(m_shaders[CS_TEMPORAL_AA]);

	// Set UAVs
	const auto uav = EZ::GetUAV(m_outputViews[UAV_PP_TAA + m_frameParity].get());
	pCommandList->SetComputeResources(DescriptorType::UAV, 0, 1, &uav);

	const EZ::ResourceView srvs[] =
	{
		EZ::GetSRV(m_renderTargets[RT_COLOR].get()),
		EZ::GetSRV(m_outputViews[UAV_PP_TAA + !m_frameParity].get()),
		EZ::GetSRV(m_renderTargets[RT_VELOCITY].get())
	};
	pCommandList->SetComputeResources(DescriptorType::SRV, 0, static_cast<uint32_t>(size(srvs)), srvs);

	// Set sampler
	const auto sampler = SamplerPreset::LINEAR_CLAMP;
	pCommandList->SetComputeSamplerStates(0, 1, &sampler);

	pCommandList->Dispatch(DIV_UP(m_viewport.x, 8), DIV_UP(m_viewport.y, 8), 1);
}
