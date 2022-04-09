//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "SHSharedConsts.h"
#include "LightProbeEZ.h"
#define _INDEPENDENT_DDS_LOADER_
#include "Advanced/XUSGDDSLoader.h"
#undef _INDEPENDENT_DDS_LOADER_

using namespace std;
using namespace DirectX;
using namespace XUSG;

LightProbeEZ::LightProbeEZ()
{
	m_shaderPool = ShaderPool::MakeUnique();
}

LightProbeEZ::~LightProbeEZ()
{
}

bool LightProbeEZ::Init(CommandList* pCommandList, uint32_t width, uint32_t height,
	vector<Resource::uptr>& uploaders, const wstring pFileNames[], uint32_t numFiles)
{
	const auto pDevice = pCommandList->GetDevice();

	// Load input image
	auto texWidth = 1u, texHeight = 1u;
	m_sources.resize(numFiles);
	for (auto i = 0u; i < numFiles; ++i)
	{
		DDS::Loader textureLoader;
		DDS::AlphaMode alphaMode;

		uploaders.emplace_back(Resource::MakeUnique());
		N_RETURN(textureLoader.CreateTextureFromFile(pCommandList, pFileNames[i].c_str(),
			8192, false, m_sources[i], uploaders.back().get(), &alphaMode), false);

		texWidth = (max)(static_cast<uint32_t>(m_sources[i]->GetWidth()), texWidth);
		texHeight = (max)(m_sources[i]->GetHeight(), texHeight);
	}

	// Create resources
	const auto format = Format::R11G11B10_FLOAT;
	m_radiance = RenderTarget::MakeShared();
	m_radiance->Create(pDevice, texWidth, texHeight, format, 6,
		ResourceFlag::ALLOW_UNORDERED_ACCESS, 1, 1, nullptr, true,
		MemoryFlag::NONE, L"Radiance");

	m_numSHTexels = SH_TEX_SIZE * SH_TEX_SIZE * 6;
	const auto numGroups = DIV_UP(m_numSHTexels, SH_GROUP_SIZE);
	const auto numSumGroups = DIV_UP(numGroups, SH_GROUP_SIZE);
	const auto maxElements = SH_MAX_ORDER * SH_MAX_ORDER * numGroups;
	const auto maxSumElements = SH_MAX_ORDER * SH_MAX_ORDER * numSumGroups;
	m_coeffSH[0] = StructuredBuffer::MakeShared();
	m_coeffSH[0]->Create(pDevice, maxElements, sizeof(float[3]),
		ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT,
		1, nullptr, 1, nullptr, MemoryFlag::NONE, L"SHCoefficients0");
	m_coeffSH[1] = StructuredBuffer::MakeShared();
	m_coeffSH[1]->Create(pDevice, maxSumElements, sizeof(float[3]),
		ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT,
		1, nullptr, 1, nullptr, MemoryFlag::NONE, L"SHCoefficients1");
	m_weightSH[0] = StructuredBuffer::MakeUnique();
	m_weightSH[0]->Create(pDevice, numGroups, sizeof(float),
		ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT,
		1, nullptr, 1, nullptr, MemoryFlag::NONE, L"SHWeights0");
	m_weightSH[1] = StructuredBuffer::MakeUnique();
	m_weightSH[1]->Create(pDevice, numSumGroups, sizeof(float),
		ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT,
		1, nullptr, 1, nullptr, MemoryFlag::NONE, L"SHWeights1");

	// Create constant buffers
	m_cbPerFrame = ConstantBuffer::MakeUnique();
	N_RETURN(m_cbPerFrame->Create(pDevice, sizeof(float[FrameCount]), FrameCount,
		nullptr, MemoryType::UPLOAD, MemoryFlag::NONE, L"CBPerFrame"), false);

	{
		m_cbCubeMapSlices = ConstantBuffer::MakeUnique();
		N_RETURN(m_cbCubeMapSlices->Create(pDevice, sizeof(uint32_t[6]), 6, nullptr,
			MemoryType::UPLOAD, MemoryFlag::NONE, L"Slices"), false);
		
		for (uint8_t i = 0; i < 6; ++i)
			*reinterpret_cast<uint32_t*>(m_cbCubeMapSlices->Map(i)) = i;
	}

	{
		m_cbSHCubeMap = ConstantBuffer::MakeUnique();
		N_RETURN(m_cbSHCubeMap->Create(pDevice, sizeof(uint32_t[2]), 1, nullptr,
			MemoryType::UPLOAD, MemoryFlag::NONE, L"CBSHCubeMap"), false);

		reinterpret_cast<uint32_t*>(m_cbSHCubeMap->Map())[1] = SH_TEX_SIZE;
	}

	{
		auto loopCount = 0u;
		for (auto n = DIV_UP(m_numSHTexels, SH_GROUP_SIZE); n > 1; n = DIV_UP(n, SH_GROUP_SIZE)) ++loopCount;
		m_cbSHSums = ConstantBuffer::MakeUnique();
		N_RETURN(m_cbSHSums->Create(pDevice, sizeof(uint32_t[2]) * loopCount, loopCount,
			nullptr, MemoryType::UPLOAD, MemoryFlag::NONE, L"CBSHSums"), false);

		loopCount = 0;
		for (auto n = DIV_UP(m_numSHTexels, SH_GROUP_SIZE); n > 1; n = DIV_UP(n, SH_GROUP_SIZE))
			reinterpret_cast<uint32_t*>(m_cbSHSums->Map(loopCount++))[1] = n;
	}

	// Create shaders
	N_RETURN(createShaders(), false);

	return true;
}

void LightProbeEZ::UpdateFrame(double time, uint8_t frameIndex)
{
	// Update per-frame CB
	{
		static const auto period = 3.0;
		const auto numSources = static_cast<uint32_t>(m_sources.size());
		auto blend = static_cast<float>(time / period);
		m_inputProbeIdx = static_cast<uint32_t>(time / period);
		blend = numSources > 1 ? blend - m_inputProbeIdx : 0.0f;
		m_inputProbeIdx %= numSources;
		*reinterpret_cast<float*>(m_cbPerFrame->Map(frameIndex)) = blend;
	}
}

void LightProbeEZ::Process(EZ::CommandList* pCommandList, uint8_t frameIndex)
{
	// Set Descriptor pools
	const uint8_t order = 3;
	generateRadiance(pCommandList, frameIndex);
	shCubeMap(pCommandList, order);
	shSum(pCommandList, order, frameIndex);
	shNormalize(pCommandList, order);
}

Texture::sptr LightProbeEZ::GetRadiance() const
{
	return m_radiance;
}

StructuredBuffer::sptr LightProbeEZ::GetSH() const
{
	return m_coeffSH[m_shBufferParity];
}

bool LightProbeEZ::createShaders()
{
	auto csIndex = 0u;

	N_RETURN(m_shaderPool->CreateShader(Shader::Stage::CS, csIndex, L"CSGenRadiance.cso"), false);
	m_shaders[CS_RADIANCE_GEN] = m_shaderPool->GetShader(Shader::Stage::CS, csIndex++);

	N_RETURN(m_shaderPool->CreateShader(Shader::Stage::CS, csIndex, L"CSSHCubeMap.cso"), false);
	m_shaders[CS_SH_CUBE_MAP] = m_shaderPool->GetShader(Shader::Stage::CS, csIndex++);

	N_RETURN(m_shaderPool->CreateShader(Shader::Stage::CS, csIndex, L"CSSHSum.cso"), false);
	m_shaders[CS_SH_SUM] = m_shaderPool->GetShader(Shader::Stage::CS, csIndex++);

	N_RETURN(m_shaderPool->CreateShader(Shader::Stage::CS, csIndex, L"CSSHNormalize.cso"), false);
	m_shaders[CS_SH_NORMALIZE] = m_shaderPool->GetShader(Shader::Stage::CS, csIndex++);

	return true;
}

void LightProbeEZ::generateRadiance(EZ::CommandList* pCommandList, uint8_t frameIndex)
{
	// Set pipeline state
	pCommandList->SetComputeShader(m_shaders[CS_RADIANCE_GEN]);

	// Set UAV
	const auto uav = EZ::GetUAV(m_radiance.get());
	pCommandList->SetComputeResources(DescriptorType::UAV, 0, 1, &uav);

	// Set CBV
	const auto cbv = EZ::GetCBV(m_cbPerFrame.get(), frameIndex);
	pCommandList->SetComputeResources(DescriptorType::CBV, 0, 1, &cbv);

	// Set SRVs
	const auto numSources = static_cast<uint32_t>(m_sources.size());
	const auto nextProbeIdx = (m_inputProbeIdx + 1) % numSources;
	const EZ::ResourceView srvs[] =
	{
		EZ::GetSRV(m_sources[m_inputProbeIdx].get()),
		EZ::GetSRV(m_sources[nextProbeIdx].get())
	};
	pCommandList->SetComputeResources(DescriptorType::SRV, 0, static_cast<uint32_t>(size(srvs)), srvs);

	const auto sampler = SamplerPreset::LINEAR_WRAP;
	pCommandList->SetComputeSamplerStates(0, 1, &sampler);

	const auto w = static_cast<uint32_t>(m_radiance->GetWidth());
	const auto h = m_radiance->GetHeight();
	pCommandList->Dispatch(DIV_UP(w, 8), DIV_UP(h, 8), 6);
}

void LightProbeEZ::shCubeMap(EZ::CommandList* pCommandList, uint8_t order)
{
	// Set pipeline state
	pCommandList->SetComputeShader(m_shaders[CS_SH_CUBE_MAP]);

	// Set UAVs
	const EZ::ResourceView uavs[] =
	{
		EZ::GetUAV(m_coeffSH[0].get()),
		EZ::GetUAV(m_weightSH[0].get())

	};
	pCommandList->SetComputeResources(DescriptorType::UAV, 0, static_cast<uint32_t>(size(uavs)), uavs);

	// Set CBV
	assert(order <= SH_MAX_ORDER);
	*reinterpret_cast<uint32_t*>(m_cbSHCubeMap->Map()) = order;
	const auto cbv = EZ::GetCBV(m_cbSHCubeMap.get());
	pCommandList->SetComputeResources(DescriptorType::CBV, 0, 1, &cbv);

	// Set SRV
	const auto srv = EZ::GetSRV(m_radiance.get());
	pCommandList->SetComputeResources(DescriptorType::SRV, 0, 1, &srv);

	const auto sampler = SamplerPreset::LINEAR_WRAP;
	pCommandList->SetComputeSamplerStates(0, 1, &sampler);

	pCommandList->Dispatch(DIV_UP(m_numSHTexels, SH_GROUP_SIZE), 1, 1);
}

void LightProbeEZ::shSum(EZ::CommandList* pCommandList, uint8_t order, uint8_t frameIndex)
{
	assert(order <= SH_MAX_ORDER);
	m_shBufferParity = 0;

	// Set pipeline state
	pCommandList->SetComputeShader(m_shaders[CS_SH_SUM]);

	auto i = 0u;
	for (auto n = DIV_UP(m_numSHTexels, SH_GROUP_SIZE); n > 1; n = DIV_UP(n, SH_GROUP_SIZE))
	{
		const auto& src = m_shBufferParity;
		const uint8_t dst = !m_shBufferParity;

		// Set UAVs
		const EZ::ResourceView uavs[] =
		{
			EZ::GetUAV(m_coeffSH[dst].get()),
			EZ::GetUAV(m_weightSH[dst].get())

		};
		pCommandList->SetComputeResources(DescriptorType::UAV, 0, static_cast<uint32_t>(size(uavs)), uavs);

		// Set SRVs
		const EZ::ResourceView srvs[] =
		{
			EZ::GetSRV(m_coeffSH[src].get()),
			EZ::GetSRV(m_weightSH[src].get())
		};
		pCommandList->SetComputeResources(DescriptorType::SRV, 0, static_cast<uint32_t>(size(srvs)), srvs);

		// Set CBV
		*reinterpret_cast<uint32_t*>(m_cbSHSums->Map(i++)) = order;
		const auto cbv = EZ::GetCBV(m_cbSHSums.get());
		pCommandList->SetComputeResources(DescriptorType::CBV, 0, 1, &cbv);

		pCommandList->Dispatch(DIV_UP(n, SH_GROUP_SIZE), order * order, 1);
		m_shBufferParity = !m_shBufferParity;
	}
}

void LightProbeEZ::shNormalize(EZ::CommandList* pCommandList, uint8_t order)
{
	assert(order <= SH_MAX_ORDER);
	const auto& src = m_shBufferParity;
	const uint8_t dst = !m_shBufferParity;

	// Set pipeline state
	pCommandList->SetComputeShader(m_shaders[CS_SH_NORMALIZE]);

	// Set UAV
	const auto uav = EZ::GetUAV(m_coeffSH[dst].get());
	pCommandList->SetComputeResources(DescriptorType::UAV, 0, 1, &uav);

	// Set SRVs
	const EZ::ResourceView srvs[] =
	{
		EZ::GetSRV(m_coeffSH[src].get()),
		EZ::GetSRV(m_weightSH[src].get())
	};
	pCommandList->SetComputeResources(DescriptorType::SRV, 0, static_cast<uint32_t>(size(srvs)), srvs);

	const auto numElements = order * order;
	pCommandList->Dispatch(DIV_UP(numElements, SH_GROUP_SIZE), 1, 1);
	m_shBufferParity = !m_shBufferParity;
}
