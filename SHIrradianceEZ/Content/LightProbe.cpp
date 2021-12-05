//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "LightProbe.h"
#define _INDEPENDENT_DDS_LOADER_
#include "Advanced/XUSGDDSLoader.h"
#undef _INDEPENDENT_DDS_LOADER_

#define SH_MAX_ORDER	6
#define SH_TEX_SIZE		256
#define SH_GROUP_SIZE	32

using namespace std;
using namespace DirectX;
using namespace XUSG;

LightProbe::LightProbe(const Device::sptr& device) :
	m_device(device)
{
	m_shaderPool = ShaderPool::MakeUnique();
	m_graphicsPipelineCache = Graphics::PipelineCache::MakeUnique(device.get());
	m_computePipelineCache = Compute::PipelineCache::MakeUnique(device.get());
	m_pipelineLayoutCache = PipelineLayoutCache::MakeUnique(device.get());
}

LightProbe::~LightProbe()
{
}

bool LightProbe::Init(CommandList* pCommandList, uint32_t width, uint32_t height,
	const DescriptorTableCache::sptr& descriptorTableCache, vector<Resource::uptr>& uploaders,
	const wstring pFileNames[], uint32_t numFiles, bool typedUAV)
{
	m_descriptorTableCache = descriptorTableCache;

	// Load input image
	auto texWidth = 1u, texHeight = 1u;
	m_sources.resize(numFiles);
	for (auto i = 0u; i < numFiles; ++i)
	{
		DDS::Loader textureLoader;
		DDS::AlphaMode alphaMode;

		uploaders.emplace_back(Resource::MakeUnique());
		N_RETURN(textureLoader.CreateTextureFromFile(m_device.get(), pCommandList, pFileNames[i].c_str(),
			8192, false, m_sources[i], uploaders.back().get(), &alphaMode), false);

		texWidth = (max)(static_cast<uint32_t>(m_sources[i]->GetWidth()), texWidth);
		texHeight = (max)(dynamic_pointer_cast<Texture2D, ShaderResource>(m_sources[i])->GetHeight(), texHeight);
	}

	// Create resources and pipelines
	const auto format = Format::R11G11B10_FLOAT;
	m_radiance = RenderTarget::MakeUnique();
	m_radiance->Create(m_device.get(), texWidth, texHeight, format, 6,
		ResourceFlag::ALLOW_UNORDERED_ACCESS, 1, 1, nullptr, true,
		MemoryFlag::NONE, L"Radiance");

	m_numSHTexels = SH_TEX_SIZE * SH_TEX_SIZE * 6;
	const auto numGroups = DIV_UP(m_numSHTexels, SH_GROUP_SIZE);
	const auto numSumGroups = DIV_UP(numGroups, SH_GROUP_SIZE);
	const auto maxElements = SH_MAX_ORDER * SH_MAX_ORDER * numGroups;
	const auto maxSumElements = SH_MAX_ORDER * SH_MAX_ORDER * numSumGroups;
	m_coeffSH[0] = StructuredBuffer::MakeShared();
	m_coeffSH[0]->Create(m_device.get(), maxElements, sizeof(float[3]),
		ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT,
		1, nullptr, 1, nullptr, MemoryFlag::NONE, L"SHCoefficients0");
	m_coeffSH[1] = StructuredBuffer::MakeShared();
	m_coeffSH[1]->Create(m_device.get(), maxSumElements, sizeof(float[3]),
		ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT,
		1, nullptr, 1, nullptr, MemoryFlag::NONE, L"SHCoefficients1");
	m_weightSH[0] = StructuredBuffer::MakeUnique();
	m_weightSH[0]->Create(m_device.get(), numGroups, sizeof(float),
		ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT,
		1, nullptr, 1, nullptr, MemoryFlag::NONE, L"SHWeights0");
	m_weightSH[1] = StructuredBuffer::MakeUnique();
	m_weightSH[1]->Create(m_device.get(), numSumGroups, sizeof(float),
		ResourceFlag::ALLOW_UNORDERED_ACCESS, MemoryType::DEFAULT,
		1, nullptr, 1, nullptr, MemoryFlag::NONE, L"SHWeights1");

	// Create constant buffers
	m_cbPerFrame = ConstantBuffer::MakeUnique();
	N_RETURN(m_cbPerFrame->Create(m_device.get(), sizeof(float[FrameCount]), FrameCount,
		nullptr, MemoryType::UPLOAD, MemoryFlag::NONE, L"CBPerFrame"), false);

	N_RETURN(createPipelineLayouts(), false);
	N_RETURN(createPipelines(format, typedUAV), false);
	N_RETURN(createDescriptorTables(), false);

	return true;
}

void LightProbe::UpdateFrame(double time, uint8_t frameIndex)
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

void LightProbe::Process(const CommandList* pCommandList, uint8_t frameIndex)
{
	// Set Descriptor pools
	const DescriptorPool descriptorPools[] =
	{
		m_descriptorTableCache->GetDescriptorPool(CBV_SRV_UAV_POOL),
		m_descriptorTableCache->GetDescriptorPool(SAMPLER_POOL)
	};
	pCommandList->SetDescriptorPools(static_cast<uint32_t>(size(descriptorPools)), descriptorPools);

	const uint8_t order = 3;
	generateRadianceCompute(pCommandList, frameIndex);
	shCubeMap(pCommandList, order);
	shSum(pCommandList, order);
	shNormalize(pCommandList, order);
}

ShaderResource* LightProbe::GetRadiance() const
{
	return m_radiance.get();
}

StructuredBuffer::sptr LightProbe::GetSH() const
{
	return m_coeffSH[m_shBufferParity];
}

bool LightProbe::createPipelineLayouts()
{
	// Generate Radiance graphics
	{
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique();
		utilPipelineLayout->SetRange(0, DescriptorType::SAMPLER, 1, 0);
		utilPipelineLayout->SetRootCBV(1, 0, 0, Shader::PS);
		utilPipelineLayout->SetConstants(2, SizeOfInUint32(uint32_t), 1, 0, Shader::PS);
		utilPipelineLayout->SetRange(3, DescriptorType::SRV, 2, 0);
		utilPipelineLayout->SetShaderStage(0, Shader::PS);
		utilPipelineLayout->SetShaderStage(3, Shader::PS);
		X_RETURN(m_pipelineLayouts[GEN_RADIANCE_GRAPHICS], utilPipelineLayout->GetPipelineLayout(
			m_pipelineLayoutCache.get(), PipelineLayoutFlag::NONE, L"RadianceGenerationLayout"), false);
	}

	// Generate Radiance compute
	{
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique();
		utilPipelineLayout->SetRange(0, DescriptorType::SAMPLER, 1, 0);
		utilPipelineLayout->SetRootCBV(1, 0);
		utilPipelineLayout->SetRange(2, DescriptorType::UAV, 1, 0, 0, DescriptorFlag::DATA_STATIC_WHILE_SET_AT_EXECUTE);
		utilPipelineLayout->SetRange(3, DescriptorType::SRV, 2, 0);
		X_RETURN(m_pipelineLayouts[GEN_RADIANCE_COMPUTE], utilPipelineLayout->GetPipelineLayout(
			m_pipelineLayoutCache.get(), PipelineLayoutFlag::NONE, L"RadianceGenerationLayout"), false);
	}

	// SH cube map transform
	{
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique();
		utilPipelineLayout->SetRange(0, DescriptorType::SAMPLER, 1, 0);
		utilPipelineLayout->SetRootUAV(1, 0);
		utilPipelineLayout->SetRootUAV(2, 1);
		utilPipelineLayout->SetRange(3, DescriptorType::SRV, 1, 0);
		utilPipelineLayout->SetConstants(4, SizeOfInUint32(uint32_t[2]), 0);
		X_RETURN(m_pipelineLayouts[SH_CUBE_MAP], utilPipelineLayout->GetPipelineLayout(
			m_pipelineLayoutCache.get(), PipelineLayoutFlag::NONE, L"SHCubeMapLayout"), false);
	}

	// SH sum
	{
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique();
		utilPipelineLayout->SetRootUAV(0, 0);
		utilPipelineLayout->SetRootUAV(1, 1);
		utilPipelineLayout->SetRootSRV(2, 0);
		utilPipelineLayout->SetRootSRV(3, 1);
		utilPipelineLayout->SetConstants(4, SizeOfInUint32(uint32_t[2]), 0);
		X_RETURN(m_pipelineLayouts[SH_SUM], utilPipelineLayout->GetPipelineLayout(
			m_pipelineLayoutCache.get(), PipelineLayoutFlag::NONE, L"SHSumLayout"), false);
	}

	// SH normalization
	{
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique();
		utilPipelineLayout->SetRootUAV(0, 0);
		utilPipelineLayout->SetRootSRV(1, 0);
		utilPipelineLayout->SetRootSRV(2, 1);
		X_RETURN(m_pipelineLayouts[SH_NORMALIZE], utilPipelineLayout->GetPipelineLayout(
			m_pipelineLayoutCache.get(), PipelineLayoutFlag::NONE, L"SHNormalizeLayout"), false);
	}

	return true;
}

bool LightProbe::createPipelines(Format rtFormat, bool typedUAV)
{
	auto vsIndex = 0u;
	auto psIndex = 0u;
	auto csIndex = 0u;

	// Generate Radiance graphics
	N_RETURN(m_shaderPool->CreateShader(Shader::Stage::VS, vsIndex, L"VSScreenQuad.cso"), false);
	{
		N_RETURN(m_shaderPool->CreateShader(Shader::Stage::PS, psIndex, L"PSGenRadiance.cso"), false);

		const auto state = Graphics::State::MakeUnique();
		state->SetPipelineLayout(m_pipelineLayouts[GEN_RADIANCE_GRAPHICS]);
		state->SetShader(Shader::Stage::VS, m_shaderPool->GetShader(Shader::Stage::VS, vsIndex));
		state->SetShader(Shader::Stage::PS, m_shaderPool->GetShader(Shader::Stage::PS, psIndex++));
		state->DSSetState(Graphics::DEPTH_STENCIL_NONE, m_graphicsPipelineCache.get());
		state->IASetPrimitiveTopologyType(PrimitiveTopologyType::TRIANGLE);
		state->OMSetNumRenderTargets(1);
		state->OMSetRTVFormat(0, rtFormat);
		X_RETURN(m_pipelines[GEN_RADIANCE_GRAPHICS], state->GetPipeline(m_graphicsPipelineCache.get(), L"RadianceGeneration_graphics"), false);
	}

	// Generate Radiance compute
	{
		N_RETURN(m_shaderPool->CreateShader(Shader::Stage::CS, csIndex, L"CSGenRadiance.cso"), false);

		const auto state = Compute::State::MakeUnique();
		state->SetPipelineLayout(m_pipelineLayouts[GEN_RADIANCE_COMPUTE]);
		state->SetShader(m_shaderPool->GetShader(Shader::Stage::CS, csIndex++));
		X_RETURN(m_pipelines[GEN_RADIANCE_COMPUTE], state->GetPipeline(m_computePipelineCache.get(), L"RadianceGeneration_compute"), false);
	}

	// SH cube map transform
	{
		N_RETURN(m_shaderPool->CreateShader(Shader::Stage::CS, csIndex, L"CSSHCubeMap.cso"), false);

		const auto state = Compute::State::MakeUnique();
		state->SetPipelineLayout(m_pipelineLayouts[SH_CUBE_MAP]);
		state->SetShader(m_shaderPool->GetShader(Shader::Stage::CS, csIndex++));
		X_RETURN(m_pipelines[SH_CUBE_MAP], state->GetPipeline(m_computePipelineCache.get(), L"SHCubeMap"), false);
	}

	// SH sum
	{
		N_RETURN(m_shaderPool->CreateShader(Shader::Stage::CS, csIndex, L"CSSHSum.cso"), false);

		const auto state = Compute::State::MakeUnique();
		state->SetPipelineLayout(m_pipelineLayouts[SH_SUM]);
		state->SetShader(m_shaderPool->GetShader(Shader::Stage::CS, csIndex++));
		X_RETURN(m_pipelines[SH_SUM], state->GetPipeline(m_computePipelineCache.get(), L"SHSum"), false);
	}

	// SH normalization
	{
		N_RETURN(m_shaderPool->CreateShader(Shader::Stage::CS, csIndex, L"CSSHNormalize.cso"), false);

		const auto state = Compute::State::MakeUnique();
		state->SetPipelineLayout(m_pipelineLayouts[SH_NORMALIZE]);
		state->SetShader(m_shaderPool->GetShader(Shader::Stage::CS, csIndex));
		X_RETURN(m_pipelines[SH_NORMALIZE], state->GetPipeline(m_computePipelineCache.get(), L"SHNormalize"), false);
	}

	return true;
}

bool LightProbe::createDescriptorTables()
{
	// Get UAV table for radiance generation
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique();
		descriptorTable->SetDescriptors(0, 1, &m_radiance->GetUAV());
		X_RETURN(m_uavTable, descriptorTable->GetCbvSrvUavTable(m_descriptorTableCache.get()), false);
	}

	// Get SRV tables for radiance generation
	const auto numSources = static_cast<uint32_t>(m_sources.size());
	m_srvTables[SRV_TABLE_INPUT].resize(m_sources.size());
	for (auto i = 0u; i + 1 < numSources; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique();
		descriptorTable->SetDescriptors(0, 1, &m_sources[i]->GetSRV());
		X_RETURN(m_srvTables[SRV_TABLE_INPUT][i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableCache.get()), false);
	}
	{
		const auto i = numSources - 1;
		const Descriptor descriptors[] =
		{
			m_sources[i]->GetSRV(),
			m_sources[0]->GetSRV()
		};
		const auto descriptorTable = Util::DescriptorTable::MakeUnique();
		descriptorTable->SetDescriptors(0, static_cast<uint32_t>(size(descriptors)), descriptors);
		X_RETURN(m_srvTables[SRV_TABLE_INPUT][i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableCache.get()), false);
	}

	// Create radiance SRV
	m_srvTables[SRV_TABLE_RADIANCE].resize(1);
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique();
		descriptorTable->SetDescriptors(0, 1, &m_radiance->GetSRV());
		X_RETURN(m_srvTables[SRV_TABLE_RADIANCE][0], descriptorTable->GetCbvSrvUavTable(m_descriptorTableCache.get()), false);
	}

	// Create the sampler table
	const auto descriptorTable = Util::DescriptorTable::MakeUnique();
	const auto sampler = LINEAR_WRAP;
	descriptorTable->SetSamplers(0, 1, &sampler, m_descriptorTableCache.get());
	X_RETURN(m_samplerTable, descriptorTable->GetSamplerTable(m_descriptorTableCache.get()), false);

	return true;
}

void LightProbe::generateRadianceGraphics(const CommandList* pCommandList, uint8_t frameIndex)
{
	ResourceBarrier barrier;
	const auto numBarriers = m_radiance->SetBarrier(&barrier, ResourceState::RENDER_TARGET);
	pCommandList->Barrier(numBarriers, &barrier);

	pCommandList->SetGraphicsPipelineLayout(m_pipelineLayouts[GEN_RADIANCE_GRAPHICS]);
	pCommandList->SetGraphicsRootConstantBufferView(1, m_cbPerFrame.get(), m_cbPerFrame->GetCBVOffset(frameIndex));

	m_radiance->Blit(pCommandList, m_srvTables[SRV_TABLE_INPUT][m_inputProbeIdx], 3, 0, 0, 0,
		m_samplerTable, 0, m_pipelines[GEN_RADIANCE_GRAPHICS]);
}

void LightProbe::generateRadianceCompute(const CommandList* pCommandList, uint8_t frameIndex)
{
	ResourceBarrier barrier;
	const auto numBarriers = m_radiance->SetBarrier(&barrier, ResourceState::UNORDERED_ACCESS);
	pCommandList->Barrier(numBarriers, &barrier);

	pCommandList->SetComputePipelineLayout(m_pipelineLayouts[GEN_RADIANCE_COMPUTE]);
	pCommandList->SetComputeRootConstantBufferView(1, m_cbPerFrame.get(), m_cbPerFrame->GetCBVOffset(frameIndex));

	m_radiance->AsTexture()->Blit(pCommandList, 8, 8, 1, m_uavTable, 2, 0,
		m_srvTables[SRV_TABLE_INPUT][m_inputProbeIdx], 3, m_samplerTable, 0, m_pipelines[GEN_RADIANCE_COMPUTE]);
}

void LightProbe::shCubeMap(const CommandList* pCommandList, uint8_t order)
{
	assert(order <= SH_MAX_ORDER);
	ResourceBarrier barrier;
	m_coeffSH[0]->SetBarrier(&barrier, ResourceState::UNORDERED_ACCESS);	// Promotion
	m_weightSH[0]->SetBarrier(&barrier, ResourceState::UNORDERED_ACCESS);	// Promotion
	const auto numBarriers = m_radiance->SetBarrier(&barrier,
		ResourceState::NON_PIXEL_SHADER_RESOURCE | ResourceState::PIXEL_SHADER_RESOURCE);
	pCommandList->Barrier(numBarriers, &barrier);

	pCommandList->SetComputePipelineLayout(m_pipelineLayouts[SH_CUBE_MAP]);
	pCommandList->SetComputeDescriptorTable(0, m_samplerTable);
	pCommandList->SetComputeRootUnorderedAccessView(1, m_coeffSH[0].get());
	pCommandList->SetComputeRootUnorderedAccessView(2, m_weightSH[0].get());
	pCommandList->SetComputeDescriptorTable(3, m_srvTables[SRV_TABLE_RADIANCE][0]);
	pCommandList->SetCompute32BitConstant(4, order);
	pCommandList->SetCompute32BitConstant(4, SH_TEX_SIZE, SizeOfInUint32(order));
	pCommandList->SetPipelineState(m_pipelines[SH_CUBE_MAP]);

	pCommandList->Dispatch(DIV_UP(m_numSHTexels, SH_GROUP_SIZE), 1, 1);
}

void LightProbe::shSum(const CommandList* pCommandList, uint8_t order)
{
	assert(order <= SH_MAX_ORDER);
	ResourceBarrier barriers[4];
	m_shBufferParity = 0;

	pCommandList->SetComputePipelineLayout(m_pipelineLayouts[SH_SUM]);
	pCommandList->SetCompute32BitConstant(4, order);
	pCommandList->SetPipelineState(m_pipelines[SH_SUM]);

	// Promotions
	m_coeffSH[1]->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS);
	m_weightSH[1]->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS);

	for (auto n = DIV_UP(m_numSHTexels, SH_GROUP_SIZE); n > 1; n = DIV_UP(n, SH_GROUP_SIZE))
	{
		const auto& src = m_shBufferParity;
		const uint8_t dst = !m_shBufferParity;
		auto numBarriers = m_coeffSH[dst]->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS);
		numBarriers = m_weightSH[dst]->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS, numBarriers);
		numBarriers = m_coeffSH[src]->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE, numBarriers);
		numBarriers = m_weightSH[src]->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE, numBarriers);
		pCommandList->Barrier(numBarriers, barriers);

		pCommandList->SetComputeRootUnorderedAccessView(0, m_coeffSH[dst].get());
		pCommandList->SetComputeRootUnorderedAccessView(1, m_weightSH[dst].get());
		pCommandList->SetComputeRootShaderResourceView(2, m_coeffSH[src].get());
		pCommandList->SetComputeRootShaderResourceView(3, m_weightSH[src].get());
		pCommandList->SetCompute32BitConstant(4, n, SizeOfInUint32(order));

		pCommandList->Dispatch(DIV_UP(n, SH_GROUP_SIZE), order * order, 1);
		m_shBufferParity = !m_shBufferParity;
	}
}

void LightProbe::shNormalize(const CommandList* pCommandList, uint8_t order)
{
	assert(order <= SH_MAX_ORDER);
	ResourceBarrier barriers[3];
	const auto& src = m_shBufferParity;
	const uint8_t dst = !m_shBufferParity;
	auto numBarriers = m_coeffSH[dst]->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS);
	numBarriers = m_coeffSH[src]->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE, numBarriers);
	numBarriers = m_weightSH[src]->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE, numBarriers);
	pCommandList->Barrier(numBarriers, barriers);

	pCommandList->SetComputePipelineLayout(m_pipelineLayouts[SH_NORMALIZE]);
	pCommandList->SetComputeRootUnorderedAccessView(0, m_coeffSH[dst].get());
	pCommandList->SetComputeRootShaderResourceView(1, m_coeffSH[src].get());
	pCommandList->SetComputeRootShaderResourceView(2, m_weightSH[src].get());
	pCommandList->SetPipelineState(m_pipelines[SH_NORMALIZE]);

	const auto numElements = order * order;
	pCommandList->Dispatch(DIV_UP(numElements, SH_GROUP_SIZE), 1, 1);
	m_shBufferParity = !m_shBufferParity;
}
