#pragma once

#include "mtl_api.h"
#include "device.h"
#include "Emu/RSX/Common/simple_array.hpp"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "Emu/RSX/gcm_enums.h"
#include "Utilities/geometry.h"

#include <cmath>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mtl
{
	// Snap a sampler LOD parameter (min/max LOD, bias) to 0.5 steps. RSX registers carry
	// continuously-varying floats (GoW Ascension's min_lod jitters ~0.03/frame) and the pool
	// dedupes by exact float equality, so raw values mint a new MTLSamplerState — and a new
	// driver GPU resource ID — per distinct value. Past ~1024 simultaneously-live
	// argument-buffer samplers the driver segfaults resolving the newest ID in
	// setSamplerState (the RSX 0x448/0x449 class: killer ID 0x400/0x401, max-created ==
	// killer). Snapping makes jitter dedupe to one shared state; half a mip step on clamp
	// and bias bounds is visually invisible (the Nearest path already rounded bias this way).
	inline float quantize_sampler_lod(float lod)
	{
		return std::floor(lod * 2.f + 0.5f) * 0.5f;
	}

	// Metal only has three fixed border colors (no custom border color). Exact RSX border colors that do not map to
	// one of these are approximated (see get_closest_border_color).
	struct border_color_t
	{
		MTL::SamplerBorderColor value = MTL::SamplerBorderColorTransparentBlack;
		color4f color_value{};
		bool exact = true; // false if the requested color had to be approximated

		border_color_t() = default;
		explicit border_color_t(const color4f& color);
		explicit border_color_t(MTL::SamplerBorderColor color);

		bool operator == (const border_color_t& that) const { return value == that.value; }
	};

	struct sampler_create_info
	{
		MTL::SamplerAddressMode clamp_u = MTL::SamplerAddressModeClampToEdge;
		MTL::SamplerAddressMode clamp_v = MTL::SamplerAddressModeClampToEdge;
		MTL::SamplerAddressMode clamp_w = MTL::SamplerAddressModeClampToEdge;
		bool unnormalized_coordinates = false;
		float mip_lod_bias = 0.f;       // Applied only on Apple10+ (sampler LOD bias support); shaders otherwise
		float max_anisotropy = 1.f;
		float min_lod = 0.f;
		float max_lod = 1000.f;
		MTL::SamplerMinMagFilter min_filter = MTL::SamplerMinMagFilterLinear;
		MTL::SamplerMinMagFilter mag_filter = MTL::SamplerMinMagFilterLinear;
		MTL::SamplerMipFilter mip_filter = MTL::SamplerMipFilterNotMipmapped;
		border_color_t border_color{};
		bool depth_compare = false;
		MTL::CompareFunction compare_function = MTL::CompareFunctionNever;

		u64 hash() const;
		bool operator == (const sampler_create_info& that) const;
	};

	// Sampler state created with supportArgumentBuffers so it can be bound through MTL4ArgumentTable (gpuResourceID).
	struct sampler
	{
		MTL::SamplerState* value = nullptr;
		sampler_create_info info{};

		sampler(const render_device& dev, const sampler_create_info& create_info);
		~sampler();

		sampler(const sampler&) = delete;
		sampler& operator=(const sampler&) = delete;

		MTL::ResourceID resource_id() const { return m_resource_id; }

	private:
		MTL::ResourceID m_resource_id{}; // Fixed for the sampler's lifetime; cached because draws bind samplers by ID
	};

	struct cached_sampler_object_t : public mtl::sampler, public rsx::ref_counted
	{
		u64 key = 0;
		using mtl::sampler::sampler;
	};

	class sampler_pool_t
	{
		std::unordered_map<u64, std::vector<std::unique_ptr<cached_sampler_object_t>>> m_pool;

	public:
		void clear();
		cached_sampler_object_t* find(const sampler_create_info& info) const;
		cached_sampler_object_t* emplace(std::unique_ptr<cached_sampler_object_t>& object);
		std::vector<std::unique_ptr<cached_sampler_object_t>> collect(std::function<bool(const cached_sampler_object_t&)> predicate);
		usz size() const;
	};

	// Helpers mapping RSX/GCM sampler state
	MTL::SamplerAddressMode get_sampler_address_mode(rsx::texture_wrap_mode wrap, bool* uses_border = nullptr);
	MTL::CompareFunction get_compare_function(rsx::comparison_function op, bool reverse_direction = false);
}
