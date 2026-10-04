#include "stdafx.h"
#include <bit>
#include <algorithm>
#include "image.h"
#include "buffer_object.h"
#include "garbage_collector.h"
#include "Emu/system_config.h"

#include <cstdlib>
#include <mutex>
#include <vector>

namespace mtl
{
	namespace
	{
		// MTL4 texture view pools backing transient views (image_view::make_transient). Setting a view at a pool index
		// allocates nothing; the index is reused once the view object is destroyed, which its owner defers through the
		// GC until the GPU has executed the commands that use it.
		class transient_view_pool
		{
			static constexpr u32 views_per_pool = 256;

			std::mutex m_mutex;
			std::vector<MTL::TextureViewPool*> m_pools;
			std::vector<u32> m_free_entries;                   // pool index * views_per_pool + view index
			MTL::TextureViewDescriptor* m_descriptor = nullptr; // Reused for every view (under m_mutex)
			bool m_unavailable = false;
			u64 m_views_created = 0;

			bool grow()
			{
				autorelease_scope pool;

				auto desc = ref(MTL::ResourceViewPoolDescriptor::alloc()->init());
				desc->setResourceViewCount(views_per_pool);
				desc->setLabel(ns_str("RSX transient texture views"));

				NS::Error* error = nullptr;
				MTL::TextureViewPool* view_pool = g_render_device->handle()->newTextureViewPool(desc.get(), &error);
				if (!view_pool)
				{
					rsx_log.error("Metal: failed to create a texture view pool (%s). Transient views are created as texture view objects.", to_string(error));
					return false;
				}

				const u32 base = ::size32(m_pools) * views_per_pool;
				m_pools.push_back(view_pool);

				for (u32 i = views_per_pool; i > 0; --i)
				{
					m_free_entries.push_back(base + i - 1);
				}

				return true;
			}

		public:
			// Sets a view of `texture` at a free entry. False if pools are unavailable (API failure, logged once).
			bool create_view(const MTL::Texture* texture, MTL::PixelFormat format, const image_view_info& info, u32& entry, MTL::ResourceID& id)
			{
				std::lock_guard lock(m_mutex);

				if (m_unavailable)
				{
					return false;
				}

				if (!m_descriptor)
				{
					m_descriptor = MTL::TextureViewDescriptor::alloc()->init();
				}

				if (m_free_entries.empty() && !grow())
				{
					m_unavailable = true;
					return false;
				}

				entry = m_free_entries.back();

				m_descriptor->setPixelFormat(format);
				m_descriptor->setTextureType(info.type);
				m_descriptor->setLevelRange(NS::Range::Make(info.base_level, info.level_count));
				m_descriptor->setSliceRange(NS::Range::Make(info.base_layer, info.layer_count));
				m_descriptor->setSwizzle(info.swizzle);

				id = m_pools[entry / views_per_pool]->setTextureView(texture, m_descriptor, entry % views_per_pool);
				if (!id._impl)
				{
					rsx_log.error("Metal: a texture view pool returned no resource ID. Transient views are created as texture view objects.");
					m_unavailable = true;
					return false;
				}

				m_free_entries.pop_back();
				m_views_created++;
				return true;
			}

			void release_view(u32 entry)
			{
				std::lock_guard lock(m_mutex);
				m_free_entries.push_back(entry);
			}

			u64 take_views_created()
			{
				std::lock_guard lock(m_mutex);
				return std::exchange(m_views_created, 0);
			}

			void destroy()
			{
				std::lock_guard lock(m_mutex);

				if (m_free_entries.size() != m_pools.size() * views_per_pool)
				{
					rsx_log.error("Metal: %u transient texture view(s) still alive at renderer teardown",
						static_cast<u32>(m_pools.size() * views_per_pool - m_free_entries.size()));
				}

				for (MTL::TextureViewPool* view_pool : m_pools)
				{
					view_pool->release();
				}

				if (m_descriptor)
				{
					m_descriptor->release();
					m_descriptor = nullptr;
				}

				m_pools.clear();
				m_free_entries.clear();
				m_unavailable = false;
			}
		};

		transient_view_pool g_transient_views;
	}

	void destroy_transient_view_pools()
	{
		g_transient_views.destroy();
	}

	u64 get_transient_views_and_reset()
	{
		return g_transient_views.take_views_created();
	}

	bool debug_labels_enabled()
	{
		static const bool s_env_enabled = []()
		{
			for (const char* name : { "RPCS3_METAL_DEBUG_LABELS", "MTL_CAPTURE_ENABLED", "MTL_DEBUG_LAYER" })
			{
				if (const char* value = ::getenv(name); value && value[0] && value[0] != '0')
				{
					return true;
				}
			}

			return false;
		}();

		return s_env_enabled || g_cfg.video.debug_output;
	}

	bool is_depth_format(MTL::PixelFormat format)
	{
		switch (format)
		{
		case MTL::PixelFormatDepth16Unorm:
		case MTL::PixelFormatDepth32Float:
		case MTL::PixelFormatDepth24Unorm_Stencil8:
		case MTL::PixelFormatDepth32Float_Stencil8:
			return true;
		default:
			return false;
		}
	}

	bool is_stencil_format(MTL::PixelFormat format)
	{
		switch (format)
		{
		case MTL::PixelFormatStencil8:
		case MTL::PixelFormatDepth24Unorm_Stencil8:
		case MTL::PixelFormatDepth32Float_Stencil8:
		case MTL::PixelFormatX32_Stencil8:
		case MTL::PixelFormatX24_Stencil8:
			return true;
		default:
			return false;
		}
	}

	u32 get_format_aspect(MTL::PixelFormat format)
	{
		u32 result = 0;
		if (is_depth_format(format)) result |= aspect_depth;
		if (is_stencil_format(format)) result |= aspect_stencil;
		return result ? result : static_cast<u32>(aspect_color);
	}

	// Linear <-> sRGB twins. Metal does not require MTLTextureUsagePixelFormatView for views that only toggle sRGB.
	static MTL::PixelFormat get_linear_format(MTL::PixelFormat format)
	{
		switch (format)
		{
		case MTL::PixelFormatR8Unorm_sRGB: return MTL::PixelFormatR8Unorm;
		case MTL::PixelFormatRG8Unorm_sRGB: return MTL::PixelFormatRG8Unorm;
		case MTL::PixelFormatRGBA8Unorm_sRGB: return MTL::PixelFormatRGBA8Unorm;
		case MTL::PixelFormatBGRA8Unorm_sRGB: return MTL::PixelFormatBGRA8Unorm;
		case MTL::PixelFormatBGR10_XR_sRGB: return MTL::PixelFormatBGR10_XR;
		case MTL::PixelFormatBGRA10_XR_sRGB: return MTL::PixelFormatBGRA10_XR;
		case MTL::PixelFormatBC1_RGBA_sRGB: return MTL::PixelFormatBC1_RGBA;
		case MTL::PixelFormatBC2_RGBA_sRGB: return MTL::PixelFormatBC2_RGBA;
		case MTL::PixelFormatBC3_RGBA_sRGB: return MTL::PixelFormatBC3_RGBA;
		case MTL::PixelFormatBC7_RGBAUnorm_sRGB: return MTL::PixelFormatBC7_RGBAUnorm;
		default: return format;
		}
	}

	// True if a view of `view_format` on a texture of `image_format` changes the component layout, i.e. needs the
	// texture to be created with MTLTextureUsagePixelFormatView. Not needed (MTLTextureUsage.pixelFormatView docs,
	// same rules as MoltenVK): same format (swizzle, texture type, level/slice range only) and linear <-> sRGB views.
	// Stencil-plane views of combined depth-stencil formats are treated as reinterpreting (MoltenVK does the same).
	static bool is_reinterpreting_view(MTL::PixelFormat image_format, MTL::PixelFormat view_format)
	{
		return get_linear_format(image_format) != get_linear_format(view_format);
	}

	static bool is_combined_depth_stencil_format(MTL::PixelFormat format)
	{
		return (get_format_aspect(format) & aspect_depth_stencil) == aspect_depth_stencil;
	}

	// Channel order of the 4-channel formats that exist in both orders. A view reinterpreting one order as the other
	// sees red and blue exchanged (BGRA8Unorm memory read as RGBA8Snorm: logical red is the stored blue byte).
	namespace
	{
		enum class rgb_order { none, rgb, bgr };
	}

	static rgb_order get_rgb_order(MTL::PixelFormat format)
	{
		switch (format)
		{
		case MTL::PixelFormatRGBA8Unorm:
		case MTL::PixelFormatRGBA8Unorm_sRGB:
		case MTL::PixelFormatRGBA8Snorm:
		case MTL::PixelFormatRGBA8Uint:
		case MTL::PixelFormatRGBA8Sint:
		case MTL::PixelFormatRGB10A2Unorm:
		case MTL::PixelFormatRGB10A2Uint:
			return rgb_order::rgb;
		case MTL::PixelFormatBGRA8Unorm:
		case MTL::PixelFormatBGRA8Unorm_sRGB:
		case MTL::PixelFormatBGR10A2Unorm:
			return rgb_order::bgr;
		default:
			return rgb_order::none;
		}
	}

	static MTL::TextureSwizzle swap_red_blue(MTL::TextureSwizzle source)
	{
		switch (source)
		{
		case MTL::TextureSwizzleRed: return MTL::TextureSwizzleBlue;
		case MTL::TextureSwizzleBlue: return MTL::TextureSwizzleRed;
		default: return source;
		}
	}

	void image::create_impl(const render_device& dev, const image_create_info& create_info)
	{
		info = create_info;
		m_format_class = create_info.format_class;

		// Metal aborts the process on an invalid texture descriptor (MTLTextureDescriptor validateWithDevice). Keep the
		// dimensions and the mip count inside its limits; callers that ask for more get what can exist (logged once).
		{
			const u32 w0 = info.width, h0 = info.height, d0 = info.depth, m0 = info.mipmaps;
			constexpr u32 max_dim = 16384;
			info.width = std::clamp<u32>(info.width, 1, max_dim);
			info.height = std::clamp<u32>(info.height, 1, max_dim);
			info.depth = std::clamp<u32>(info.depth, 1, 2048);

			const u32 largest = std::max({ info.width, info.height, info.type == MTL::TextureType3D ? info.depth : 1u });
			const u32 max_levels = (info.samples > 1) ? 1u : static_cast<u32>(std::bit_width(largest));
			info.mipmaps = std::clamp<u32>(info.mipmaps, 1, max_levels);

			if (info.width != w0 || info.height != h0 || (info.type == MTL::TextureType3D && info.depth != d0) || (m0 > info.mipmaps))
			{
				static atomic_t<u32> s_reports = 0;
				if (s_reports++ < 8)
				{
					rsx_log.error("Metal: texture '%s' requested %ux%ux%u with %u mip level(s), outside Metal's limits: created as %ux%ux%u with %u",
						m_debug_name, w0, h0, d0, m0, info.width, info.height, info.depth, info.mipmaps);
				}
			}
		}

		autorelease_scope pool;
		auto desc = ref(MTL::TextureDescriptor::alloc()->init());
		desc->setTextureType(info.type);
		desc->setPixelFormat(info.format);
		desc->setWidth(info.width);
		desc->setHeight(info.height);
		desc->setDepth(info.type == MTL::TextureType3D ? info.depth : 1);
		desc->setMipmapLevelCount(std::max(1u, info.mipmaps));
		desc->setSampleCount(std::max<u8>(1, info.samples));

		u32 array_length = std::max(1u, info.layers);
		if (info.type == MTL::TextureTypeCube || info.type == MTL::TextureTypeCubeArray)
		{
			ensure(array_length % 6 == 0);
			array_length /= 6;
		}
		desc->setArrayLength(array_length);

		// Lossless compression: Apple GPUs compress Private textures transparently (allowGPUOptimizedContents, left at
		// its default of true) unless the usage forbids it. MTLTextureUsagePixelFormatView is one of the things that
		// disables it, so it is only set where a view with another component layout can be created:
		//  - requested by the creator through info.usage (texture cache images that may be sampled through an snorm
		//    view, see MTLTextureCache.cpp),
		//  - always for combined depth-stencil formats, whose stencil is sampled through an X32_Stencil8 view.
		// RSX surface aliasing / typeless transfers never use views (they copy through buffers, see MTLTexture.cpp), and
		// swizzle, sRGB, texture type and subresource range views do not need the flag.
		if (is_combined_depth_stencil_format(info.format))
		{
			info.usage |= MTL::TextureUsagePixelFormatView;
		}

		desc->setUsage(info.usage);
		desc->setStorageMode(info.storage == memory_location::host_visible ? MTL::StorageModeShared : MTL::StorageModePrivate);
		desc->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);

		value = dev.handle()->newTexture(desc.get());
		if (!(value))
		{
			fmt::throw_exception("Metal: failed to create texture %ux%ux%u fmt=%d mips=%u layers=%u samples=%u", info.width, info.height, info.depth, static_cast<int>(info.format), info.mipmaps, info.layers, info.samples);
		}

		g_render_device->make_resident(value);
	}

	image::image(const render_device& dev, const image_create_info& create_info)
	{
		create_impl(dev, create_info);
	}

	image::~image()
	{
		if (value)
		{
			if (g_render_device)
			{
				g_render_device->evict(value);
			}

			value->release();
			value = nullptr;
		}
	}

	bool image::supports_view_format(MTL::PixelFormat view_format) const
	{
		return (info.usage & MTL::TextureUsagePixelFormatView) || !is_reinterpreting_view(format(), view_format);
	}

	void image::set_debug_name(const std::string& name)
	{
		m_debug_name = name;

		// A label is an NSString plus a setLabel message; only worth it when a debugging tool shows it
		if (value && debug_labels_enabled())
		{
			autorelease_scope pool;
			value->setLabel(ns_str(name));
		}
	}

	// ---------------------------------------------------------------------------------------------

	image_view::image_view(mtl::image* resource, const image_view_info& view_info)
		: m_resource(resource), info(view_info)
	{
		create_impl(false);
	}

	image_view::image_view(mtl::image* resource, const image_view_info& view_info, transient_tag)
		: m_resource(resource), info(view_info)
	{
		create_impl(true);
	}

	std::unique_ptr<image_view> image_view::make_transient(mtl::image* resource, const image_view_info& view_info)
	{
		return std::unique_ptr<image_view>(new image_view(resource, view_info, transient_tag{}));
	}

	void image_view::create_impl(bool transient)
	{
		ensure(m_resource && m_resource->value);
		m_parent_texture = m_resource->value;

		if (info.format == MTL::PixelFormatInvalid)
		{
			info.format = m_resource->format();
		}

		if (info.type == static_cast<MTL::TextureType>(~0ull))
		{
			info.type = m_resource->type();
		}

		if (info.level_count == ~0u)
		{
			info.level_count = m_resource->mipmaps() - info.base_level;
		}

		if (info.layer_count == ~0u)
		{
			info.layer_count = m_resource->layers() - info.base_layer;
		}

		// Depth-stencil formats: select the plane the shader will sample.
		MTL::PixelFormat view_format = info.format;
		const u32 fmt_aspect = get_format_aspect(view_format);
		if ((fmt_aspect & aspect_depth_stencil) == aspect_depth_stencil && (info.aspect & aspect_depth_stencil) == aspect_stencil)
		{
			// Stencil-only view
			view_format = (view_format == MTL::PixelFormatDepth24Unorm_Stencil8) ? MTL::PixelFormatX24_Stencil8 : MTL::PixelFormatX32_Stencil8;
		}

		info.format = view_format;

		if (!(m_resource->info.usage & MTL::TextureUsagePixelFormatView) &&
			is_reinterpreting_view(m_resource->format(), view_format)) [[unlikely]]
		{
			// Caller bug: the image must be created with MTLTextureUsagePixelFormatView (see image::create_impl).
			// Without it the view may read the (losslessly compressed) texture with the wrong layout.
			rsx_log.error("Metal: view format %d of texture '%s' (format %d) needs MTLTextureUsagePixelFormatView, which the texture was not created with",
				static_cast<int>(view_format), m_resource->debug_name(), static_cast<int>(m_resource->format()));
		}

		if (transient && g_transient_views.create_view(m_resource->value, view_format, info, m_pool_entry, resource_id))
		{
			return;
		}

		value = m_resource->value->newTextureView(
			view_format,
			info.type,
			NS::Range::Make(info.base_level, info.level_count),
			NS::Range::Make(info.base_layer, info.layer_count),
			info.swizzle);

		if (!(value))
		{
			fmt::throw_exception("Metal: failed to create texture view (fmt=%d type=%d)", static_cast<int>(view_format), static_cast<int>(info.type));
		}

		resource_id = value->gpuResourceID();
	}

	image_view::~image_view()
	{
		m_subviews.clear();

		if (m_pool_entry != umax)
		{
			g_transient_views.release_view(m_pool_entry);
		}

		if (value)
		{
			value->release();
			value = nullptr;
		}
	}

	image_view* image_view::as(MTL::PixelFormat format)
	{
		if (info.format == format)
		{
			return this;
		}

		auto self = m_root_view ? m_root_view : this;
		if (auto found = self->m_subviews.find(format); found != self->m_subviews.end())
		{
			return found->second.get();
		}

		image_view_info sub_info = info;
		sub_info.format = format;

		// The swizzle selects channels of the view format: across BGR/RGB orders the stored red and blue bytes trade
		// places, so exchange the selectors to keep returning the channels this view returns.
		const auto from_order = get_rgb_order(info.format);
		const auto to_order = get_rgb_order(format);
		if (from_order != rgb_order::none && to_order != rgb_order::none && from_order != to_order)
		{
			sub_info.swizzle.red = swap_red_blue(sub_info.swizzle.red);
			sub_info.swizzle.green = swap_red_blue(sub_info.swizzle.green);
			sub_info.swizzle.blue = swap_red_blue(sub_info.swizzle.blue);
			sub_info.swizzle.alpha = swap_red_blue(sub_info.swizzle.alpha);
		}

		auto view = std::make_unique<image_view>(m_resource, sub_info);
		view->m_root_view = self;

		auto result = view.get();
		self->m_subviews.emplace(format, std::move(view));
		return result;
	}

	// ---------------------------------------------------------------------------------------------

	MTL::TextureSwizzleChannels apply_swizzle_remap(const std::array<MTL::TextureSwizzle, 4>& base_remap_argb, const rsx::texture_channel_remap_t& remap)
	{
		const auto final_mapping = remap.remap(base_remap_argb, MTL::TextureSwizzleZero, MTL::TextureSwizzleOne);
		MTL::TextureSwizzleChannels result;
		result.red = final_mapping[1];
		result.green = final_mapping[2];
		result.blue = final_mapping[3];
		result.alpha = final_mapping[0];
		return result;
	}

	image_view* viewable_image::get_view(const rsx::texture_channel_remap_t& remap, u32 aspect_mask)
	{
		u32 remap_encoding = remap.encoded;
		if (remap_encoding == MTL_REMAP_IDENTITY && native_component_map == swizzle_identity)
		{
			remap_encoding = RSX_TEXTURE_REMAP_IDENTITY;
		}

		const u64 storage_key = remap_encoding | (static_cast<u64>(aspect_mask) << 32);
		if (auto found = views.find(storage_key); found != views.end())
		{
			return found->second.get();
		}

		MTL::TextureSwizzleChannels mapping;
		switch (remap_encoding)
		{
		case MTL_REMAP_IDENTITY:
			mapping = swizzle_identity;
			break;
		case RSX_TEXTURE_REMAP_IDENTITY:
			mapping = native_component_map;
			break;
		default:
			mapping = apply_swizzle_remap(
				{ native_component_map.alpha, native_component_map.red, native_component_map.green, native_component_map.blue },
				remap);
			break;
		}

		image_view_info view_info{};
		view_info.swizzle = mapping;
		view_info.aspect = aspect() & aspect_mask;
		ensure(view_info.aspect);

		auto view = std::make_unique<mtl::image_view>(this, view_info);
		auto result = view.get();
		views.emplace(storage_key, std::move(view));
		return result;
	}

	image_view* viewable_image::get_identity_view(u32 aspect_mask)
	{
		rsx::texture_channel_remap_t identity{};
		identity.encoded = MTL_REMAP_IDENTITY;
		identity.control_map = { CELL_GCM_TEXTURE_REMAP_REMAP, CELL_GCM_TEXTURE_REMAP_REMAP, CELL_GCM_TEXTURE_REMAP_REMAP, CELL_GCM_TEXTURE_REMAP_REMAP };
		identity.channel_map = { 0, 1, 2, 3 };
		return get_view(identity, aspect_mask);
	}

	image_view* viewable_image::get_subresource_view(u32 level, u32 layer, u32 aspect)
	{
		// get_view() keys use bits 0-34 (remap encoding | aspect_mask << 32); bit 63 tags subresource views
		ensure(level < (1u << 16) && aspect < (1u << 7));
		const u64 storage_key = (1ull << 63) | (static_cast<u64>(aspect) << 56) | (static_cast<u64>(level) << 32) | layer;
		if (auto found = views.find(storage_key); found != views.end())
		{
			return found->second.get();
		}

		image_view_info view_info{};
		view_info.type = MTL::TextureType2D;
		view_info.base_level = level;
		view_info.level_count = 1;
		view_info.base_layer = layer;
		view_info.layer_count = 1;
		view_info.aspect = aspect;

		auto view = std::make_unique<mtl::image_view>(this, view_info);
		auto result = view.get();
		views.emplace(storage_key, std::move(view));
		return result;
	}

	void viewable_image::set_native_component_layout(const MTL::TextureSwizzleChannels& new_layout)
	{
		if (!(new_layout == native_component_map))
		{
			native_component_map = new_layout;

			// Views depend on the native layout; drop them (deferred, GPU may still reference them)
			release_views();
		}
	}

	void viewable_image::release_views()
	{
		// Metal 4 command buffers do not retain resources and argument tables hold raw resource IDs,
		// so a view may still be referenced by in-flight work. Defer destruction through the GC.
		if (auto gc = get_gc())
		{
			for (auto& [key, view] : views)
			{
				gc->dispose(view);
			}
		}
		views.clear();
	}
}
