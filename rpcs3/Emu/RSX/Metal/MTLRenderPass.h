#pragma once

#include "mtlutils/mtl_api.h"
#include "Utilities/geometry.h"

#include <array>
#include <vector>

// Render pass helpers. Vulkan's VkRenderPass + VkFramebuffer (+ their caches) collapse into a render pass descriptor
// built once per surface set, plus scope tracking of the renderer's main pass on a command list.

namespace mtl
{
	class image;
	class command_list;

	// A set of attachments (framebuffer). Colour attachments are bound in order at indices 0..color_count-1.
	struct framebuffer_info
	{
		std::array<mtl::image*, 4> color{};
		u32 color_count = 0;
		mtl::image* depth_stencil = nullptr;
		u32 width = 0;
		u32 height = 0;
		u8 samples = 1;

		void clear() { *this = {}; }
		bool empty() const { return !color_count && !depth_stencil; }
		bool has_stencil() const;

		// Snapshot of the attachment textures (images may swap their MTLTexture, e.g. when unspilled)
		std::array<const MTL::Texture*, 5> get_textures() const;
	};

	// Key describing attachment formats + sample count (the part of a render pass a pipeline must be compatible with).
	u64 get_renderpass_key(const std::vector<mtl::image*>& images);
	u64 get_renderpass_key(MTL::PixelFormat color_format, MTL::PixelFormat depth_format = MTL::PixelFormatInvalid, u8 color_attachment_count = 1, u8 sample_count = 1);

	// Creates a render pass descriptor (+1 reference) for a framebuffer. All attachments load and store their contents
	// (deferred clears of the attachments turn loads into clears when the pass begins, see command_list::defer_clear).
	// `visibility_result_buffer` may be null. Visibility results accumulate across passes (see MTLQueryPool.h).
	MTL4::RenderPassDescriptor* create_render_pass_descriptor(const framebuffer_info& fb, const MTL::Buffer* visibility_result_buffer = nullptr);

	// Updates the visibility result buffer of an existing descriptor
	void set_visibility_result_buffer(MTL4::RenderPassDescriptor* desc, const MTL::Buffer* visibility_result_buffer);

	// Clears [0, width) x [0, height) of a colour texture: the load action (Clear) of the next pass rendering into it,
	// or a clear-only pass (command_list::defer_clear)
	void clear_color_texture(mtl::command_list& cmd, MTL::Texture* target, u32 width, u32 height, const MTL::ClearColor& color);

	// Tracks the renderer's main render pass on a command list (vk::begin_renderpass/renderpass_op equivalent).
	// Other components open their own passes through cmd.begin_render_pass(), which silently ends ours; Metal cannot
	// tell which descriptor an encoder was created from, so we remember the encoder we opened and compare. The encoder is
	// retained so that its address cannot be recycled for another encoder while we hold it.
	class render_pass_tracker
	{
		const mtl::command_list* m_cmd = nullptr;
		ref<MTL4::RenderCommandEncoder> m_encoder;
		u64 m_pass_id = 0;

	public:
		render_pass_tracker() = default;

		// Begins the renderer's draw pass (see command_list::begin_render_pass)
		MTL4::RenderCommandEncoder* begin(mtl::command_list& cmd, MTL4::RenderPassDescriptor* desc);
		void reset();

		// True if the pass we opened is still the active encoder of `cmd`
		bool is_open(const mtl::command_list& cmd) const;

		// The active encoder of our pass or nullptr
		MTL4::RenderCommandEncoder* encoder(const mtl::command_list& cmd) const;

		// Number of passes opened so far (monotonic)
		u64 pass_id() const { return m_pass_id; }
	};
}
