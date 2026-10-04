#pragma once
#include "../Program/FragmentProgramDecompiler.h"
#include "../Program/GLSLTypes.h"
#include "MTLProgramPipeline.h"

// Port of VK/VKFragmentProgram. Emits the same Vulkan-flavoured GLSL 450 (set 1 = fragment resources, identical
// binding numbering and inputs list) which is translated to MSL by the pipe compiler. Metal-specific differences:
//  - 1D textures are declared/sampled as 2D (height 1) textures at t = 0.5 (Metal 1D textures cannot have mipmaps).
//  - No GL_EXT_fragment_shader_barycentric: the renderer must report supports_normalized_barycentrics = true
//    (Apple GPUs interpolate correctly), RSX_SHADER_CONTROL_ATTRIBUTE_INTERPOLATION is ignored if ever set.
//  - Programmable blending subpassInput frag_src_N reads [[color(N)]] via framebuffer fetch (no binding needed).
//  - Program inputs list the 16 texture units first, then stencil mirrors, then frag_depth, so that if a stage
//    runs out of the 16 Metal sampler slots only those (nearest-sampled / texelFetch-only) lose their sampler slot.
//    depth_bounds_location follows them: the copy of the draw's depth buffer the depth bounds test reads.
//  - RSX_SHADER_CONTROL_DEPTH_BOUNDS_TEST (GPUs without a hardware depth bounds test): main() first tests the depth
//    STORED in the draw's depth buffer at the pixel (read from a copy made outside the pass; sample 0 of a
//    multisampled buffer) against the pushed bounds and discards when outside, like EXT_depth_bounds_test.
//  - Without hardware sampler LOD bias (pre-Apple10), the per-unit LOD bias is applied in the shader (see
//    MTLFragmentProgram::requires_lod_bias); gradient sampling scales the gradients by 2^bias; shadow compares and
//    depth->RGBA reads are not biased.
//  - Logic ops are not emulated: the shared ROP code exposes no logic-op control bits (see report / DESIGN.md).
//  - The per-draw offsets (fragment constants, context, texture parameters, stipple) are push constants
//    (draw_offsets_push_offset), not the VK backend's flat draw_params_payload varying: loads indexed by them are
//    uniform, so the Metal compiler can fetch them once per draw instead of once per fragment.

namespace mtl::interpreter
{
	struct source_builder;
}

class MTLFragmentDecompilerThread : public FragmentProgramDecompiler
{
	friend struct mtl::interpreter::source_builder; // Builds the shader interpreter from the same resource declarations

	std::string& m_shader;
	ParamArray& m_parrDummy;
	std::vector<mtl::glsl::program_input> inputs;
	class MTLFragmentProgram* mtl_prog;
	glsl::shader_properties m_shader_props{};

	void prepareBindingTable();

public:
	MTLFragmentDecompilerThread(std::string& shader, ParamArray& parr, const RSXFragmentProgram& prog, u32& size, class MTLFragmentProgram& dst)
		: FragmentProgramDecompiler(prog, size)
		, m_shader(shader)
		, m_parrDummy(parr)
		, mtl_prog(&dst)
	{
	}

	void Task();
	const std::vector<mtl::glsl::program_input>& get_inputs() { return inputs; }

	struct
	{
		// Samplers cannot apply mipLodBias on this GPU (sampler.cpp only sets it on Apple10): add the per-unit bias
		// in the shader instead (see MTLFragmentProgram::lod_bias_push_offset).
		bool emulate_sampler_lod_bias = false;
	}
	metal_props;

protected:
	std::string getFloatTypeName(usz elementCount) override;
	std::string getHalfTypeName(usz elementCount) override;
	std::string getFunction(FUNCTION f) override;
	std::string compareFunction(COMPARE f, std::string_view Op0, std::string_view Op1) override;

	void insertHeader(std::stringstream& OS) override;
	void insertInputs(std::stringstream& OS) override;
	void insertOutputs(std::stringstream& OS) override;
	void insertConstants(std::stringstream& OS) override;
	void insertGlobalFunctions(std::stringstream& OS) override;
	void insertMainStart(std::stringstream& OS) override;
	void insertMainEnd(std::stringstream& OS) override;
};

/** Storage for a Fragment Program in the process of recompilation.
 *  Decompile() + Compile() run on the thread that first requests the program (RSX thread or shader-cache loader).
 *  The expensive GLSL -> MSL -> MTLLibrary step is deferred to the pipe compiler (shader.compile(), once).
 */
class MTLFragmentProgram
{
public:
	MTLFragmentProgram();
	~MTLFragmentProgram();

	ParamArray parr;
	mtl::glsl::shader* handle = nullptr;   // Set by Compile(); points at `shader` (VK: the VkShaderModule)
	u32 id = 0;
	mtl::glsl::shader shader;
	std::vector<u32> constant_offsets;

	std::array<u32, 4> output_color_masks{ {} };
	std::vector<mtl::glsl::program_input> uniforms;

	// Shader-side sampler LOD bias (GPUs without sampler LOD bias support, i.e. !caps().apple10).
	// When set, the renderer must push one float per fragment texture unit before binding the program:
	//   push_constants(glsl::binding_set_index_fragment, lod_bias_push_offset, lod_bias_push_size, float[16])
	// where element i = the mip LOD bias of texture unit i (0 for unused units / units without mipmaps).
	// GLSL: push_constants_block { ...; layout(offset = 32) vec4 texture_lod_bias[4]; } (unit i = [i / 4][i % 4]).
	bool requires_lod_bias = false;
	static constexpr u32 lod_bias_push_offset = 32;
	static constexpr u32 lod_bias_push_size = 16 * sizeof(f32);

	// Per-draw ring offsets, pushed by the renderer for every draw (every fragment program declares them):
	//   push_constants(glsl::binding_set_index_fragment, draw_offsets_push_offset, draw_offsets_push_size, u32[4])
	// = { fragment constants offset (vec4 units), fragment context index, texture parameters base index, stipple
	// pattern offset (uvec4 units) }, the same values as draw_parameters_t::fs_*. GLSL: uvec4 fs_draw_offsets.
	static constexpr u32 draw_offsets_push_offset = 96;
	static constexpr u32 draw_offsets_push_size = 4 * sizeof(u32);

	// Depth bounds test in the shader (RSX_SHADER_CONTROL_DEPTH_BOUNDS_TEST; binding_table.depth_bounds_location != umax):
	// the renderer pushes { min, max } (clamped to [0, 1] like the hardware test) for every draw; the program tests its
	// depth stored in the draw's depth buffer (a copy, bound at depth_bounds_location) against them. GLSL:
	// push_constants_block { ...; layout(offset = 112) vec2 depth_bounds; }. Only these programs' push block reaches
	// past draw_offsets, so the layout of every other program is unchanged.
	static constexpr u32 depth_bounds_push_offset = draw_offsets_push_offset + draw_offsets_push_size;
	static constexpr u32 depth_bounds_push_size = 2 * sizeof(f32);

	struct
	{
		u32 context_buffer_location = umax;           // Rasterizer context
		u32 cbuf_location = umax;                     // Constants register file
		u32 tex_param_location = umax;                // Texture configuration data
		u32 polygon_stipple_params_location = umax;   // Polygon stipple settings
		u32 ftex_location[16];                        // Texture locations array
		u32 ftex_stencil_location[16];                // Texture stencil mirror array
		u32 frag_depth_input_location = umax;         // Fragment depth compare
		u32 depth_bounds_location = umax;             // Set for programs performing the depth bounds test (flagged programs only): copy of the depth buffer
		u32 frag_src_location[4];                     // Fragment input attachment locations (framebuffer fetch on Metal)

	} binding_table;

	void SetInputs(std::vector<mtl::glsl::program_input>& inputs);
	/**
	 * Decompile a fragment shader located in the PS3's Memory.  This function operates synchronously.
	 * @param prog RSXShaderProgram specifying the location and size of the shader in memory
	 */
	void Decompile(const RSXFragmentProgram& prog);

	/** Prepare the decompiled shader for the pipe compiler (MSL translation happens on first pipeline build). */
	void Compile();

private:
	/** Deletes the shader and any stored information */
	void Delete();
};
