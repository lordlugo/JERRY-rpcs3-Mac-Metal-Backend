#pragma once

// GLSL (Vulkan semantics) -> SPIR-V (glslang) -> MSL (SPIRV-Cross) -> MTL::Library (MTL4Compiler).
//
// The translation half is pure CPU work and is exposed separately so it can be reused by tooling (shader cache
// inspection, offline validation). Everything here is thread-safe and may run on pipe-compiler worker threads.
// The implementation TU is the only one in the backend built with exceptions (SPIRV-Cross throws on errors).

#include "MTLProgramPipeline.h"
#include "util/atomic.hpp"

#include <array>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mtl::glsl
{
	struct msl_translation_result
	{
		std::string msl;                         // Generated Metal Shading Language source
		std::string entry_point;                 // Entry point function name inside `msl`
		bool needs_buffer_size_buffer = false;   // MSL reads spvBufferSizeConstants at layout.buffer_count
		bool writes_storage = false;             // A storage buffer or image lacks NonWritable (GLSL readonly)
		std::array<u32, 3> workgroup_size{ 1, 1, 1 }; // Compute only: GLSL local_size (dispatch must use it)
		std::vector<std::pair<u32, MTL::VertexFormat>> vertex_attributes; // Vertex only: [[stage_in]] (location, format)
		// SPIR-V specialization constants (GLSL layout(constant_id = N) const ...), which the MSL declares as function
		// constants [[function_constant(N)]] defaulting to their GLSL value: (N, type), sorted by N
		std::vector<std::pair<u32, MTL::DataType>> function_constants;
	};

	// Unique (per stage) MSL entry point names given to every translated shader.
	std::string_view get_entry_point_name(::glsl::program_domain domain);

	// Translate Vulkan-flavoured GLSL to MSL using `layout` for every (set, binding) -> Metal index mapping.
	// Returns false and logs the GLSL (and whatever MSL could be produced) on failure.
	bool translate_glsl_to_msl(
		::glsl::program_domain domain,
		const std::string& glsl_source,
		const binding_layout& layout,
		msl_translation_result& result);

	// Build a Metal library from MSL through the device's MTL4Compiler. Returns an owned (+1) library or nullptr
	// (errors, including the MSL, are logged). `fast_math` (!g_cfg.video.disable_msl_fast_math) selects
	// MTL::MathModeRelaxed + fast floating-point functions (Inf/NaN preserved), otherwise MathModeSafe + precise functions.
	MTL::Library* compile_msl_library(const std::string& msl, std::string_view label, bool fast_math);

	// Debug switch: when set, every generated MSL source is written to the log (notice level).
	// Independently, "Log shader programs" (g_cfg.video.log_programs) dumps the MSL to shaderlog/*.msl.
	extern atomic_t<bool> g_dump_msl_to_log;
}
