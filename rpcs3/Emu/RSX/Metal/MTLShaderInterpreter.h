#pragma once

// Port of VK/VKShaderInterpreter: RSX vertex and fragment programs executed by uber shaders (the shared GLSL
// interpreters, Program/GLSLInterpreter) while their own pipelines compile, or instead of them in the "Interpreter
// only" shader mode. See DESIGN.md §8.
//
// Metal specifics:
//  - Each interpreter stage is compiled ONCE per library variant (the few options that change its interface: vertex
//    instancing; fragment colour output count, depth output and point sprite coordinates). Every other option is a
//    function constant (GLSL specialization constant, fragment_feature bits): the uber pipeline of a render state has
//    all of them on and serves every program; the pipeline specialized for a program's exact features is built in the
//    background and replaces it when ready (talks 1/4: "use the uber shader while the specialized variant compiles").
//  - Fragment textures are separate image arrays per sampling type plus one array of 16 samplers indexed by texture
//    unit (a stage has 16 sampler slots). Shadow, cube shadow and stencil mirror arrays are compacted: a unit's slot is
//    its rank among the units of that kind.
//  - Pipelines are built through the same path as recompiled programs (pipe_compiler::compile: flexible render
//    pipelines) and never waited for: a draw whose interpreter pipeline does not exist yet is skipped. A render state
//    whose unspecialized interpreter pipeline exists (same library variants, features and fixed state) gets its
//    pipeline right away, by specialization on the calling thread. Ownership: the interpreter decides which program
//    draws (uber or the pipeline specialized for the program's features); the pipe compiler's background full-state
//    upgrade replaces the pipeline state inside a program object, the interpreter's ones included (each object is
//    upgraded at most once, and the upgrade job does not own the program).
//  - Programs the interpreter cannot run as the recompiler would (see get()) are not interpreted.

#include "MTLProgramPipeline.h"
#include "Emu/RSX/Program/ProgramStateCache.h"
#include "Utilities/mutex.h"
#include "util/atomic.hpp"

#include <array>
#include <memory>
#include <unordered_map>

class MTLVertexProgram;
class MTLFragmentProgram;
struct RSXVertexProgram;
struct RSXFragmentProgram;

namespace mtl
{
	class image;
	class image_view;
	class command_list;
	struct sampler;

	namespace interpreter
	{
		// Library variants: options that change a stage's interface (bindings, outputs), so they are not function
		// constants. Vertex: instanced constants (the constants are two storage buffers instead of a uniform buffer).
		constexpr u32 vs_library_instancing = 1;
		constexpr u32 vs_library_count = 2;

		// Fragment: bits 0-2 colour output count (0-4, it must match the pipeline's colour attachments), depth output
		// (depth export, or early Z disabled), point sprite coordinates ([[point_coord]], point pipelines only)
		constexpr u32 fs_library_color_output_mask = 7;
		constexpr u32 fs_library_depth_output = 8;
		constexpr u32 fs_library_point_coord = 16;
		constexpr u32 fs_library_count = 32;

		// Fragment interpreter function constants: bit N is `layout(constant_id = N) const bool`. A set bit compiles the
		// feature's code in; whether it applies to a draw is still decided at run time from the instruction block
		// header, so a pipeline with more bits set than a program needs draws it exactly the same.
		enum fragment_feature : u32
		{
			fs_feature_textures = 1u << 0,          // Any texture sampling instruction
			fs_feature_texture_2d = 1u << 1,        // Colour sampling of 1D (as 2D) and 2D units, depth part of redirected units
			fs_feature_texture_3d = 1u << 2,
			fs_feature_texture_cube = 1u << 3,
			fs_feature_shadow = 1u << 4,            // Depth compare (2D and cube shadow units)
			fs_feature_redirect = 1u << 5,          // Depth + stencil read as RGBA8 (Z24X8 reconstruction)
			fs_feature_texture_convert = 1u << 6,   // Texel processing: format conversion (sign, gamma, renormalization) and alpha kill
			fs_feature_texture_expand = 1u << 7,    // BX2 expansion of texture reads
			fs_feature_flow_control = 1u << 8,
			fs_feature_precision = 1u << 9,         // Precision modifiers (fp16 / fixed point clamps) on sources or destinations
			fs_feature_texcoord_control = 1u << 10, // 2D texture coordinates, point sprite coordinates, perspective correction
			fs_feature_alpha_test = 1u << 11,
			fs_feature_alpha_to_coverage = 1u << 12,
			fs_feature_srgb_output = 1u << 13,
			fs_feature_output_rounding = 1u << 14,
			fs_feature_output_remap = 1u << 15,
			fs_feature_polygon_stipple = 1u << 16,

			fs_feature_count = 17,
			fs_feature_all = (1u << fs_feature_count) - 1
		};

		// GLSL bindings not described by the MTLVertexProgram / MTLFragmentProgram binding tables
		constexpr u32 vertex_instructions_location = 7;
		constexpr u32 fragment_instructions_location = 3;
		constexpr u32 fragment_textures_location = 4;      // 6 arrays: 2D, 3D, cube, shadow 2D, shadow cube, stencil
		constexpr u32 fragment_samplers_location = 10;

		// Fragment texture arrays
		constexpr u32 texture_units = 16;
		constexpr u32 shadow_2d_slots = 8;
		constexpr u32 shadow_cube_slots = 4;
		constexpr u32 stencil_slots = 4;

		// Instruction blocks: header words before the microcode
		constexpr u32 vertex_header_size = 16;
		constexpr u32 fragment_header_size = 32;

		// GLSL of the interpreter libraries (MTLShaderInterpreterSource.cpp). Pure source generation, no Metal objects,
		// so tools can run it off-device. Fills the program's shader source, inputs and binding table.
		struct source_builder
		{
			static void build_vertex(MTLVertexProgram& prog, u32 library);
			static void build_fragment(MTLFragmentProgram& prog, u32 library, bool emulate_lod_bias);
		};
	}

	class shader_interpreter
	{
	public:
		// Why get() returned no program
		enum class skip_reason : u32
		{
			none = 0,
			not_ready,   // The interpreter pipeline for this render state is being built
			unsupported, // The interpreter cannot run this program or texture setup
			inexact,     // It would not render it exactly as the recompiler does
			failed,      // The interpreter pipeline could not be built
			count
		};

		// Where the current fragment program's texture units are bound (update_fragment_textures)
		struct texture_unit_binding
		{
			const mtl::image_view* view = nullptr;          // Image, depth view (shadow, redirected) or placeholder
			const mtl::image_view* stencil_view = nullptr;  // Redirected units: stencil view
			const mtl::sampler* sampler = nullptr;
		};

		struct texture_environment
		{
			std::array<texture_unit_binding, interpreter::texture_units> units{};

			// Placeholders for array elements no unit uses: colour 2D, 3D, cube; depth 2D, depth cube
			std::array<const mtl::image_view*, 5> null_views{};
		};

		struct stats
		{
			u32 pipelines_built = 0;   // Uber pipelines (one per render state)
			u32 variants_built = 0;    // Pipelines specialized for a program's features
			u32 pipelines_failed = 0;
		};

		shader_interpreter();
		~shader_interpreter();

		shader_interpreter(const shader_interpreter&) = delete;
		shader_interpreter& operator=(const shader_interpreter&) = delete;

		// `apple10`: samplers apply the LOD bias (otherwise the shader adds it, as recompiled programs do)
		void init(bool apple10);
		void destroy();

		// Queues the interpreter's shaders and a first pipeline on the pipe compiler workers. Never blocks.
		void preload();

		// The interpreter program for this draw, or nullptr (see get_skip_reason()). Requests the pipelines it lacks.
		// Programs it cannot draw exactly as the recompiler does are left to the recompiler (skip_reason unsupported or
		// inexact), in both shader modes that use the interpreter.
		glsl::program* get(
			const mtl::pipeline_props& properties,
			const RSXFragmentProgram& fp,
			const program_hash_util::fragment_program_utils::fragment_program_metadata& fp_metadata,
			u64 fp_ucode_hash,
			const RSXVertexProgram& vp,
			const program_hash_util::vertex_program_utils::vertex_program_metadata& vp_metadata,
			u64 vp_ucode_hash);

		skip_reason get_skip_reason() const { return m_skip_reason; }

		bool is_interpreter(const glsl::program* prog) const;

		// Binding tables of the current interpreter program (valid after get() returned a program)
		std::pair<const MTLVertexProgram*, const MTLFragmentProgram*> get_shaders() const;

		// Instruction block headers of the current program (vertex_header_size / fragment_header_size bytes)
		void write_vertex_header(void* dst, const RSXVertexProgram& vp, bool two_sided_lighting) const;
		void write_fragment_header(void* dst, const RSXFragmentProgram& fp, u32 shader_control) const;

		// Binds the texture arrays and samplers of the current program
		void update_fragment_textures(mtl::command_list& cmd, glsl::program& program, const RSXFragmentProgram& fp, const texture_environment& env);

		// The fragment interpreter adds the sampler LOD bias itself (MTLFragmentProgram::lod_bias_push_offset)
		bool requires_lod_bias() const { return !m_apple10; }

		stats get_stats_and_reset();
		usz get_pipeline_count() const;

	private:
		struct pipeline_key
		{
			u32 vs_library = 0;
			u32 fs_library = 0;
			u32 fs_features = 0;
			mtl::pipeline_props properties{};

			bool operator==(const pipeline_key& other) const
			{
				return vs_library == other.vs_library && fs_library == other.fs_library && fs_features == other.fs_features &&
					properties == other.properties;
			}
		};

		struct key_hasher
		{
			usz operator()(const pipeline_key& key) const;
		};

		enum class pipeline_status : u8
		{
			building,
			ready,
			failed
		};

		struct pipeline_entry
		{
			pipeline_status status = pipeline_status::building;
			std::unique_ptr<glsl::program> program;
		};

		// What the microcode of a fragment program needs (cached by ucode hash)
		struct fragment_analysis
		{
			u32 features = 0;         // fragment_feature bits
			u32 unsupported = 0;      // Reasons it can never be interpreted (analysis_* bits)
			u32 inexact = 0;          // Reasons the interpreter would not match the recompiler
			u32 register_count = 0;   // Full registers the interpreter must zero-initialize
			bool reads_texcoords = false;         // Reads TEX0-TEX9 (texture coordinate control may apply)
			bool perspective_correction = false;  // Some read requests perspective correction
		};

		// What the current draw uses (texture state, controls), for the instruction block header
		struct fragment_draw_state
		{
			u32 ctrl = 0;
			u32 register_count = 0;
			u32 shadow_2d_mask = 0;
			u32 shadow_cube_mask = 0;
		};

		bool m_apple10 = false;

		std::array<std::unique_ptr<MTLVertexProgram>, interpreter::vs_library_count> m_vs_libraries;
		std::array<std::unique_ptr<MTLFragmentProgram>, interpreter::fs_library_count> m_fs_libraries;

		// Fragment shaders specialized for a feature set (the uber variant: every feature), key = (library << 32) | features.
		// They share the library shader's MSL and MTLLibrary.
		std::unordered_map<u64, std::unique_ptr<glsl::shader>> m_fs_variants;

		std::unordered_map<pipeline_key, pipeline_entry, key_hasher> m_pipelines;
		mutable shared_mutex m_pipeline_lock;
		u32 m_variant_requests = 0;

		std::unordered_map<u64, fragment_analysis> m_fragment_analysis; // ucode hash + export configuration
		std::unordered_map<u64, u32> m_vertex_analysis; // ucode hash -> inexact reasons

		// Current program (get())
		pipeline_key m_current_key{};
		glsl::program* m_current_program = nullptr;
		bool m_current_is_uber = false;
		u32 m_current_vs_library = 0;
		u32 m_current_fs_library = 0;
		fragment_draw_state m_fs_state{};
		skip_reason m_skip_reason = skip_reason::none;

		// Placeholder for stencil mirror slots no unit uses (R8Uint)
		std::unique_ptr<mtl::image> m_null_stencil_image;

		u64 m_start_time = 0;
		atomic_t<bool> m_ready_logged = false;
		atomic_t<u32> m_pipelines_built = 0;
		atomic_t<u32> m_variants_built = 0;
		atomic_t<u32> m_pipelines_failed = 0;

		MTLVertexProgram& get_vertex_library(u32 library);
		MTLFragmentProgram& get_fragment_library(u32 library);
		glsl::shader* get_fragment_shader(u32 library, u32 features);

		// Requests the pipeline (status building). `ahead`: queued in front of the other background compiles. Returns
		// the program when pipe_compiler::compile created it right away (a specialization of an unspecialized pipeline
		// that exists: status ready), nullptr when it was queued.
		glsl::program* request_pipeline(const pipeline_key& key, bool ahead);

		const fragment_analysis& analyse_fragment_program(const RSXFragmentProgram& fp, u64 ucode_hash);
		u32 analyse_vertex_program(const RSXVertexProgram& vp, u64 ucode_hash);
	};
}
