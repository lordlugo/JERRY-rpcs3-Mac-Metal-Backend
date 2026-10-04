#include "stdafx.h"
#include "MTLShaderInterpreter.h"
#include "MTLVertexProgram.h"
#include "MTLFragmentProgram.h"
#include "MTLPipelineCompiler.h"
#include "MTLHelpers.h"
#include "mtlutils/device.h"
#include "mtlutils/image.h"
#include "mtlutils/sampler.h"

#include "Emu/system_config.h"
#include "Emu/Cell/timers.hpp"
#include "util/fnv_hash.hpp"

#include <bit>

namespace mtl
{
	using namespace interpreter;

	namespace
	{
		// Why a program is not interpreted: unsupported (never) or inexact (only in the Interpreter only mode)
		enum analysis_reason : u32
		{
			reason_vertex_textures = 1u << 0,       // Vertex texture fetch
			reason_msaa_textures = 1u << 1,         // Sampling a multisampled texture
			reason_programmable_blending = 1u << 2, // Blending in the shader (framebuffer fetch)
			reason_depth_compare = 1u << 3,         // Emulated depth compare (reads the bound depth buffer)
			reason_texture_type = 1u << 4,          // Depth compare on a 3D unit, depth read as RGBA8 from a cube or 3D unit
			reason_texture_slots = 1u << 5,         // More depth compare / stencil mirror units than the interpreter binds
			reason_register_index = 1u << 6,        // Register index outside the register file
			reason_program_layout = 1u << 7,        // Branch target outside the program or not forward, ambiguous instruction length
			reason_nested_flow = 1u << 8,           // IF/ELSE inside an IF, loop inside a loop, loop and IF ending together
			reason_loop_counter = 1u << 9,          // Loop with a start or step the recompiler ignores
			reason_instruction = 1u << 10,          // Instruction the interpreter does not execute like the recompiler
			reason_input_register = 1u << 11,       // Undefined input register
			reason_indexed_input = 1u << 12,        // Input register indexed by the loop counter
			reason_register_aliasing = 1u << 13,    // A register read with the other precision than it was written with
			reason_depth_bounds = 1u << 14,         // Depth bounds test in the program (recompiled programs only)

			reason_count = 15
		};

		constexpr std::array<const char*, reason_count> s_reason_names =
		{
			"vertex texture fetch",
			"multisampled texture sampling",
			"programmable blending",
			"emulated depth compare",
			"depth compare on 3D textures or depth read as colour from cube/3D textures",
			"more than 8 depth compare 2D, 4 depth compare cube or 4 depth-as-colour texture units",
			"register index outside the register file",
			"branch targets outside the program or ambiguous instruction length",
			"nested IF/ELSE or loops",
			"loops with a start or step value",
			"TEXBEM/TXPBEM/BEMLUM/TIMESWTEX/POW or undefined instructions",
			"undefined input register",
			"input registers indexed by the loop counter",
			"registers read with another precision than they were written with",
			"the depth bounds test (emulated in the fragment program)",
		};

		// Reasons reported so far (logged once per session each)
		atomic_t<u32> g_reported_reasons = 0;

		void report_reasons(u32 reasons, bool inexact)
		{
			const u32 fresh = reasons & ~g_reported_reasons.fetch_or(reasons);
			for (u32 bit = 0; bit < reason_count; ++bit)
			{
				if (fresh & (1u << bit))
				{
					if (inexact)
					{
						rsx_log.notice("Metal: the shader interpreter does not run programs using %s: their draws wait for the recompiled pipeline.", s_reason_names[bit]);
					}
					else
					{
						rsx_log.notice("Metal: the shader interpreter cannot run programs using %s: their draws wait for the recompiled pipeline.", s_reason_names[bit]);
					}
				}
			}
		}

		constexpr u32 register_file_size = 48; // R0-R47 / H0-H47

		u32 fp_swap(u32 word)
		{
			// Fragment program words are stored as big-endian 16-bit halves
			return ((word & 0x00FF00FF) << 8) | ((word & 0xFF00FF00) >> 8);
		}

		// Operands the shared interpreter reads for an opcode (FragmentInterpreter.glsl instruction classes)
		u32 get_interpreter_operand_count(u32 opcode)
		{
			switch (opcode)
			{
			case RSX_FP_OPCODE_NOP:
			case RSX_FP_OPCODE_FENCT:
			case RSX_FP_OPCODE_FENCB:
			case RSX_FP_OPCODE_KIL:
				return 0;
			case RSX_FP_OPCODE_MOV:
			case RSX_FP_OPCODE_FRC:
			case RSX_FP_OPCODE_FLR:
			case RSX_FP_OPCODE_DDX:
			case RSX_FP_OPCODE_DDY:
			case RSX_FP_OPCODE_RCP:
			case RSX_FP_OPCODE_RSQ:
			case RSX_FP_OPCODE_EX2:
			case RSX_FP_OPCODE_LG2:
			case RSX_FP_OPCODE_STR:
			case RSX_FP_OPCODE_SFL:
			case RSX_FP_OPCODE_COS:
			case RSX_FP_OPCODE_SIN:
			case RSX_FP_OPCODE_NRM:
			case RSX_FP_OPCODE_LIT:
			case RSX_FP_OPCODE_LIF:
			case RSX_FP_OPCODE_TEX:
			case RSX_FP_OPCODE_TXP:
			case RSX_FP_OPCODE_PK2:
			case RSX_FP_OPCODE_PK4:
			case RSX_FP_OPCODE_PK16:
			case RSX_FP_OPCODE_PKG:
			case RSX_FP_OPCODE_PKB:
			case RSX_FP_OPCODE_UP2:
			case RSX_FP_OPCODE_UP4:
			case RSX_FP_OPCODE_UP16:
			case RSX_FP_OPCODE_UPG:
			case RSX_FP_OPCODE_UPB:
				return 1;
			case RSX_FP_OPCODE_MUL:
			case RSX_FP_OPCODE_ADD:
			case RSX_FP_OPCODE_DP2:
			case RSX_FP_OPCODE_DP3:
			case RSX_FP_OPCODE_DP4:
			case RSX_FP_OPCODE_DST:
			case RSX_FP_OPCODE_MIN:
			case RSX_FP_OPCODE_MAX:
			case RSX_FP_OPCODE_SLT:
			case RSX_FP_OPCODE_SGE:
			case RSX_FP_OPCODE_SLE:
			case RSX_FP_OPCODE_SGT:
			case RSX_FP_OPCODE_SNE:
			case RSX_FP_OPCODE_SEQ:
			case RSX_FP_OPCODE_POW:
			case RSX_FP_OPCODE_DIV:
			case RSX_FP_OPCODE_DIVSQ:
			case RSX_FP_OPCODE_REFL:
			case RSX_FP_OPCODE_TXL:
			case RSX_FP_OPCODE_TXB:
				return 2;
			default:
				return 3;
			}
		}

		bool is_texture_opcode(u32 opcode)
		{
			switch (opcode)
			{
			case RSX_FP_OPCODE_TEX:
			case RSX_FP_OPCODE_TXP:
			case RSX_FP_OPCODE_TXD:
			case RSX_FP_OPCODE_TXB:
			case RSX_FP_OPCODE_TXL:
				return true;
			default:
				return false;
			}
		}

		// Instructions the interpreter implements like the recompiler (FragmentProgramDecompiler::handle_sct_scb/tex_srb)
		bool is_interpreted_opcode(u32 opcode)
		{
			switch (opcode)
			{
			case RSX_FP_OPCODE_POW:       // Not decompiled
			case RSX_FP_OPCODE_TEXBEM:    // Not implemented by the interpreter
			case RSX_FP_OPCODE_TXPBEM:
			case RSX_FP_OPCODE_BEMLUM:
			case RSX_FP_OPCODE_TIMESWTEX:
				return false;
			default:
				return opcode <= RSX_FP_OPCODE_FENCB && opcode != 0x30 && opcode != 0x32;
			}
		}

		bool shader_precision_is_low()
		{
			return g_cfg.video.shader_precision == gpu_preset_level::low;
		}
	}

	usz shader_interpreter::key_hasher::operator()(const pipeline_key& key) const
	{
		usz hash = rpcs3::hash_struct(key.properties);
		hash = rpcs3::hash64(hash, (u64{key.vs_library} << 40) | (u64{key.fs_library} << 32) | key.fs_features);
		return hash;
	}

	shader_interpreter::shader_interpreter() = default;

	shader_interpreter::~shader_interpreter()
	{
		destroy();
	}

	void shader_interpreter::init(bool apple10)
	{
		m_apple10 = apple10;
		m_start_time = get_system_time();
	}

	void shader_interpreter::destroy()
	{
		// The pipe compiler is gone (no job references the shaders) and the GPU is idle (renderer teardown)
		m_current_program = nullptr;

		{
			std::lock_guard lock(m_pipeline_lock);
			m_pipelines.clear();
		}

		m_fs_variants.clear();

		for (auto& library : m_vs_libraries)
		{
			library.reset();
		}

		for (auto& library : m_fs_libraries)
		{
			library.reset();
		}

		m_fragment_analysis.clear();
		m_vertex_analysis.clear();
		m_null_stencil_image.reset();
	}

	MTLVertexProgram& shader_interpreter::get_vertex_library(u32 library)
	{
		auto& prog = m_vs_libraries[library];
		if (!prog)
		{
			prog = std::make_unique<MTLVertexProgram>();
			source_builder::build_vertex(*prog, library);
			prog->id = 0xF0000000 | library;

			if (g_cfg.video.log_programs)
			{
				fs::write_file(fs::get_cache_dir() + fmt::format("shaderlog/VertexInterpreter%u.glsl", library), fs::rewrite, prog->shader.get_source());
			}
		}

		return *prog;
	}

	MTLFragmentProgram& shader_interpreter::get_fragment_library(u32 library)
	{
		auto& prog = m_fs_libraries[library];
		if (!prog)
		{
			prog = std::make_unique<MTLFragmentProgram>();
			source_builder::build_fragment(*prog, library, !m_apple10);
			prog->id = 0xF0000000 | library;

			if (g_cfg.video.log_programs)
			{
				fs::write_file(fs::get_cache_dir() + fmt::format("shaderlog/FragmentInterpreter%u.glsl", library), fs::rewrite, prog->shader.get_source());
			}
		}

		return *prog;
	}

	glsl::shader* shader_interpreter::get_fragment_shader(u32 library, u32 features)
	{
		// Every pipeline sets all the constants (the uber pipeline: every feature)
		auto& base = get_fragment_library(library).shader;
		auto& variant = m_fs_variants[(u64{library} << 32) | features];
		if (!variant)
		{
			std::vector<glsl::function_constant> constants;
			constants.reserve(fs_feature_count);

			for (u32 id = 0; id < fs_feature_count; ++id)
			{
				constants.push_back({ .id = id, .value = (features >> id) & 1u });
			}

			variant = std::make_unique<glsl::shader>();
			variant->create_specialization(base, std::move(constants));
		}

		return variant.get();
	}

	glsl::program* shader_interpreter::request_pipeline(const pipeline_key& key, bool ahead)
	{
		{
			std::lock_guard lock(m_pipeline_lock);
			const auto [it, inserted] = m_pipelines.try_emplace(key);
			if (!inserted)
			{
				return nullptr;
			}
		}

		auto& vs = get_vertex_library(key.vs_library);
		auto& fs = get_fragment_library(key.fs_library);
		glsl::shader* fs_shader = get_fragment_shader(key.fs_library, key.fs_features);

		// `on_worker`: built by a pipe compiler job, whose build times this thread's record holds
		auto on_built = [this, key](std::unique_ptr<glsl::program>& program, bool on_worker)
		{
			const bool built = !!program;
			{
				std::lock_guard lock(m_pipeline_lock);
				auto& entry = m_pipelines[key];
				entry.program = std::move(program);
				entry.status = built ? pipeline_status::ready : pipeline_status::failed;
			}

			if (!built)
			{
				m_pipelines_failed++;
				rsx_log.error("Metal: a shader interpreter pipeline could not be built (vertex library %u, fragment library %u, features 0x%x, %u color attachments, %u samples). Draws that need it are skipped.",
					key.vs_library, key.fs_library, key.fs_features, key.properties.state.color_count, key.properties.state.sample_count);
				return;
			}

			if (key.fs_features != fs_feature_all)
			{
				m_variants_built++;
				return;
			}

			m_pipelines_built++;

			if (!m_ready_logged.exchange(true))
			{
				// Where its time went (the first uber pipeline is built by a job: libraries and pipeline)
				const compile_timings timings = on_worker ? this_thread_compile_timings() : compile_timings{};
				rsx_log.notice("Metal: shader interpreter ready %.0f ms after renderer start: draws whose pipeline is compiling are drawn by the interpreter "
					"(its job: GLSL->MSL %.0f ms for %u shader(s), MTLLibrary %.0f ms, pipelines %.0f ms without the pipeline archive, %.1f ms specialization)",
					(get_system_time() - m_start_time) / 1000., timings.translate_us / 1000., timings.translated, timings.library_us / 1000.,
					(timings.compiled_pipeline_us + timings.archive_pipeline_us) / 1000., timings.specialization_us / 1000.);
			}
		};

		pipe_compiler::op_flags flags = pipe_compiler::COMPILE_DEFERRED | pipe_compiler::SEPARATE_SHADER_OBJECTS;
		if (ahead)
		{
			flags |= pipe_compiler::COMPILE_AHEAD;
		}

		// Same path as recompiled programs (pipe compiler workers), without the pipeline archive (the fragment
		// interpreter is specialized with function constants: uses_pipeline_archive). The shaders outlive the job. A
		// specialization of an unspecialized pipeline that already exists is created right here instead: compile()
		// returns it and does not call the callback, so it is stored the same way.
		auto callback = [on_built](std::unique_ptr<glsl::program>& program)
		{
			on_built(program, true);
		};

		if (auto program = get_pipe_compiler()->compile(key.properties, vs.handle, fs_shader, flags, callback, vs.uniforms, fs.uniforms))
		{
			glsl::program* result = program.get();
			on_built(program, false);
			return result;
		}

		return nullptr;
	}

	void shader_interpreter::preload()
	{
		// Translates and compiles the common libraries (one colour output, no instancing) on a worker with a first uber
		// pipeline: a single BGRA8 attachment, the most common render state. Other libraries and pipelines follow the
		// draws that need them.
		pipeline_key key{};
		key.vs_library = 0;
		key.fs_library = 1;
		key.fs_features = fs_feature_all;

		auto& state = key.properties.state;
		state.color_count = 1;
		state.color[0].pixel_format = static_cast<u32>(MTL::PixelFormatBGRA8Unorm);
		state.color[0].write_mask = static_cast<u8>(MTL::ColorWriteMaskAll);
		state.sample_count = 1;
		state.topology_class = static_cast<u8>(MTL::PrimitiveTopologyClassTriangle);
		state.rasterization_enabled = 1;

		request_pipeline(key, true);

		rsx_log.notice("Metal: shader interpreter enabled: its shaders are being compiled in the background, ahead of every other background "
			"compile and at emulation priority (%s sampler LOD bias). Until a render state has its interpreter pipeline, draws whose pipeline "
			"is compiling are skipped.", m_apple10 ? "hardware" : "shader-side");
	}

	u32 shader_interpreter::analyse_vertex_program(const RSXVertexProgram& vp, u64 ucode_hash)
	{
		if (const auto found = m_vertex_analysis.find(ucode_hash); found != m_vertex_analysis.end())
		{
			return found->second;
		}

		if (m_vertex_analysis.size() >= 16384)
		{
			m_vertex_analysis.clear();
		}

		u32 reasons = 0;
		const u32 instruction_count = std::min<u32>(::size32(vp.data) / 4, rsx::max_vertex_program_instructions);

		auto check_temp_read = [&](u32 src)
		{
			const SRC source{ .HEX = src };
			if (source.reg_type == RSX_VP_REGISTER_TYPE_TEMP && source.tmp_src >= 32)
			{
				reasons |= reason_register_index;
			}
		};

		for (u32 i = 0; i < instruction_count; ++i)
		{
			if (!vp.instruction_mask[i])
			{
				continue;
			}

			const D0 d0{ .HEX = vp.data[i * 4 + 0] };
			const D1 d1{ .HEX = vp.data[i * 4 + 1] };
			const D2 d2{ .HEX = vp.data[i * 4 + 2] };
			const D3 d3{ .HEX = vp.data[i * 4 + 3] };

			const u32 src0 = (d1.src0h << 9) | d2.src0l;
			const u32 src1 = d2.src1;
			const u32 src2 = (d2.src2h << 11) | d3.src2l;

			// Vector unit (VertexInterpreter.glsl reads src0, and src1 / src2 as the opcode needs)
			switch (const u32 opcode = d1.vec_opcode)
			{
			case RSX_VEC_OPCODE_NOP:
				break;
			case RSX_VEC_OPCODE_MOV:
			case RSX_VEC_OPCODE_ARL:
			case RSX_VEC_OPCODE_FRC:
			case RSX_VEC_OPCODE_FLR:
			case RSX_VEC_OPCODE_SFL:
			case RSX_VEC_OPCODE_STR:
			case RSX_VEC_OPCODE_SSG:
			case RSX_VEC_OPCODE_TXL:
				check_temp_read(src0);
				break;
			case RSX_VEC_OPCODE_ADD:
				check_temp_read(src0);
				check_temp_read(src2);
				break;
			case RSX_VEC_OPCODE_MAD:
				check_temp_read(src0);
				check_temp_read(src1);
				check_temp_read(src2);
				break;
			default:
				if (opcode > RSX_VEC_OPCODE_SSG)
				{
					// Undefined: the recompiler ends the program, the interpreter would copy src0
					reasons |= reason_instruction;
				}

				check_temp_read(src0);
				check_temp_read(src1);
				break;
			}

			if (d1.vec_opcode != RSX_VEC_OPCODE_NOP && d1.vec_opcode != RSX_VEC_OPCODE_ARL && d0.dst_tmp != 0x3f && d0.dst_tmp >= 32)
			{
				reasons |= reason_register_index;
			}

			// Scalar unit (reads src2 for every opcode)
			if (const u32 opcode = d1.sca_opcode; opcode != RSX_SCA_OPCODE_NOP)
			{
				check_temp_read(src2);

				if (d3.sca_dst_tmp != 0x3f && d3.sca_dst_tmp >= 32)
				{
					reasons |= reason_register_index;
				}

				if (opcode == RSX_SCA_OPCODE_CLI || opcode >= RSX_SCA_OPCODE_PSH)
				{
					// CLI is a call for the recompiler and a no-op for the interpreter; PSH/POP and undefined opcodes
					// are no-ops for the recompiler and moves for the interpreter
					reasons |= reason_instruction;
				}

				if (opcode == RSX_SCA_OPCODE_BRA)
				{
					// Jump by address register: the RSX analysis does not follow it, so the target may lie outside the
					// microcode in the instruction block (the interpreter would read past it)
					reasons |= reason_program_layout;
				}
			}
		}

		m_vertex_analysis[ucode_hash] = reasons;
		return reasons;
	}

	const shader_interpreter::fragment_analysis& shader_interpreter::analyse_fragment_program(const RSXFragmentProgram& fp, u64 ucode_hash)
	{
		// The output stage reads the exported registers: the export configuration is part of the key
		const u32 export_config = (fp.ctrl & (CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS | CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT)) | (fp.mrt_buffers_count << 16);
		const u64 key = rpcs3::hash64(ucode_hash, export_config);

		if (const auto found = m_fragment_analysis.find(key); found != m_fragment_analysis.end())
		{
			return found->second;
		}

		if (m_fragment_analysis.size() >= 16384)
		{
			m_fragment_analysis.clear();
		}

		fragment_analysis result{};
		const u8* data = static_cast<const u8*>(fp.get_data());
		const u32 slot_count = data ? fp.ucode_length / 16 : 0;

		// Registers written so far, per precision (bit = register index); the highest indices used
		u64 written32 = 0;
		u64 written16 = 0;
		u32 used32 = 0;
		u32 used16 = 0;

		// Structure: slots holding an instruction, flow control targets, open IF/ELSE then-block and loop
		std::vector<bool> is_instruction(slot_count, false);
		std::vector<u32> targets;
		u32 then_end = umax;
		u32 loop_end = umax;
		u32 last_instruction = umax;

		auto read_register = [&](u32 index, bool fp16)
		{
			if (index >= register_file_size)
			{
				result.unsupported |= reason_register_index;
				return;
			}

			if (fp16)
			{
				used16 = std::max(used16, index + 1);

				// H<n> is half of R<n/2>: the recompiler converts the lanes (RegisterDependencyPass), the interpreter does not
				result.inexact |= ((written32 >> (index / 2)) & 1) ? reason_register_aliasing : 0;
			}
			else
			{
				used32 = std::max(used32, index + 1);
				result.inexact |= (index < 24 && ((written16 >> (index * 2)) & 3)) ? reason_register_aliasing : 0;
			}
		};

		auto write_register = [&](u32 index, bool fp16)
		{
			if (index >= register_file_size)
			{
				result.unsupported |= reason_register_index;
				return;
			}

			if (fp16)
			{
				used16 = std::max(used16, index + 1);
				written16 |= 1ull << index;
			}
			else
			{
				used32 = std::max(used32, index + 1);
				written32 |= 1ull << index;
			}
		};

		bool ended = false;
		for (u32 slot = 0; slot < slot_count && !ended; ++slot)
		{
			u32 words[4];
			std::memcpy(words, data + slot * 16, sizeof(words));

			const OPDEST dst{ .HEX = fp_swap(words[0]) };
			const SRC0 src0{ .HEX = fp_swap(words[1]) };
			const SRC1 src1{ .HEX = fp_swap(words[2]) };
			const SRC2 src2{ .HEX = fp_swap(words[3]) };
			const u32 opcode = dst.opcode | (src1.opcode_hi << 6);

			is_instruction[slot] = true;
			last_instruction = slot;
			ended = !!dst.end;

			if (then_end != umax && slot >= then_end) then_end = umax;
			if (loop_end != umax && slot >= loop_end) loop_end = umax;

			if (src1.opcode_hi)
			{
				// Flow control (no operands, never followed by a literal)
				result.features |= fs_feature_flow_control;

				switch (opcode)
				{
				case RSX_FP_OPCODE_BRK:
				case RSX_FP_OPCODE_RET:
				case RSX_FP_OPCODE_CAL: // Not implemented by the recompiler nor the interpreter
					break;
				case RSX_FP_OPCODE_IFE:
				{
					const u32 else_slot = src1.else_offset >> 2;
					const u32 endif_slot = src2.end_offset >> 2;

					if (else_slot <= slot || endif_slot < else_slot)
					{
						result.unsupported |= reason_program_layout;
						break;
					}

					targets.push_back(else_slot);
					targets.push_back(endif_slot);

					if (else_slot < endif_slot)
					{
						// The interpreter has one pending ELSE trap; it is also checked before the loop end
						if (then_end != umax || else_slot == loop_end)
						{
							result.inexact |= reason_nested_flow;
						}

						then_end = else_slot;
					}
					break;
				}
				case RSX_FP_OPCODE_LOOP:
				case RSX_FP_OPCODE_REP:
				{
					const u32 end_slot = src2.end_offset >> 2;
					if (end_slot <= slot)
					{
						result.unsupported |= reason_program_layout;
						break;
					}

					targets.push_back(end_slot);

					// One loop state in the interpreter; the recompiler iterates rep_count times whatever the start and step
					if (loop_end != umax || end_slot == then_end)
					{
						result.inexact |= reason_nested_flow;
					}

					if (src1.init_counter != 0 || src1.increment != 1)
					{
						result.inexact |= reason_loop_counter;
					}

					loop_end = end_slot;
					break;
				}
				default:
					result.inexact |= reason_instruction;
					break;
				}

				continue;
			}

			if (!is_interpreted_opcode(opcode))
			{
				result.inexact |= reason_instruction;
			}

			if (is_texture_opcode(opcode))
			{
				result.features |= fs_feature_textures;
				if (dst.exp_tex)
				{
					result.features |= fs_feature_texture_expand;
				}
			}

			// Operands: a literal follows any instruction with a constant source (hardware rule, FPToCFG). The interpreter
			// only skips it when it reads that source: both must agree, or it would execute the literal.
			const u32 operand_count = get_interpreter_operand_count(opcode);
			const u32 sources[3] = { src0.HEX, src1.HEX, src2.HEX };
			const u32 precision_modifiers[3] = { src1.src0_prec_mod, src1.src1_prec_mod, src1.src2_prec_mod };
			bool has_literal = false;
			bool reads_literal = false;

			for (u32 i = 0; i < 3; ++i)
			{
				const SRC_Common source{ .HEX = sources[i] };
				const bool is_constant = source.reg_type == RSX_FP_REGISTER_TYPE_CONSTANT;
				has_literal |= is_constant;

				if (i >= operand_count)
				{
					continue;
				}

				reads_literal |= is_constant;

				if (precision_modifiers[i] >= RSX_FP_PRECISION_HALF && precision_modifiers[i] <= RSX_FP_PRECISION_FIXED9)
				{
					result.features |= fs_feature_precision;
				}

				switch (source.reg_type)
				{
				case RSX_FP_REGISTER_TYPE_TEMP:
					read_register(source.tmp_reg_index, !!source.fp16);
					break;
				case RSX_FP_REGISTER_TYPE_INPUT:
				{
					if (src2.use_index_reg)
					{
						result.inexact |= reason_indexed_input;
					}

					const u32 input = dst.src_attr_reg_num;
					if (input > 14)
					{
						result.unsupported |= reason_input_register;
					}
					else if (input >= 4 && input <= 13)
					{
						result.reads_texcoords = true;
						result.perspective_correction |= !!src2.perspective_corr;
					}
					break;
				}
				case RSX_FP_REGISTER_TYPE_CONSTANT:
					break;
				default:
					// Invalid source type: the recompiler marks the program invalid
					result.inexact |= reason_instruction;
					break;
				}
			}

			if (has_literal != reads_literal)
			{
				result.unsupported |= reason_program_layout;
			}

			if (operand_count && !dst.no_dest)
			{
				write_register(dst.dest_reg, !!dst.fp16);

				if (dst.prec && !dst.saturate)
				{
					result.features |= fs_feature_precision;
				}
			}

			if (has_literal)
			{
				++slot;
			}
		}

		// The output stage reads the exported registers
		const bool fp32_exports = !!(fp.ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS);
		static constexpr u32 exports32[4] = { 0, 2, 3, 4 };
		static constexpr u32 exports16[4] = { 0, 4, 6, 8 };

		for (u32 i = 0; i < std::min(fp.mrt_buffers_count, 4u); ++i)
		{
			read_register(fp32_exports ? exports32[i] : exports16[i], !fp32_exports);
		}

		if (fp.ctrl & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT)
		{
			read_register(1, false);
		}

		// Branch targets must be instructions of the program (forward jumps only: execution always ends)
		for (const u32 target : targets)
		{
			if (last_instruction == umax || target > last_instruction || !is_instruction[target])
			{
				result.unsupported |= reason_program_layout;
			}
		}

		// Registers the interpreter zero-initializes: all it uses (the recompiler initializes every temporary), at
		// least the exported ones (H8 = R4)
		result.register_count = std::min(std::max({ used32, (used16 + 1) / 2, 5u }), register_file_size);

		return m_fragment_analysis[key] = result;
	}

	glsl::program* shader_interpreter::get(
		const mtl::pipeline_props& properties,
		const RSXFragmentProgram& fp,
		const program_hash_util::fragment_program_utils::fragment_program_metadata& fp_metadata,
		u64 fp_ucode_hash,
		const RSXVertexProgram& vp,
		const program_hash_util::vertex_program_utils::vertex_program_metadata& vp_metadata,
		u64 vp_ucode_hash)
	{
		m_skip_reason = skip_reason::none;

		u32 unsupported = 0;
		u32 inexact = analyse_vertex_program(vp, vp_ucode_hash);

		if (vp_metadata.referenced_textures_mask)
		{
			unsupported |= reason_vertex_textures;
		}

		const fragment_analysis& analysis = analyse_fragment_program(fp, fp_ucode_hash);
		unsupported |= analysis.unsupported;
		inexact |= analysis.inexact;

		const u32 ctrl = fp.ctrl;
		u32 features = analysis.features;

		if (ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING)
		{
			// Framebuffer fetch: only recompiled programs, with full-state pipelines (see is_specializable,
			// MTLPipelineCompiler.cpp)
			unsupported |= reason_programmable_blending;
		}

		if (ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE)
		{
			unsupported |= reason_depth_compare;
		}

		if (ctrl & RSX_SHADER_CONTROL_DEPTH_BOUNDS_TEST)
		{
			// Only recompiled programs perform the test (MTLFragmentProgram)
			unsupported |= reason_depth_bounds;
		}

		// Texture units: where each one is bound (MTLShaderInterpreterSource.cpp, _texture_fetch)
		const auto& texture_state = fp.texture_state;
		u32 shadow_2d = 0;
		u32 shadow_cube = 0;
		u32 redirected = 0;

		for (u32 mask = fp_metadata.referenced_textures_mask, unit = 0; mask; mask >>= 1, ++unit)
		{
			if (!(mask & 1))
			{
				continue;
			}

			const u32 bit = 1u << unit;
			const auto dimension = fp.get_texture_dimension(static_cast<u8>(unit));

			if (texture_state.multisampled_textures & bit)
			{
				unsupported |= reason_msaa_textures;
				continue;
			}

			if (texture_state.shadow_textures & bit)
			{
				features |= fs_feature_shadow;

				switch (dimension)
				{
				case rsx::texture_dimension_extended::texture_dimension_2d: shadow_2d |= bit; break;
				case rsx::texture_dimension_extended::texture_dimension_cubemap: shadow_cube |= bit; break;
				case rsx::texture_dimension_extended::texture_dimension_3d: unsupported |= reason_texture_type; break;
				default: break; // 1D: not implemented by the recompiler (reads 0), nothing bound
				}
				continue;
			}

			if (texture_state.redirected_textures & bit)
			{
				if (dimension == rsx::texture_dimension_extended::texture_dimension_cubemap ||
					dimension == rsx::texture_dimension_extended::texture_dimension_3d)
				{
					unsupported |= reason_texture_type;
				}

				redirected |= bit;
				features |= fs_feature_redirect | fs_feature_texture_2d;
				continue;
			}

			switch (dimension)
			{
			case rsx::texture_dimension_extended::texture_dimension_cubemap: features |= fs_feature_texture_cube; break;
			case rsx::texture_dimension_extended::texture_dimension_3d: features |= fs_feature_texture_3d; break;
			default: features |= fs_feature_texture_2d; break;
			}
		}

		if (std::popcount(shadow_2d) > static_cast<int>(shadow_2d_slots) ||
			std::popcount(shadow_cube) > static_cast<int>(shadow_cube_slots) ||
			std::popcount(redirected) > static_cast<int>(stencil_slots))
		{
			unsupported |= reason_texture_slots;
		}

		if (unsupported)
		{
			report_reasons(unsupported, false);
			m_skip_reason = skip_reason::unsupported;
			return nullptr;
		}

		if (inexact)
		{
			report_reasons(inexact, true);
			m_skip_reason = skip_reason::inexact;
			return nullptr;
		}

		// Output stage and texel processing, decided per draw like the recompiler (MTLFragmentDecompilerThread::insertGlobalFunctions)
		u32 effective_ctrl = ctrl;
		if (shader_precision_is_low())
		{
			// No 8-bit output rounding at low shader precision
			effective_ctrl &= ~static_cast<u32>(RSX_SHADER_CONTROL_8BIT_FRAMEBUFFER);
		}

		if (ctrl & (RSX_SHADER_CONTROL_TEXTURE_FORMAT_CONVERT | RSX_SHADER_CONTROL_TEXTURE_ALPHA_KILL)) features |= fs_feature_texture_convert;
		if (ctrl & RSX_SHADER_CONTROL_ALPHA_TEST) features |= fs_feature_alpha_test;
		if (ctrl & RSX_SHADER_CONTROL_ALPHA_TO_COVERAGE) features |= fs_feature_alpha_to_coverage;
		if (ctrl & RSX_SHADER_CONTROL_SRGB_FRAMEBUFFER) features |= fs_feature_srgb_output;
		if (effective_ctrl & RSX_SHADER_CONTROL_8BIT_FRAMEBUFFER) features |= fs_feature_output_rounding;
		if (ctrl & RSX_SHADER_CONTROL_ROP_OUTPUT_REMAP) features |= fs_feature_output_remap;
		if (ctrl & RSX_SHADER_CONTROL_POLYGON_STIPPLE) features |= fs_feature_polygon_stipple;

		if (analysis.reads_texcoords && (fp.texcoord_control_mask || analysis.perspective_correction))
		{
			features |= fs_feature_texcoord_control;
		}

		if (!(features & fs_feature_textures))
		{
			// Texture state of units the program does not sample
			features &= ~(fs_feature_texture_2d | fs_feature_texture_3d | fs_feature_texture_cube | fs_feature_shadow |
				fs_feature_redirect | fs_feature_texture_convert | fs_feature_texture_expand);
		}

		// Library variants
		pipeline_key key{};
		key.vs_library = (vp.ctrl & RSX_SHADER_CONTROL_INSTANCED_CONSTANTS) ? vs_library_instancing : 0;
		key.fs_library = std::min<u32>(properties.state.color_count, 4);

		if (ctrl & (CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT | RSX_SHADER_CONTROL_DISABLE_EARLY_Z))
		{
			key.fs_library |= fs_library_depth_output;
		}

		if (fp.texcoord_control_mask >> 16)
		{
			// Point sprite coordinates (set for point primitives only)
			key.fs_library |= fs_library_point_coord;
		}

		// Metal 4 pipelines do not bake the depth/stencil format; colour writes past the program's outputs are off, as
		// for recompiled programs (MTLTraits::validate_pipeline_properties)
		key.properties = properties;
		key.properties.state.depth_stencil_format = 0;
		for (u32 i = fp.mrt_buffers_count; i < key.properties.state.color.size(); ++i)
		{
			key.properties.state.color[i].write_mask = 0;
		}

		key.fs_features = features;

		m_current_vs_library = key.vs_library;
		m_current_fs_library = key.fs_library;
		m_fs_state.ctrl = effective_ctrl;
		m_fs_state.register_count = analysis.register_count;
		m_fs_state.shadow_2d_mask = shadow_2d;
		m_fs_state.shadow_cube_mask = shadow_cube;

		if (m_current_program && !m_current_is_uber && key == m_current_key)
		{
			return m_current_program;
		}

		pipeline_key uber_key = key;
		uber_key.fs_features = fs_feature_all;

		pipeline_status exact_status = pipeline_status::failed;
		bool exact_known = false;
		pipeline_status uber_status = pipeline_status::failed;
		bool uber_known = false;
		glsl::program* exact_program = nullptr;
		glsl::program* uber_program = nullptr;

		{
			reader_lock lock(m_pipeline_lock);

			if (const auto found = m_pipelines.find(key); found != m_pipelines.end())
			{
				exact_known = true;
				exact_status = found->second.status;
				exact_program = found->second.program.get();
			}

			if (const auto found = m_pipelines.find(uber_key); found != m_pipelines.end())
			{
				uber_known = true;
				uber_status = found->second.status;
				uber_program = found->second.program.get();
			}
		}

		if (exact_known && exact_status == pipeline_status::ready)
		{
			m_current_key = key;
			m_current_program = exact_program;
			m_current_is_uber = (key.fs_features == fs_feature_all);
			return exact_program;
		}

		if (!uber_known)
		{
			// First draw of this render state: its uber pipeline serves every program, build it before other compiles
			// (created right away when it is a specialization of an existing unspecialized pipeline)
			if (glsl::program* program = request_pipeline(uber_key, true))
			{
				m_current_key = uber_key;
				m_current_program = program;
				m_current_is_uber = true;
				return program;
			}

			m_skip_reason = skip_reason::not_ready;
			return nullptr;
		}

		if (uber_status != pipeline_status::ready)
		{
			m_skip_reason = uber_status == pipeline_status::building ? skip_reason::not_ready : skip_reason::failed;
			return nullptr;
		}

		// Specialize in the background (at most max_variant_pipelines per session: they only make the interpreter faster)
		static constexpr u32 max_variant_pipelines = 128;
		if (!exact_known && m_variant_requests < max_variant_pipelines)
		{
			m_variant_requests++;

			if (glsl::program* program = request_pipeline(key, false))
			{
				// Created right away (a specialization of an existing unspecialized pipeline)
				m_current_key = key;
				m_current_program = program;
				m_current_is_uber = false;
				return program;
			}
		}

		m_current_key = uber_key;
		m_current_program = uber_program;
		m_current_is_uber = true;
		return uber_program;
	}

	bool shader_interpreter::is_interpreter(const glsl::program* prog) const
	{
		return prog && prog == m_current_program;
	}

	std::pair<const MTLVertexProgram*, const MTLFragmentProgram*> shader_interpreter::get_shaders() const
	{
		return { m_vs_libraries[m_current_vs_library].get(), m_fs_libraries[m_current_fs_library].get() };
	}

	void shader_interpreter::write_vertex_header(void* dst, const RSXVertexProgram& vp, bool two_sided_lighting) const
	{
		const u32 header[4] = { vp.base_address, vp.entry, vp.output_mask, two_sided_lighting ? 1u : 0u };
		static_assert(sizeof(header) == vertex_header_size);
		std::memcpy(dst, header, sizeof(header));
	}

	void shader_interpreter::write_fragment_header(void* dst, const RSXFragmentProgram& fp, u32 shader_control) const
	{
		// The register count (bits 24-29) covers every register the program uses, so that all of them start at 0
		const u32 register_count = std::max<u32>((shader_control >> 24) & 0x3f, m_fs_state.register_count);

		const u32 header[8] =
		{
			(shader_control & ~(0x3fu << 24)) | (std::min(register_count, 0x3fu) << 24),
			fp.texture_state.texture_dimensions,
			m_fs_state.ctrl,
			fp.mrt_buffers_count,
			fp.texcoord_control_mask,
			fp.texture_state.shadow_textures,
			fp.texture_state.redirected_textures,
			m_fs_state.shadow_2d_mask | (m_fs_state.shadow_cube_mask << 16),
		};

		static_assert(sizeof(header) == fragment_header_size);
		std::memcpy(dst, header, sizeof(header));
	}

	void shader_interpreter::update_fragment_textures(mtl::command_list& cmd, glsl::program& program, const RSXFragmentProgram& fp, const texture_environment& env)
	{
		if (!m_null_stencil_image)
		{
			// Unused stencil mirror slots (utexture2D)
			image_create_info info{};
			info.format = MTL::PixelFormatR8Uint;
			info.width = 4;
			info.height = 4;
			info.usage = MTL::TextureUsageShaderRead;
			info.format_class = RSX_FORMAT_CLASS_COLOR;

			m_null_stencil_image = std::make_unique<mtl::image>(*ensure(g_render_device), info);
			m_null_stencil_image->set_debug_name("Interpreter null stencil texture");
			clear_image(cmd, m_null_stencil_image.get(), {});
		}

		using binding = glsl::image_binding_info;
		const binding null_2d(env.null_views[0], nullptr);
		const binding null_3d(env.null_views[1], nullptr);
		const binding null_cube(env.null_views[2], nullptr);
		const binding null_shadow_2d(env.null_views[3], nullptr);
		const binding null_shadow_cube(env.null_views[4], nullptr);
		const binding null_stencil(m_null_stencil_image->value, nullptr);

		std::array<binding, texture_units> tex2d, tex3d, texcube, samplers;
		std::array<binding, shadow_2d_slots> shadow2d;
		std::array<binding, shadow_cube_slots> shadowcube;
		std::array<binding, stencil_slots> stencil;
		tex2d.fill(null_2d);
		tex3d.fill(null_3d);
		texcube.fill(null_cube);
		shadow2d.fill(null_shadow_2d);
		shadowcube.fill(null_shadow_cube);
		stencil.fill(null_stencil);

		const auto& texture_state = fp.texture_state;
		u32 shadow_2d_slot = 0;
		u32 shadow_cube_slot = 0;
		u32 stencil_slot = 0;

		// Units in ascending order: a compacted slot is the rank of the unit among the units of its kind
		for (u32 unit = 0; unit < texture_units; ++unit)
		{
			const auto& input = env.units[unit];
			samplers[unit] = binding(static_cast<const mtl::image_view*>(nullptr), input.sampler);

			if (!input.view)
			{
				continue;
			}

			const u32 bit = 1u << unit;
			const binding image(input.view, nullptr);

			switch (fp.get_texture_dimension(static_cast<u8>(unit)))
			{
			case rsx::texture_dimension_extended::texture_dimension_cubemap:
				if (texture_state.shadow_textures & bit)
				{
					shadowcube[std::min(shadow_cube_slot++, shadow_cube_slots - 1)] = image;
				}
				else
				{
					texcube[unit] = image;
				}
				break;
			case rsx::texture_dimension_extended::texture_dimension_3d:
				tex3d[unit] = image;
				break;
			default:
				if (texture_state.shadow_textures & bit)
				{
					if (m_fs_state.shadow_2d_mask & bit)
					{
						shadow2d[std::min(shadow_2d_slot++, shadow_2d_slots - 1)] = image;
					}
				}
				else
				{
					tex2d[unit] = image;

					if (texture_state.redirected_textures & bit)
					{
						// The shader's slot is the unit's rank among all redirected units (header): a unit without an image
						// (placeholder view, no stencil view) keeps its slot, which holds the placeholder
						const u32 slot = std::min(stencil_slot++, stencil_slots - 1);
						if (input.stencil_view)
						{
							stencil[slot] = binding(input.stencil_view, nullptr);
						}
					}
				}
				break;
			}
		}

		const u32 set = glsl::binding_set_index_fragment;
		program.bind_uniform_array(tex2d, set, fragment_textures_location + 0);
		program.bind_uniform_array(tex3d, set, fragment_textures_location + 1);
		program.bind_uniform_array(texcube, set, fragment_textures_location + 2);
		program.bind_uniform_array(shadow2d, set, fragment_textures_location + 3);
		program.bind_uniform_array(shadowcube, set, fragment_textures_location + 4);
		program.bind_uniform_array(stencil, set, fragment_textures_location + 5);
		program.bind_uniform_array(samplers, set, fragment_samplers_location);
	}

	shader_interpreter::stats shader_interpreter::get_stats_and_reset()
	{
		return
		{
			.pipelines_built = m_pipelines_built.exchange(0),
			.variants_built = m_variants_built.exchange(0),
			.pipelines_failed = m_pipelines_failed.exchange(0),
		};
	}

	usz shader_interpreter::get_pipeline_count() const
	{
		reader_lock lock(m_pipeline_lock);
		return m_pipelines.size();
	}
}
