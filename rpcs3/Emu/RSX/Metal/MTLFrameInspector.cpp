#include "stdafx.h"

#include "MTLFrameInspector.h"
#include "MTLHelpers.h"
#include "mtlutils/device.h"

#include "Emu/Cell/timers.hpp"

#include "Utilities/StrUtil.h"

#include <algorithm>
#include <cstring>

namespace mtl
{
	namespace
	{
		enum warn_kind : u32
		{
			warn_black = 0,
			warn_partly_black,
			warn_bad_math,
			warn_overflow,
		};

		constexpr u64 warn_interval_us = 10'000'000;

		std::string build_inspect_source(bool multisampled)
		{
			std::string src =
				"#version 450\n"
				"layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n"
				"layout(set = 0, binding = 0) uniform %SAMPLER% InputTexture;\n"
				"layout(set = 0, binding = 1, std430) buffer ssbo1 { uint stats[]; };\n"
				"layout(push_constant) uniform ubo { uvec4 params; };\n" // texture width, height
				"\n"
				"#define GRID_W %GW%u\n"
				"#define GRID_H %GH%u\n"
				"#define MAP_W %MW%u\n"
				"#define MAP_H %MH%u\n"
				"#define HEADER %HD%u\n"
				"\n"
				"void main()\n"
				"{\n"
				"	const uvec2 id = gl_GlobalInvocationID.xy;\n"
				"	if (id.x >= GRID_W || id.y >= GRID_H)\n"
				"		return;\n"
				"\n"
				"	const ivec2 size = max(ivec2(params.xy), ivec2(1));\n"
				"	const ivec2 p = clamp(ivec2((vec2(id) + 0.5) * vec2(size) / vec2(GRID_W, GRID_H)), ivec2(0), size - ivec2(1));\n"
				"	const vec4 c = texelFetch(InputTexture, p, 0);\n"
				"\n"
				"	// Exponent all ones = Inf or NaN. Bit test: the shaders are built with fast math, where isnan() may fold away\n"
				"	const uvec4 bits = floatBitsToUint(c);\n"
				"	const bool bad = any(equal(bits & uvec4(0x7F800000u), uvec4(0x7F800000u)));\n"
				"	if (bad) atomicAdd(stats[0], 1u);\n"
				"	if (!bad && any(lessThan(c, vec4(0.0)))) atomicAdd(stats[1], 1u);\n"
				"	if (!bad && any(greaterThan(abs(c), vec4(65000.0)))) atomicAdd(stats[2], 1u);\n"
				"\n"
				"	const float luma = bad ? 0.0 : dot(clamp(c.rgb, vec3(0.0), vec3(1.0)), vec3(0.299, 0.587, 0.114));\n"
				"	if (luma < 0.03) atomicAdd(stats[3], 1u);\n"
				"	const uint l8 = uint(luma * 255.0 + 0.5);\n"
				"	atomicAdd(stats[4], l8);\n"
				"	atomicAdd(stats[5], 1u);\n"
				"	const uint cell = ((id.y * MAP_H) / GRID_H) * MAP_W + (id.x * MAP_W) / GRID_W;\n"
				"	atomicAdd(stats[HEADER + cell], l8);\n"
				"}\n";

			const std::pair<std::string_view, std::string> replacements[] =
			{
				{ "%SAMPLER%", multisampled ? "sampler2DMS" : "sampler2D" },
				{ "%GW%", std::to_string(frame_inspector::grid_w) },
				{ "%GH%", std::to_string(frame_inspector::grid_h) },
				{ "%MW%", std::to_string(frame_inspector::map_w) },
				{ "%MH%", std::to_string(frame_inspector::map_h) },
				{ "%HD%", std::to_string(frame_inspector::header_words) },
			};

			return fmt::replace_all(src, replacements);
		}

		const char* format_name(MTL::PixelFormat format)
		{
			switch (format)
			{
			case MTL::PixelFormatRGBA16Float: return "RGBA16F";
			case MTL::PixelFormatRGBA32Float: return "RGBA32F";
			case MTL::PixelFormatR32Float: return "R32F";
			case MTL::PixelFormatRG16Float: return "RG16F";
			case MTL::PixelFormatR16Float: return "R16F";
			case MTL::PixelFormatRGBA8Unorm: return "RGBA8";
			case MTL::PixelFormatBGRA8Unorm: return "BGRA8";
			case MTL::PixelFormatRGBA8Unorm_sRGB: return "RGBA8 sRGB";
			case MTL::PixelFormatBGRA8Unorm_sRGB: return "BGRA8 sRGB";
			case MTL::PixelFormatRGB10A2Unorm: return "RGB10A2";
			case MTL::PixelFormatB5G6R5Unorm: return "RGB565";
			default: return "other";
			}
		}
	}

	struct frame_inspector::cs_inspect_task : compute_task
	{
		const bool m_multisampled;
		MTL::Texture* m_input = nullptr;
		const mtl::buffer* m_output = nullptr;
		u32 m_output_offset = 0;
		std::array<u32, 4> m_params{};
		std::unique_ptr<mtl::sampler> m_sampler;
		bool m_build_failed = false;

		explicit cs_inspect_task(bool multisampled)
			: m_multisampled(multisampled)
		{
			m_src = build_inspect_source(multisampled);
			ssbo_count = 0; // Declared in get_inputs (binding 1, after the texture)
			use_push_constants = true;
			push_constants_size = 16;
			workgroup_size_x = 16;
			workgroup_size_y = 16;
			create();
		}

		std::vector<glsl::program_input> get_inputs() override
		{
			std::vector<glsl::program_input> result;

			result.push_back(glsl::program_input::make(
				::glsl::program_domain::glsl_compute_program,
				"InputTexture",
				glsl::input_type_texture,
				glsl::binding_set_index_compute,
				0));

			result.push_back(glsl::program_input::make(
				::glsl::program_domain::glsl_compute_program,
				"ssbo1",
				glsl::program_input_type::input_type_storage_buffer,
				glsl::binding_set_index_compute,
				1));

			result.push_back(glsl::program_input::make(
				::glsl::program_domain::glsl_compute_program,
				"push_constants",
				glsl::program_input_type::input_type_push_constant,
				glsl::binding_set_index_compute,
				umax,
				glsl::push_constant_ref{ .offset = 0, .size = push_constants_size }));

			return result;
		}

		// Builds the pipeline once. Telemetry must never take the renderer down: a failure disables the checks.
		bool prepare()
		{
			if (m_program)
			{
				return true;
			}

			if (m_build_failed)
			{
				return false;
			}

			m_program = glsl::create_compute_program(m_src, get_inputs());

			if (!m_program || m_program->max_total_threads_per_threadgroup() < workgroup_size_x * workgroup_size_y)
			{
				rsx_log.error("Metal: graphics self-check kernel (%s) could not be built; the graphics check is off.", m_multisampled ? "multisampled" : "single-sample");
				m_program.reset();
				m_build_failed = true;
				return false;
			}

			return true;
		}

		void bind_resources(mtl::command_list& /*cmd*/) override
		{
			if (!m_sampler)
			{
				sampler_create_info info{};
				info.min_filter = MTL::SamplerMinMagFilterNearest;
				info.mag_filter = MTL::SamplerMinMagFilterNearest;
				m_sampler = std::make_unique<mtl::sampler>(*g_render_device, info);
			}

			push_constants(0, push_constants_size, m_params.data());
			m_program->bind_uniform(glsl::image_binding_info(m_input, m_sampler->value), glsl::binding_set_index_compute, 0);
			m_program->bind_uniform(glsl::buffer_binding_info(m_output, m_output_offset, slot_words * 4), glsl::binding_set_index_compute, 1);
		}

		void dispatch(mtl::command_list& cmd, MTL::Texture* input, const mtl::buffer* output, u32 output_offset)
		{
			m_input = input;
			m_output = output;
			m_output_offset = output_offset;
			m_params = { static_cast<u32>(input->width()), static_cast<u32>(input->height()), 0, 0 };

			compute_task::run(cmd, grid_w / 16, (grid_h + 15) / 16, 1);
		}
	};

	frame_inspector::frame_inspector() = default;

	// Destroyed with the renderer, after the GPU has gone idle
	frame_inspector::~frame_inspector() = default;

	bool frame_inspector::is_float_format(MTL::PixelFormat format)
	{
		switch (format)
		{
		case MTL::PixelFormatRGBA16Float:
		case MTL::PixelFormatRGBA32Float:
		case MTL::PixelFormatR32Float:
		case MTL::PixelFormatRG16Float:
		case MTL::PixelFormatR16Float:
			return true;
		default:
			return false;
		}
	}

	bool frame_inspector::wants_targets() const
	{
		return !m_disabled && ((m_flip_count + 1) % target_interval) == 0;
	}

	bool frame_inspector::record(mtl::command_list& cmd, MTL::Texture* texture, const job_info& info, u64 draw_calls)
	{
		if (!texture || texture->width() < 16 || texture->height() < 16 || m_free_slots.empty())
		{
			return false;
		}

		const bool multisampled = texture->sampleCount() > 1;
		auto& task = multisampled ? m_task_ms : m_task_2d;

		if (!task)
		{
			task = std::make_unique<cs_inspect_task>(multisampled);
		}

		if (!task->prepare())
		{
			m_disabled = true;
			return false;
		}

		const u32 slot = m_free_slots.back();
		m_free_slots.pop_back();

		// The slot was read back (or never used): the GPU is done with it
		std::memset(m_results->map(u64{slot} * slot_words * 4), 0, slot_words * 4);

		task->dispatch(cmd, texture, m_results.get(), slot * slot_words * 4);

		m_pending.push_back({ .slot = slot, .frame_id = get_current_frame_id(), .frame_serial = m_frame_serial, .draw_calls = draw_calls, .info = info });
		return true;
	}

	void frame_inspector::on_flip(mtl::command_list& cmd, MTL::Texture* presented, u32 present_address, u64 draw_calls, const std::vector<target_candidate>& targets)
	{
		m_flip_count++;
		m_frame_serial++;

		if (m_disabled)
		{
			return;
		}

		const bool check_present = (m_flip_count % present_interval) == 0;
		const bool check_targets = (m_flip_count % target_interval) == 0;

		if (!check_present && !check_targets)
		{
			return;
		}

		if (!m_results)
		{
			m_results = std::make_unique<mtl::buffer>(*g_render_device, u64{slot_count} * slot_words * 4, memory_location::host_visible, "graphics self-check results");
			m_free_slots.reserve(slot_count);
			for (u32 i = slot_count; i > 0; i--)
			{
				m_free_slots.push_back(i - 1);
			}
		}

		if (check_present && presented)
		{
			record(cmd, presented, { .kind = job_kind::presented, .address = present_address, .width = static_cast<u32>(presented->width()),
				.height = static_cast<u32>(presented->height()), .format = presented->pixelFormat(), .samples = static_cast<u8>(presented->sampleCount()) }, draw_calls);
		}

		if (check_targets)
		{
			u32 checked = 0;
			for (const auto& target : targets)
			{
				if (checked >= max_targets_per_check || m_disabled)
				{
					break;
				}

				if (!target.texture || !is_float_format(target.texture->pixelFormat()))
				{
					continue;
				}

				checked += record(cmd, target.texture, { .kind = job_kind::render_target, .address = target.address, .width = static_cast<u32>(target.texture->width()),
					.height = static_cast<u32>(target.texture->height()), .format = target.texture->pixelFormat(), .samples = static_cast<u8>(target.texture->sampleCount()) }, draw_calls) ? 1 : 0;
			}
		}
	}

	void frame_inspector::poll()
	{
		const u64 completed = get_last_completed_frame_id();

		while (!m_pending.empty() && m_pending.front().frame_id < completed)
		{
			const pending_job job = m_pending.front();
			m_pending.pop_front();

			const u32* stats = static_cast<const u32*>(m_results->map(u64{job.slot} * slot_words * 4));

			if (stats[5] > 0)
			{
				if (job.info.kind == job_kind::presented)
				{
					analyse_presented(job, stats);
				}
				else
				{
					analyse_target(job, stats);
				}
			}

			m_free_slots.push_back(job.slot);
		}
	}

	bool frame_inspector::warn_due(u32 kind)
	{
		const u64 now = get_system_time();
		if (now - m_last_warn_us[kind] < warn_interval_us)
		{
			return false;
		}

		m_last_warn_us[kind] = now;
		m_total_findings++;
		return true;
	}

	std::string frame_inspector::describe(const job_info& info)
	{
		return fmt::format("0x%x, %s %ux%u%s", info.address, format_name(info.format), info.width, info.height,
			info.samples > 1 ? fmt::format(", %ux MSAA", info.samples).c_str() : "");
	}

	std::string frame_inspector::format_map(const u32* cells)
	{
		// 10 brightness steps, darkest first. One row per line, framed so leading spaces survive in the log.
		static constexpr char ramp[] = " .:-=+*#%@";

		std::string result;
		result.reserve((map_w + 4) * map_h);

		for (u32 y = 0; y < map_h; y++)
		{
			result += "\n    |";
			for (u32 x = 0; x < map_w; x++)
			{
				const u32 mean = cells[y * map_w + x] / samples_per_cell;
				result += ramp[std::min<u32>(9, (mean * 10) / 256)];
			}
			result += '|';
		}

		return result;
	}

	void frame_inspector::analyse_presented(const pending_job& job, const u32* stats)
	{
		const u32 samples = stats[5];
		const u32* cells = stats + header_words;
		const f64 black_fraction = static_cast<f64>(stats[3]) / samples;

		m_window.presented++;

		std::copy_n(cells, m_last_map.size(), m_last_map.begin());
		m_last_map_frame = job.frame_serial;

		// Cells: black = mean luma below 3%, lit = above 15%
		constexpr u32 black_mean = 8;
		constexpr u32 lit_mean = 40;

		u32 newly_black = 0;
		bool unchanged = m_have_prev;

		for (u32 i = 0; i < map_w * map_h; i++)
		{
			const u32 mean = cells[i] / samples_per_cell;
			const bool black = mean < black_mean;

			if (m_have_prev)
			{
				newly_black += (black && (m_prev_map[i] / samples_per_cell) > lit_mean) ? 1 : 0;
				unchanged &= (cells[i] == m_prev_map[i]);
			}
		}

		const u32 cell_count = map_w * map_h;
		const bool rendering = job.draw_calls > 50;

		// Whole screen black while the game draws for real (loading screens and fades draw little: not counted)
		if (black_fraction >= 0.995 && rendering)
		{
			m_window.black++;
			m_black_streak++;

			if (m_black_streak == 3 && warn_due(warn_black))
			{
				rsx_log.warning("Graphics check: black screen for %u checks in a row (~%u frames) while the game drew %llu draws per frame "
					"(display buffer %s). The game renders, but nothing reaches the image it shows.",
					m_black_streak, m_black_streak * present_interval, job.draw_calls, describe(job.info));
			}
		}
		else
		{
			m_black_streak = 0;
		}

		// Parts of the screen turned black: a block of the image went from lit to black, the rest still shows
		if (rendering && black_fraction < 0.95 && newly_black >= cell_count / 10)
		{
			m_window.partly_black++;

			if (warn_due(warn_partly_black))
			{
				rsx_log.warning("Graphics check: parts of the screen turned black: %u of %u map cells went from lit to black since the previous check "
					"(%.0f%% of the image is black now, %llu draws in the frame, display buffer %s). Screen now (32x18 brightness map, ' ' = black, '@' = white):%s",
					newly_black, cell_count, black_fraction * 100., job.draw_calls, describe(job.info), format_map(cells));
			}
		}

		if (unchanged && rendering)
		{
			m_window.unchanged++;
			m_unchanged_streak++;
		}
		else
		{
			m_unchanged_streak = 0;
		}

		std::copy_n(cells, m_prev_map.size(), m_prev_map.begin());
		m_have_prev = true;
	}

	void frame_inspector::analyse_target(const pending_job& job, const u32* stats)
	{
		const u32 samples = stats[5];
		const f64 bad = static_cast<f64>(stats[0]) / samples;
		const f64 overflow = static_cast<f64>(stats[2]) / samples;

		m_window.targets_checked++;

		if (stats[0])
		{
			m_window.targets_bad_math++;

			if (bad > m_window.worst_bad_math)
			{
				m_window.worst_bad_math = bad;
				m_window.worst_bad_math_info = job.info;
			}

			if (warn_due(warn_bad_math))
			{
				rsx_log.warning("Graphics check: GPU math produced NaN/Inf in render target %s: %u of %u samples (%.2f%%). "
					"NaN/Inf come from invalid shader math (0/0, sqrt or log of a negative, Inf*0, values beyond the format) and spread "
					"through later passes (blur, bloom, lighting) as black or white blocks. Brightness map of the target:%s",
					describe(job.info), stats[0], samples, bad * 100., format_map(stats + header_words));
			}
		}

		if (stats[2] && job.info.format != MTL::PixelFormatRGBA32Float && job.info.format != MTL::PixelFormatR32Float)
		{
			m_window.targets_overflow++;

			if (overflow > m_window.worst_overflow)
			{
				m_window.worst_overflow = overflow;
				m_window.worst_overflow_info = job.info;
			}

			if (warn_due(warn_overflow))
			{
				rsx_log.warning("Graphics check: fp16 overflow in render target %s: %u of %u samples (%.2f%%) hold values beyond 65000 "
					"(the next blend or multiply turns them into Inf).",
					describe(job.info), stats[2], samples, overflow * 100.);
			}
		}
	}

	void frame_inspector::report()
	{
		if (m_disabled)
		{
			return;
		}

		const auto& w = m_window;
		std::string verdict;

		const auto add = [&verdict](const std::string& text)
		{
			verdict += verdict.empty() ? text : "; " + text;
		};

		if (w.black)
		{
			add(fmt::format("black screen in %u of %u checks", w.black, w.presented));
		}

		if (w.partly_black)
		{
			add(fmt::format("parts of the screen turned black %u time(s)", w.partly_black));
		}

		if (w.targets_bad_math)
		{
			add(fmt::format("NaN/Inf in %u of %u render target checks (worst %.2f%% in %s)", w.targets_bad_math, w.targets_checked,
				w.worst_bad_math * 100., describe(w.worst_bad_math_info)));
		}

		if (w.targets_overflow)
		{
			add(fmt::format("fp16 overflow in %u of %u render target checks (worst %.2f%% in %s)", w.targets_overflow, w.targets_checked,
				w.worst_overflow * 100., describe(w.worst_overflow_info)));
		}

		rsx_log.notice("Metal: graphics check over 30s: %u presented frames and %u float render targets checked, image unchanged in %u checks while the game drew. %s",
			w.presented, w.targets_checked, w.unchanged, verdict.empty() ? "Verdict: no problems found" : ("Verdict: SUSPECT: " + verdict).c_str());

		if (w.presented)
		{
			rsx_log.notice("Metal: screen at frame %llu (32x18 brightness map, ' ' = black, '@' = white):%s", m_last_map_frame, format_map(m_last_map.data()));
		}

		m_window = {};
	}
}
