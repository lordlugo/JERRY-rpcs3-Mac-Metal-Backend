#pragma once

// Graphics self-check (telemetry). The renderer's other statistics count what the backend did; this checks what the GPU
// actually produced, so the log says when a frame looks broken instead of leaving it to a screenshot:
//  - every 15th presented frame: a 128x72 sample grid of the image about to be shown (luma, black area, a 32x18 luma map
//    that is printed as text with every finding, so the log carries a picture of the screen);
//  - every 30th presented frame: the floating point render targets drawn to since the last check (RGBA16F/RGBA32F/R32F,
//    where shader math results are stored unclamped), checked for NaN/Inf and fp16 overflow.
// The checks are small compute dispatches (9216 texel reads each) recorded with the frame, read back once the GPU has
// finished it: nothing waits for them. Nothing they find changes rendering.
//
// Findings (rate limited, "Graphics check:" warnings with the screen map):
//  - black screen while the game draws (>= 3 checks in a row, > 50 draws per frame);
//  - parts of the screen turned black (>= 10% of the map went from lit to black between two checks, not the whole screen);
//  - NaN/Inf or fp16 overflow in a render target (invalid shader math: 0/0, sqrt/log of negatives, inf*0, precision).
// A 30 s summary line ("graphics check over 30s") gives the counts and a verdict, plus a screen map.

#include "MTLCompute.h"
#include "mtlutils/buffer_object.h"
#include "mtlutils/sampler.h"

#include <array>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace mtl
{
	class frame_inspector
	{
	public:
		static constexpr u32 grid_w = 128;
		static constexpr u32 grid_h = 72;
		static constexpr u32 map_w = 32;
		static constexpr u32 map_h = 18;
		static constexpr u32 samples_per_cell = (grid_w / map_w) * (grid_h / map_h);
		static constexpr u32 header_words = 8;
		static constexpr u32 slot_words = header_words + map_w * map_h;
		static constexpr u32 slot_count = 64;

		static constexpr u32 present_interval = 15;  // flips between checks of the presented image
		static constexpr u32 target_interval = 30;   // flips between checks of the float render targets
		static constexpr u32 max_targets_per_check = 4;

		enum class job_kind : u8
		{
			presented,
			render_target
		};

		struct job_info
		{
			job_kind kind = job_kind::presented;
			u32 address = 0;
			u32 width = 0;
			u32 height = 0;
			MTL::PixelFormat format = MTL::PixelFormatInvalid;
			u8 samples = 1;
		};

		struct target_candidate
		{
			MTL::Texture* texture = nullptr;
			u32 address = 0;
		};

		frame_inspector();
		~frame_inspector();

		frame_inspector(const frame_inspector&) = delete;
		frame_inspector& operator=(const frame_inspector&) = delete;

		// Called by flip() once per presented frame, before the frame's command list is submitted.
		bool wants_targets() const;
		void on_flip(mtl::command_list& cmd, MTL::Texture* presented, u32 present_address, u64 draw_calls, const std::vector<target_candidate>& targets);

		// Analyses the checks the GPU has finished (logs findings). Called every flip.
		void poll();

		// 30 s summary (with the other presentation statistics)
		void report();

		static bool is_float_format(MTL::PixelFormat format);

	private:
		struct cs_inspect_task;

		struct pending_job
		{
			u32 slot = 0;
			u64 frame_id = 0;
			u64 frame_serial = 0;
			u64 draw_calls = 0;
			job_info info{};
		};

		struct window_stats
		{
			u32 presented = 0;
			u32 black = 0;
			u32 partly_black = 0;
			u32 unchanged = 0;
			u32 targets_checked = 0;
			u32 targets_bad_math = 0;
			u32 targets_overflow = 0;
			f64 worst_bad_math = 0.;
			job_info worst_bad_math_info{};
			f64 worst_overflow = 0.;
			job_info worst_overflow_info{};
		};

		bool record(mtl::command_list& cmd, MTL::Texture* texture, const job_info& info, u64 draw_calls);
		void analyse_presented(const pending_job& job, const u32* stats);
		void analyse_target(const pending_job& job, const u32* stats);
		bool warn_due(u32 kind);

		static std::string describe(const job_info& info);
		static std::string format_map(const u32* cells);

		std::unique_ptr<cs_inspect_task> m_task_2d;
		std::unique_ptr<cs_inspect_task> m_task_ms;
		std::unique_ptr<mtl::buffer> m_results;
		std::vector<u32> m_free_slots;
		std::deque<pending_job> m_pending;
		bool m_disabled = false;

		u64 m_flip_count = 0;
		u64 m_frame_serial = 0;

		// Presented image history
		std::array<u32, map_w * map_h> m_prev_map{};
		std::array<u32, map_w * map_h> m_last_map{};
		bool m_have_prev = false;
		u64 m_last_map_frame = 0;
		u32 m_black_streak = 0;
		u32 m_unchanged_streak = 0;

		window_stats m_window{};
		u64 m_total_findings = 0;
		std::array<u64, 4> m_last_warn_us{};
	};
}
