#include "stdafx.h"
#include "MTLGraphicsLog.h"

#include "Emu/system_config.h"
#include "util/logs.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>
#include "Emu/RSX/Common/forced_msaa.hpp"

namespace mtl
{
	namespace
	{
		// Counter storage. Order matches gfx_flow; names are printed in the flow report.
		const char* const s_flow_names[] =
		{
			"resolve-color", "resolve-depth", "resolve-depthstencil",
			"unresolve-color", "unresolve-depth", "unresolve-depthstencil",
			"blit-pass", "scaled-copy", "plain-copy",
			"compute-detile", "compute-gather", "compute-shuffle", "compute-fconvert", "compute-deswizzle", "compute-other",
			"upload-image", "dma-readback",
			"surface-read-init", "surface-clear-init", "surface-spill", "surface-unspill",
			"depth-copy", "gather-build", "gather-reuse",
			"present-frame",
		};

		static_assert(std::size(s_flow_names) == static_cast<usz>(gfx_flow::count));

		struct flow_state
		{
			std::array<atomic_t<unsigned long long>, static_cast<usz>(gfx_flow::count)> counters{};
			atomic_t<unsigned long long> dma_readback_bytes{ 0 };
			atomic_t<unsigned long long> frame_serial{ 0 };
		};

		flow_state& state()
		{
			static flow_state* s = new flow_state();
			return *s;
		}
	}

	// --- Flow recording --------------------------------------------------------------------------------

	void note_resolve(const char* kind, bool unresolve, unsigned w, unsigned h)
	{
		gfx_flow flow = gfx_flow::resolve_color;
		if (kind[0] == 'd')
		{
			flow = (kind[5] == 's') ? gfx_flow::resolve_depthstencil : gfx_flow::resolve_depth;
		}
		if (unresolve)
		{
			flow = static_cast<gfx_flow>(static_cast<unsigned>(flow) + 3);
		}

		state().counters[static_cast<usz>(flow)]++;
		rsx_log.trace("Metal flow: %s %s %ux%u (frame %llu)", unresolve ? "unresolve" : "resolve", kind, w, h,
			state().frame_serial.load());
	}

	void note_blit(unsigned dst_aspect, int output_type, int src_w, int src_h, int dst_w, int dst_h)
	{
		state().counters[static_cast<usz>(gfx_flow::blit_pass)]++;
		rsx_log.trace("Metal flow: blit pass aspect 0x%x output %d %dx%d -> %dx%d (frame %llu)", dst_aspect, output_type,
			src_w, src_h, dst_w, dst_h, state().frame_serial.load());
	}

	void note_scaled_copy(int src_w, int src_h, int dst_w, int dst_h)
	{
		state().counters[static_cast<usz>(gfx_flow::scaled_copy)]++;
		rsx_log.trace("Metal flow: scaled copy %dx%d -> %dx%d (frame %llu)", src_w, src_h, dst_w, dst_h,
			state().frame_serial.load());
	}

	void note_plain_copy()
	{
		state().counters[static_cast<usz>(gfx_flow::plain_copy)]++;
	}

	void note_compute(const char* task)
	{
		gfx_flow flow = gfx_flow::compute_other;
		if (!std::strcmp(task, "detile") || !std::strcmp(task, "tile-encode")) flow = gfx_flow::compute_detile;
		else if (!std::strcmp(task, "gather") || !std::strcmp(task, "scatter")) flow = gfx_flow::compute_gather;
		else if (!std::strcmp(task, "shuffle")) flow = gfx_flow::compute_shuffle;
		else if (!std::strcmp(task, "fconvert")) flow = gfx_flow::compute_fconvert;
		else if (!std::strcmp(task, "deswizzle")) flow = gfx_flow::compute_deswizzle;

		state().counters[static_cast<usz>(flow)]++;
		rsx_log.trace("Metal flow: compute %s (frame %llu)", task, state().frame_serial.load());
	}

	void note_upload(unsigned w, unsigned h, unsigned layers)
	{
		state().counters[static_cast<usz>(gfx_flow::upload_image)]++;
		rsx_log.trace("Metal flow: upload %ux%ux%u (frame %llu)", w, h, layers, state().frame_serial.load());
	}

	void note_readback(unsigned long long bytes)
	{
		state().counters[static_cast<usz>(gfx_flow::dma_readback)]++;
		state().dma_readback_bytes += bytes;
		rsx_log.trace("Metal flow: DMA readback %llu bytes to guest memory (frame %llu)", bytes,
			state().frame_serial.load());
	}

	void note_surface_init(bool from_memory)
	{
		state().counters[static_cast<usz>(from_memory ? gfx_flow::surface_read_init : gfx_flow::surface_clear_init)]++;
		rsx_log.trace("Metal flow: surface init (%s) (frame %llu)", from_memory ? "guest memory" : "clear",
			state().frame_serial.load());
	}

	void note_spill(bool unspill)
	{
		state().counters[static_cast<usz>(unspill ? gfx_flow::surface_unspill : gfx_flow::surface_spill)]++;
		rsx_log.trace("Metal flow: surface %s (frame %llu)", unspill ? "unspill" : "spill", state().frame_serial.load());
	}

	void note_depth_copy()
	{
		state().counters[static_cast<usz>(gfx_flow::depth_copy)]++;
		rsx_log.trace("Metal flow: depth copy for shader reads (frame %llu)", state().frame_serial.load());
	}

	void note_gather(bool reused)
	{
		state().counters[static_cast<usz>(reused ? gfx_flow::gather_reuse : gfx_flow::gather_build)]++;
		rsx_log.trace("Metal flow: texture gather %s (frame %llu)", reused ? "reused" : "built",
			state().frame_serial.load());
	}

	void note_present(const char* upscaler)
	{
		state().counters[static_cast<usz>(gfx_flow::present_frame)]++;
		rsx_log.trace("Metal flow: present via %s (frame %llu)", upscaler, state().frame_serial.load());
	}

	unsigned long long graphics_log_next_frame()
	{
		return ++state().frame_serial;
	}

	unsigned long long graphics_log_current_frame()
	{
		return state().frame_serial.load();
	}

	// --- Issue census ----------------------------------------------------------------------------------

	namespace
	{
		struct issue_record
		{
			unsigned long long count = 0;
			unsigned long long reported = 0; // count already included in a report
			unsigned long long first_frame = 0;
			u64 first_time = 0;
		};

		class issue_collector : public logs::listener
		{
			std::mutex m_mutex;
			std::map<std::string, issue_record> m_issues;
			unsigned long long m_dropped_kinds = 0;

			static constexpr usz max_kinds = 128;
			static constexpr usz max_text = 300;

		public:
			issue_collector()
			{
				logs::listener::add(this);
			}

			void log(u64 stamp, const logs::message& msg, std::string_view prefix, std::string_view text) override
			{
				(void)prefix;
				const logs::level sev = msg;

				// Graphics issues only: warnings and worse on the RSX channel. Notice/trace (including this
				// collector's own reports) never land here, so reporting cannot recurse into itself.
				if (sev != logs::level::fatal && sev != logs::level::error &&
					sev != logs::level::warning && sev != logs::level::todo)
				{
					return;
				}

				if (!msg->name || std::string_view(msg->name) != "RSX")
				{
					return;
				}

				std::string key(text.substr(0, max_text));

				std::lock_guard lock(m_mutex);
				auto it = m_issues.find(key);
				if (it == m_issues.end())
				{
					if (m_issues.size() >= max_kinds)
					{
						m_dropped_kinds++;
						return;
					}

					issue_record rec{};
					rec.count = 1;
					rec.first_frame = state().frame_serial.load();
					rec.first_time = stamp;
					m_issues.emplace(std::move(key), rec);
					return;
				}

				it->second.count++;
			}

			// Emit newly seen issues (delta since the last report). Returns true if anything was printed.
			bool report()
			{
				using entry = std::pair<std::string, issue_record>;
				std::vector<entry> fresh;

				unsigned long long total = 0;
				unsigned long long kinds = 0;
				unsigned long long dropped = 0;
				{
					std::lock_guard lock(m_mutex);
					for (auto& [text, rec] : m_issues)
					{
						total += rec.count;
						if (rec.count != rec.reported)
						{
							fresh.emplace_back(text, rec);
							rec.reported = rec.count;
						}
					}

					kinds = m_issues.size();
					dropped = m_dropped_kinds;
				}

				if (fresh.empty())
				{
					return false;
				}

				std::sort(fresh.begin(), fresh.end(), [](const entry& a, const entry& b)
				{
					return a.second.count > b.second.count;
				});

				const std::string dropped_text = dropped ?
					fmt::format(" (%llu further kinds dropped)", dropped) : std::string();

				rsx_log.notice("Metal: graphics issues so far: %llu report(s) in %llu kind(s)%s. Worst since last report:",
					total, kinds, dropped_text.c_str());

				const usz shown = fresh.size() < 8 ? fresh.size() : 8;
				for (usz i = 0; i < shown; i++)
				{
					rsx_log.notice("Metal: issue %llu: [%llux, first frame %llu] %s", i + 1, fresh[i].second.count,
						fresh[i].second.first_frame, fresh[i].first.c_str());
				}

				if (fresh.size() > shown)
				{
					rsx_log.notice("Metal: ... and %llu more issue kind(s) (full text at trace level above, if enabled)",
						static_cast<unsigned long long>(fresh.size() - shown));
				}

				return true;
			}
		};

		issue_collector& collector()
		{
			static issue_collector* s = new issue_collector();
			return *s;
		}
	}

	void graphics_log_install()
	{
		static bool installed = false;
		if (installed)
		{
			return;
		}

		installed = true;
		(void)collector();
		rsx_log.notice("Metal: graphics-flow logging installed (per-kernel trace + issue census)");
	}

	void graphics_log_boot_snapshot()
	{
		const char* msaa = (g_cfg.video.antialiasing_level == msaa_level::none) ? "None" : "Auto";
		const char* scaling = "";
		switch (g_cfg.video.output_scaling.get())
		{
		case output_scaling_mode::nearest: scaling = "nearest"; break;
		case output_scaling_mode::bilinear: scaling = "bilinear"; break;
		case output_scaling_mode::fsr: scaling = "metalfx"; break;
		}

		if (const u8 forced = rsx::forced_msaa_samples(); forced > 1)
		{
			rsx_log.notice("Metal: forced MSAA %ux on every single-sample render target (games' own PS3 MSAA keeps its sample count); output upscaled by %s",
				forced, scaling);
		}

		rsx_log.notice("Metal: graphics config: MSAA %s, force hardware MSAA resolve %s, read color buffers %s, "
			"read depth buffer %s, write depth buffer %s, resolution scale %u%%, output scaling %s, strict rendering %s, "
			"vsync %d. Per-kernel flow lines log at trace level; this summary and the issue census log at notice level.",
			msaa,
			g_cfg.video.force_hw_MSAA_resolve ? "on" : "off",
			g_cfg.video.read_color_buffers ? "on" : "off",
			g_cfg.video.read_depth_buffer ? "on" : "off",
			g_cfg.video.write_depth_buffer ? "on" : "off",
			static_cast<unsigned>(g_cfg.video.resolution_scale_percent.get()),
			scaling,
			g_cfg.video.strict_rendering_mode ? "on" : "off",
			static_cast<int>(g_cfg.video.vsync.get()));
	}

	void graphics_log_report(unsigned long long frames_in_window)
	{
		ensure(frames_in_window != 0);

		std::string flow;
		for (usz i = 0; i < static_cast<usz>(gfx_flow::count); i++)
		{
			const auto count = state().counters[i].exchange(0);
			if (count)
			{
				if (!flow.empty())
				{
					flow += ", ";
				}

				flow += fmt::format("%s %.1f", s_flow_names[i], static_cast<f64>(count) / frames_in_window);
			}
		}

		const auto readback_bytes = state().dma_readback_bytes.exchange(0);
		if (readback_bytes)
		{
			if (!flow.empty())
			{
				flow += ", ";
			}

			flow += fmt::format("dma-readback-bytes %.1f KiB", static_cast<f64>(readback_bytes) / 1024. / frames_in_window);
		}

		rsx_log.notice("Metal: graphics flow per frame%s%s (frame %llu)",
			flow.empty() ? ": no GPU flows outside draw passes" : ": ",
			flow.empty() ? "" : flow.c_str(),
			state().frame_serial.load());

		collector().report();
	}
}
