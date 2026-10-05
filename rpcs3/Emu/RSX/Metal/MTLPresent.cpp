#include "stdafx.h"
#include "MTLGraphicsLog.h"
#include "MTLGSRender.h"
#include "MTLCommandStream.h"
#include "MTLFormats.h"
#include "MTLRenderPass.h"
#include "mtlutils/buffer_object.h"
#include "mtlutils/metal_layer.h"

#include "Emu/RSX/Overlays/overlay_manager.h"
#include "Emu/RSX/Overlays/overlay_debug_overlay.h"
#include "Emu/Cell/Modules/cellVideoOut.h"
#include "Emu/RSX/Common/guest_frame_limit.hpp"

#include "util/asm.hpp"
#include "util/video_provider.h"

#include <algorithm>
#include <cmath>
#include <cstring>

extern atomic_t<bool> g_user_asked_for_screenshot;
extern atomic_t<recording_mode> g_recording_mode;

// Presentation model
// ------------------
// Metal 4 requires the queue that renders into a drawable to wait for it (waitForDrawable). That wait blocks every
// later command buffer of the queue, up to a display refresh. So the renderer keeps the drawable off the main queue:
//  1. flip() records the present passes (letterbox clear, calibration, upscaling, overlays) into the frame's last
//     main-queue list, rendering into the frame context's present_image (same size/format as the drawable).
//  2. present() records a present list (one texture copy) on the present queue: wait for the RSX timeline value of the
//     frame's work, wait for the drawable, commit, signal the drawable, then present with pacing (present_drawable()).
// The present list only touches the frame context's present_image and the drawable, both owned by the frame context,
// so the event-id GC (which assumes a single in-order queue) never frees anything it uses. The frame context (and
// its heap snapshot) is recycled once the present list has completed: frame_context_t::swap_command_buffer points to
// it, and its completion implies the completion of the frame's main-queue work.
//
// Pacing (present_drawable()): with R = the screen's minimum refresh interval and G = the guest frame interval
// (get_guest_frame_interval()), every frame stays on screen for k = max(1, ~G / R) refreshes
// (presentAfterMinimumDuration(k * R - R / 2)), e.g. 60 fps on a 120 Hz ProMotion panel = every other refresh, 30 fps =
// every 4th, instead of jittering between 1, 2 and 3 refreshes. Fullscreen on an Adaptive-Sync screen:
// presentAfterMinimumDuration(G - 0.5 ms). VSync off: plain present().
//
// Pacing must never make the game slower than it can run: it is a feedback loop (frames held on screen longer delay
// the flips, and the guest may wait for them in ways the renderer cannot see, e.g. game logic that picks its own
// swap interval from flip timings). Hence:
//  - G is a low estimate (lower quartile) of the intervals measured without the time the RSX thread waited for the
//    display, only ever snapped down to whole vblanks, and never below the frame limit;
//  - the RSX thread keeps serving the guest (memory flushes, labels held back for zcull reports) while it waits for a
//    frame context, so that the wait does not propagate to the guest;
//  - while G paces the game slower than its frame limit, frames are paced at the frame limit for a moment now and then
//    (update_pacing_probe). If the game keeps up, the history measured under the slower pacing is dropped;
//  - the history is dropped when the presentation mode changes (fullscreen/windowed, refresh rate, VSync mode).

namespace
{
	constexpr u64 guest_interval_reset_us = 250'000;         // Longer gaps (loading, pause) restart the guest interval history
	constexpr u64 surface_poll_interval_us = 1'000'000;      // The window can move to another screen without a resize
	constexpr u64 present_stats_interval_us = 30'000'000;    // Telemetry log rate limit
	constexpr u64 display_backlog_threshold_us = 1'000;      // A flip that waited this long for the display: frames are queued

	// Downward probes (MTLGSRender::update_pacing_probe)
	constexpr u64 probe_initial_backoff_us = 2'000'000;      // First probe after the pace dropped below the frame limit
	constexpr u64 probe_max_backoff_us = 32'000'000;         // Probes that find nothing double the delay up to this
	constexpr u64 probe_min_duration_us = 300'000;           // Time for queued frames to drain and for the game to react
	constexpr u64 probe_max_duration_us = 1'000'000;
	constexpr u32 probe_min_samples = 8;

	// Lower quartile of the `count` most recent entries of a ring buffer whose next write position is `next`.
	// Overestimating the guest frame interval throttles the game (every frame is held on screen too long) while
	// underestimating it only degrades to plain FIFO, so low values win: hitches never raise it, and a game alternating
	// between two cadences is paced at the faster one.
	template <usz N>
	f64 recent_lower_quartile(const std::array<f64, N>& ring, u32 next, u32 count)
	{
		std::array<f64, N> values{};
		for (u32 i = 0; i < count; ++i)
		{
			values[i] = ring[(next + N - 1 - i) % N];
		}

		const auto nth = values.begin() + count / 4;
		std::nth_element(values.begin(), nth, values.begin() + count);
		return *nth;
	}

	// Guest frames follow the emulated vblank (1/60 s by default): an interval somewhat above a whole number of vblanks
	// (60, 30, 20, 15 fps) is snapped down to it, so that jitter does not move the pacing target. Never snapped up by more
	// than timer jitter: rounding a game that runs between two cadences up would throttle it (30.5 ms -> 33.3 ms made
	// it 4 refreshes per frame instead of 3 on a 120 Hz screen).
	f64 snap_to_vblanks(f64 interval)
	{
		const f64 vblank = 1. / static_cast<f64>(std::max<s64>(1, g_cfg.video.vblank_rate));
		const f64 vblanks = std::floor(interval / vblank + 0.06);

		if (vblanks >= 1. && vblanks <= 4. && interval - vblanks * vblank <= vblank * 0.2)
		{
			return vblanks * vblank;
		}

		return interval;
	}

	MTL::PixelFormat RSX_display_format_to_mtl_format(u8 format)
	{
		switch (format)
		{
		default:
			rsx_log.error("Unhandled video output format 0x%x", static_cast<s32>(format));
			[[fallthrough]];
		case CELL_VIDEO_OUT_BUFFER_COLOR_FORMAT_X8R8G8B8:
			return MTL::PixelFormatBGRA8Unorm;
		case CELL_VIDEO_OUT_BUFFER_COLOR_FORMAT_X8B8G8R8:
			return MTL::PixelFormatRGBA8Unorm;
		case CELL_VIDEO_OUT_BUFFER_COLOR_FORMAT_R16G16B16X16_FLOAT:
			return MTL::PixelFormatRGBA16Float;
		}
	}
}

void MTLGSRender::assert_metal_layer_state()
{
	if (!m_metal_layer)
	{
		return;
	}

	// The present list copies the frame's present image into the drawable (compute encoder copy), so drawables must be
	// copy destinations: framebuffer-only drawables can only be render pass attachments.
	const bool framebuffer_only = false;
	if (framebuffer_only != m_layer_framebuffer_only || framebuffer_only != m_metal_layer->framebufferOnly())
	{
		m_metal_layer->setFramebufferOnly(framebuffer_only);
		m_layer_framebuffer_only = framebuffer_only;
	}

	// VSync off: no display sync (tears in direct-to-display fullscreen; the compositor still syncs windows).
	// Adaptive/full: display synced and paced by present_drawable().
	const bool display_sync = g_cfg.video.vsync != vsync_mode::off;
	if (display_sync != m_metal_layer->displaySyncEnabled())
	{
		m_metal_layer->setDisplaySyncEnabled(display_sync);
	}

	if (m_vsync_mode != g_cfg.video.vsync)
	{
		// Intervals measured under the other presentation mode include its back-pressure
		m_present_pacing.reset_guest_intervals();
	}
	m_vsync_mode = g_cfg.video.vsync;

	if (m_metal_layer->maximumDrawableCount() != MTL_MAX_DRAWABLE_COUNT)
	{
		m_metal_layer->setMaximumDrawableCount(MTL_MAX_DRAWABLE_COUNT);
	}

	// Only touch drawableSize when it really changes: every change replaces the layer's drawables and the next
	// nextDrawable() can stall up to its 1 s timeout (skipped frame, visible as a hitch or flicker). Qt keeps
	// drawableSize at bounds * contentsScale itself, which is the same value as the client size used here
	// (QWindow size * devicePixelRatio), so normally there is nothing to do.
	if (m_swapchain_dims.width && m_swapchain_dims.height)
	{
		const CGSize current = m_metal_layer->drawableSize();
		const CGSize wanted = { static_cast<CGFloat>(m_swapchain_dims.width), static_cast<CGFloat>(m_swapchain_dims.height) };

		if (current.width != wanted.width || current.height != wanted.height)
		{
			m_metal_layer->setDrawableSize(wanted);
		}
	}
}

void MTLGSRender::configure_metal_layer()
{
	if (!m_metal_layer)
	{
		return;
	}

	assert_metal_layer_state();

	// contentsScale (the window's backing scale factor, the same source as the client size) and opaque are applied on
	// the main thread, together with sampling the screen's refresh properties. Never blocks.
	mtl::request_surface_update(m_view, m_metal_layer);
	m_present_pacing.surface_request_time = get_system_time();
}

bool MTLGSRender::reinitialize_swapchain()
{
	m_swapchain_dims.width = m_frame->client_width();
	m_swapchain_dims.height = m_frame->client_height();

	// Reject requests to resize the surface if the window is minimized
	if (!m_metal_layer || m_swapchain_dims.width == 0 || m_swapchain_dims.height == 0)
	{
		swapchain_unavailable = true;
		return false;
	}

	// Unlike a Vulkan swapchain, a CAMetalLayer is never out of date: in-flight drawables stay valid and new ones are
	// created with the new drawableSize. No GPU sync is required. Frame contexts recreate their present image when the
	// drawable size changes, and the upscaler handles output size changes itself.
	configure_metal_layer();

	swapchain_unavailable = false;
	should_reinitialize_swapchain = false;
	return true;
}

void MTLGSRender::present_feedback_t::on_presented(f64 presented_time)
{
	if (presented_time <= 0.)
	{
		// Never displayed (replaced by a newer drawable before it reached the screen)
		dropped++;
		return;
	}

	const f64 previous = last_presented_time.exchange(presented_time);
	const f64 unit = refresh_interval.load();

	if (previous <= 0. || presented_time <= previous || unit <= 0.)
	{
		return;
	}

	// On-screen time of the previous frame, in refreshes
	const s64 refreshes = std::clamp<s64>(std::llround((presented_time - previous) / unit), 1, 6);
	intervals[refreshes - 1]++;
}

void MTLGSRender::update_present_pacing(bool emu_flip)
{
	const u64 now = get_system_time();
	auto& pacing = m_present_pacing;

	// Screen properties are sampled on the main thread. Poll: moving the window to another screen is not a resize.
	if (now - pacing.surface_request_time >= surface_poll_interval_us)
	{
		mtl::request_surface_update(m_view, m_metal_layer);
		pacing.surface_request_time = now;
	}

	if (const mtl::surface_properties props = mtl::get_surface_properties(); props.serial != pacing.surface_serial)
	{
		const bool first_update = !pacing.surface_serial;
		pacing.surface_serial = props.serial;

		// NSScreen.minimumRefreshInterval is the fastest refresh (1/120 s on ProMotion). Fall back to 60 Hz.
		const f64 refresh_interval = (props.min_refresh_interval >= 0.001 && props.min_refresh_interval <= 0.1)
			? props.min_refresh_interval
			: 1. / 60.;

		// Adaptive-Sync/ProMotion: minimum and maximum refresh intervals differ (WWDC21 "Optimize for variable refresh rate displays")
		const bool variable_refresh = props.max_refresh_interval > refresh_interval * 1.01;

		if (first_update || refresh_interval != pacing.refresh_interval || variable_refresh != pacing.variable_refresh || props.fullscreen != pacing.fullscreen)
		{
			rsx_log.notice("Metal: display refresh %.2f Hz (%s refresh, interval %.2f-%.2f ms, granularity %.2f ms), %s window, backing scale %.1f",
				1. / refresh_interval, variable_refresh ? "variable" : "fixed", props.min_refresh_interval * 1000., props.max_refresh_interval * 1000.,
				props.update_granularity * 1000., props.fullscreen ? "fullscreen" : "windowed", props.backing_scale);

			// A display transition (fullscreen toggle, other screen, mode change) may have disturbed the layer
			// behind the renderer's back. Reassert the presentation state while sizes are stable, when no resize
			// would do it.
			assert_metal_layer_state();

			// The intervals measured so far include the back-pressure of the previous presentation mode (Adaptive-Sync
			// fullscreen or composited window, other refresh rate), and the transition itself stalls the display:
			// start over at the frame limit
			pacing.reset_guest_intervals();
		}

		pacing.refresh_interval = refresh_interval;
		pacing.variable_refresh = variable_refresh;
		pacing.fullscreen = props.fullscreen;
	}

	if (!emu_flip)
	{
		// Overlay-only flips (native UI while the game is paused, ...) say nothing about the guest's frame rate
		return;
	}

	if (pacing.last_emu_flip_time)
	{
		const u64 delta = now - pacing.last_emu_flip_time;

		if (delta >= guest_interval_reset_us)
		{
			// Loading, pause or a stall: start over
			pacing.reset_guest_intervals();
		}
		else
		{
			// Time the RSX thread spent waiting for drawables/frame contexts is back-pressure from the display, not the
			// guest's pace. Counting it would let a pacing target sustain itself even after the game got faster.
			const f64 sample = static_cast<f64>(delta - std::min(delta, pacing.blocked_time)) / 1'000'000.;

			pacing.guest_intervals[pacing.guest_interval_next] = sample;
			pacing.guest_interval_next = (pacing.guest_interval_next + 1) % pacing.guest_intervals.size();
			pacing.guest_interval_count = std::min<u32>(pacing.guest_interval_count + 1, ::size32(pacing.guest_intervals));

			update_pacing_probe(now);
		}
	}

	pacing.last_emu_flip_time = now;
	m_rsx_time_stats.display_wait_us += pacing.blocked_time;
	pacing.blocked_time = 0;
}

void MTLGSRender::update_pacing_probe(u64 now)
{
	// The subtraction of blocked_time only removes the display waits of the RSX thread. When the guest itself waits on
	// flips that pacing delays (or adapts its own cadence to them), the measured intervals follow the pacing target and
	// nothing would ever lower it again: a hitch could lock a 60 fps game at 30 or 15 fps. So while the target paces the
	// game slower than its frame limit, it is tested now and then: frames are paced at the frame limit for a moment and
	// if the game keeps up, the intervals measured under the slower pacing are dropped. A game that really runs slower
	// only sees a short stretch of unpaced frames, less and less often.
	auto& pacing = m_present_pacing;
	const f64 limit_interval = get_frame_limit_interval();
	const f64 tolerance = pacing.refresh_interval * 0.25;

	if (pacing.probe_start_time)
	{
		// Probe running: get_guest_frame_interval() returns the frame limit
		pacing.probe_samples++;

		const u64 elapsed = now - pacing.probe_start_time;
		if ((pacing.probe_samples < probe_min_samples || elapsed < probe_min_duration_us) && elapsed < probe_max_duration_us)
		{
			return;
		}

		pacing.probe_start_time = 0;

		// The most recent intervals are the probe's. The first ones may still include frames queued at the old pace, but
		// only low values count.
		const u32 count = std::min({ pacing.probe_samples, pacing.guest_interval_count, ::size32(pacing.guest_intervals) });
		const f64 probed = (count >= 4)
			? std::max(snap_to_vblanks(recent_lower_quartile(pacing.guest_intervals, pacing.guest_interval_next, count)), limit_interval)
			: 0.;

		if (count >= 4 && get_pacing_slot(probed) < pacing.probe_from_slot - tolerance)
		{
			rsx_log.notice("Metal: pacing: the game kept up with faster presentation (frame interval %.2f ms, paced at %.2f ms before)",
				probed * 1000., pacing.probe_from_slot * 1000.);

			// Keep only the probe's intervals: the older ones were measured while pacing held the game back
			std::array<f64, std::tuple_size_v<decltype(pacing.guest_intervals)>> recent{};
			for (u32 i = 0; i < count; ++i)
			{
				recent[i] = pacing.guest_intervals[(pacing.guest_interval_next + ::size32(recent) - count + i) % ::size32(recent)];
			}

			pacing.guest_intervals = recent;
			pacing.guest_interval_count = count;
			pacing.guest_interval_next = count % ::size32(recent);
			pacing.probe_backoff = probe_initial_backoff_us;
		}
		else
		{
			// The game really runs at that pace
			pacing.probe_backoff = std::min(std::max(pacing.probe_backoff, probe_initial_backoff_us) * 2, probe_max_backoff_us);
		}

		pacing.next_probe_time = now + pacing.probe_backoff;
		return;
	}

	const f64 slot = get_pacing_slot(get_guest_frame_interval());

	if (m_vsync_mode == vsync_mode::off || slot <= get_pacing_slot(limit_interval) + tolerance)
	{
		// Paced at the frame limit (or not paced): nothing to test
		pacing.next_probe_time = 0;
		pacing.probe_backoff = probe_initial_backoff_us;
		return;
	}

	if (!pacing.next_probe_time)
	{
		pacing.next_probe_time = now + std::max(pacing.probe_backoff, probe_initial_backoff_us);
	}
	else if (now >= pacing.next_probe_time)
	{
		pacing.probe_start_time = now;
		pacing.probe_samples = 0;
		pacing.probe_from_slot = slot;
	}
}

f64 MTLGSRender::get_frame_limit_interval() const
{
	// Frame limit, as applied by rsx::thread::handle_emu_flip
	f64 limit = 0.;
	switch (g_disable_frame_limit ? frame_limit_type::none : g_cfg.video.frame_limit.get())
	{
	case frame_limit_type::none: limit = g_cfg.core.max_cpu_preempt_count_per_frame ? static_cast<f64>(g_cfg.video.vblank_rate) : 0.; break;
	case frame_limit_type::_30: limit = 30.; break;
	case frame_limit_type::_50: limit = 50.; break;
	case frame_limit_type::_60: limit = 60.; break;
	case frame_limit_type::_120: limit = 120.; break;
	case frame_limit_type::display_rate: limit = 1. / m_present_pacing.refresh_interval; break;
	case frame_limit_type::_auto: limit = static_cast<f64>(g_cfg.video.vblank_rate); break;
	case frame_limit_type::_ps3:
	case frame_limit_type::infinite:
		break;
	}

	if (const f64 limit2 = g_cfg.video.second_frame_limit; limit2 >= 0.1 && (limit2 < limit || !limit))
	{
		limit = limit2;
	}

	// Same cap as the guest flip limiter (RSXThread handle_emu_flip): a timestep-patched game is paced at exactly 2x the
	// vblank rate (120 fps: one frame per refresh on a 120 Hz screen), and the optional lock holds other games at the vblank
	if (!g_disable_frame_limit)
	{
		const u32 multiplier = g_guest_logic_rate_multiplier.load();
		limit = multiplier > 1 ? rsx::apply_game_speed_lock(limit, 60., multiplier, true)
		                       : rsx::apply_game_speed_lock(limit, rsx::effective_vblank_hz(g_cfg.video.vblank_rate.get(), g_cfg.video.vblank_ntsc.get()),
			                         1, g_guest_speed_lock.load());
	}

	return limit > 0. ? 1. / limit : 0.;
}

f64 MTLGSRender::get_measured_frame_interval() const
{
	// Lower quartile of the recent intervals (see recent_lower_quartile), 0 until there are enough of them
	const auto& pacing = m_present_pacing;

	if (const u32 count = pacing.guest_interval_count; count >= 4)
	{
		return snap_to_vblanks(recent_lower_quartile(pacing.guest_intervals, pacing.guest_interval_next, count));
	}

	return 0.;
}

f64 MTLGSRender::get_guest_frame_interval() const
{
	const f64 limit_interval = get_frame_limit_interval();

	if (m_present_pacing.probe_start_time)
	{
		// Downward probe: pace at the frame limit (see update_pacing_probe)
		return limit_interval;
	}

	// The limit is a lower bound: games often run below it (a 30 fps game with the default 60 fps limit)
	return std::max(get_measured_frame_interval(), limit_interval);
}

f64 MTLGSRender::get_pacing_slot(f64 guest_interval, u32* refreshes) const
{
	const auto& pacing = m_present_pacing;
	const f64 refresh = pacing.refresh_interval;

	if (pacing.fullscreen && pacing.variable_refresh)
	{
		// Adaptive-Sync (fullscreen only): the display refreshes when the frame is due
		if (refreshes)
		{
			*refreshes = 0;
		}

		return std::max(guest_interval, refresh);
	}

	// Fixed refresh grid: keep every frame on screen for the same whole number of refreshes. Intervals are only rounded
	// up when they are close to the next multiple: rounding a game running between two cadences up would throttle it.
	const u32 slot_refreshes = std::max(1u, static_cast<u32>(guest_interval / refresh + 0.25));

	if (refreshes)
	{
		*refreshes = slot_refreshes;
	}

	return slot_refreshes * refresh;
}

void MTLGSRender::present_drawable(mtl::frame_context_t* ctx)
{
	auto drawable = ctx->drawable;
	auto& pacing = m_present_pacing;

	if (!m_present_feedback)
	{
		m_present_feedback = std::make_shared<present_feedback_t>();
	}

	// Telemetry: actual on-screen times. Handlers must be added before presenting. The handler owns a reference to the
	// feedback block, so it may run after the renderer is gone.
	m_present_feedback->refresh_interval.store(pacing.refresh_interval);
	drawable->addPresentedHandler(MTL::DrawablePresentedHandlerFunction([feedback = m_present_feedback](MTL::Drawable* presented)
	{
		feedback->on_presented(presented->presentedTime());
	}));

	const f64 now = mtl::get_media_time();
	const f64 refresh = pacing.refresh_interval;
	f64 min_duration = 0.;
	u32 slot_refreshes = 0;

	if (m_vsync_mode != vsync_mode::off)
	{
		if (const f64 guest_interval = get_guest_frame_interval(); guest_interval > 0.)
		{
			f64 slot = get_pacing_slot(guest_interval, &slot_refreshes);

			if (const u64 flip_blocked = pacing.blocked_time - std::min(pacing.blocked_time, pacing.flip_blocked_start);
				flip_blocked >= display_backlog_threshold_us && slot > refresh * 1.5)
			{
				// This flip had to wait for the display (a drawable or a frame context): frames are queued ahead of this
				// one. A frame is never shown sooner than a slot after the previous one, so the queue would stay full
				// for good after a hitch, and every later flip would wait for the display too: the flip completes late
				// for the guest, and a guest slightly faster than the slot is throttled to it. One refresh less drains
				// the queue.
				slot -= refresh;
				slot_refreshes = slot_refreshes ? slot_refreshes - 1 : 0;
				pacing.catch_up_frames++;
			}

			// Adaptive-Sync: shown when due. Fixed refresh grid: any time between the previous refresh and the target
			// one, half a refresh of margin absorbs jitter.
			min_duration = slot_refreshes ? slot - refresh * 0.5 : slot - 0.0005;

			if (m_vsync_mode == vsync_mode::adaptive && pacing.last_present_time > 0. &&
				now - pacing.last_present_time > slot + refresh * 0.5)
			{
				// Adaptive: the frame is already late for its slot. Show it as soon as possible instead of holding the cadence.
				min_duration = 0.;
				slot_refreshes = 0;
			}
		}
	}

	if (min_duration > 0.)
	{
		drawable->presentAfterMinimumDuration(min_duration);
	}
	else
	{
		drawable->present();
	}

	pacing.last_present_time = now;
	pacing.min_duration = min_duration;
	pacing.slot_refreshes = slot_refreshes;

	// Rate-limited telemetry
	const u64 now_us = get_system_time();
	if (!pacing.stats_time)
	{
		pacing.stats_time = now_us;
		m_rsx_time_stats = { .waits = rsx::g_sync_wait_stats.snapshot() };
	}
	else if (now_us - pacing.stats_time >= present_stats_interval_us)
	{
		auto& feedback = *m_present_feedback;
		std::array<u32, 6> counts{};
		for (usz i = 0; i < counts.size(); ++i)
		{
			counts[i] = feedback.intervals[i].exchange(0);
		}

		const u32 dropped = feedback.dropped.exchange(0);
		// The pacing target (this frame may have been shown sooner: adaptive VSync, queue drain)
		const f64 target_interval = get_guest_frame_interval();
		u32 target_refreshes = 0;
		get_pacing_slot(target_interval, &target_refreshes);

		std::string mode = (m_vsync_mode == vsync_mode::off) ? "vsync off" :
			(target_interval <= 0.) ? "unpaced" :
			target_refreshes ? fmt::format("%u refresh(es) per frame", target_refreshes) : std::string("variable refresh");

		if (pacing.probe_start_time)
		{
			mode += " (testing the frame limit)";
		}

		rsx_log.notice("Metal: presentation over %us: frames on screen for 1/2/3/4/5/6+ refreshes (%.2f ms): %u/%u/%u/%u/%u/%u, not displayed: %u. "
			"Pacing: %s, guest frame interval %.2f ms, minimum duration %.2f ms, %u frame(s) shortened to drain queued frames",
			(now_us - pacing.stats_time) / 1'000'000, refresh * 1000., counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], dropped,
			mode, target_interval * 1000., min_duration * 1000., pacing.catch_up_frames);

		pacing.catch_up_frames = 0;

		// GPU load. A rising GPU time per frame in the same scene means the GPU clock dropped (heat).
		u32 frames = dropped;
		for (const u32 count : counts)
		{
			frames += count;
		}

		const auto gpu = mtl::get_gpu_stats_and_reset();
		const f64 window_ms = (now_us - pacing.stats_time) / 1000.;
		if (frames && window_ms > 0.)
		{
			const f64 busy_ms = gpu.busy_ns / 1'000'000.;
			const auto per_frame = [frames](u64 count) { return static_cast<f64>(count) / frames; };
			const auto program_cache_sizes = m_prog_buffer->get_cache_sizes();
			// Skipped draws / pipeline waits: shader compilation stutter (the shader interpreter line says why draws
			// were not interpreted). Render passes and feedback splits: the render passes line below.
			rsx_log.notice("Metal: GPU busy %.2f ms per frame (%.0f%% of the time), "
				"%.1f image uploads from memory per frame (%.1f ahead of the render pass, %.1f ended one). "
				"Pipelines: %.1f draws skipped and %.2f ms waited per frame (program cache: %llu pipelines, %llu vertex and %llu fragment programs)",
				busy_ms / frames, 100. * busy_ms / window_ms,
				per_frame(gpu.uploads_ahead + gpu.uploads_inline), per_frame(gpu.uploads_ahead), per_frame(gpu.uploads_inline_split),
				per_frame(m_skipped_draws), per_frame(m_pipeline_wait_us) / 1000.,
				static_cast<unsigned long long>(program_cache_sizes.pipelines),
				static_cast<unsigned long long>(program_cache_sizes.vertex_programs),
				static_cast<unsigned long long>(program_cache_sizes.fragment_programs));

			// Render pass structure: on a tile-based GPU every pass loads and stores its attachments, so a high pass count
			// means attachments are stored and reloaded often (feedback splits, surface changes, clears, copies). Shader
			// reads of the depth buffer a pass attaches (texture units, depth compare emulation) read a copy of it, made
			// outside the pass (ending an open one) whenever depth changed since the previous copy. The depth bounds
			// test reads no depth buffer, so its draws copy nothing.
			const std::string depth_bounds = m_device->caps().depth_bounds ? std::string("in hardware") :
				fmt::format("%.1f draws per frame emulated in the fragment shader", per_frame(m_depth_bounds_draws));
			rsx_log.notice("Metal: render passes: %s; depth bounds test: %s; shader reads of the depth buffer during its pass: "
				"%.1f copies of it per frame (%.1f pass splits)", mtl::describe_render_passes(gpu, frames), depth_bounds,
				per_frame(m_depth_copies), per_frame(gpu.splits_by_reason[static_cast<u32>(mtl::pass_split_reason::depth_copy)]));

			// CPU encoding cost: driver calls the argument table / encoder state caches avoided
			rsx_log.notice("Metal: encoding per frame: %.1f submissions, %.0f argument table writes (%.0f skipped as redundant), "
				"%.0f pipeline state / argument table sets (%.0f skipped), %.1f transient views from the texture view pool, %.1f residency set commits",
				per_frame(gpu.submissions), per_frame(gpu.table_writes), per_frame(gpu.table_writes_skipped),
				per_frame(gpu.state_sets), per_frame(gpu.state_sets_skipped), per_frame(mtl::get_transient_views_and_reset()),
				per_frame(m_device->get_residency_commits_and_reset()));

			// GPU concurrency (hazard-tracked barriers): compute commands that began without waiting for earlier work can
			// overlap it; render passes begin with the two pass barriers (tile-based GPU: nothing after a pass's first draw
			// orders it, so "after the first draw" must stay 0); submissions that did not conflict with earlier ones still
			// running start right away
			rsx_log.notice("Metal: barriers per frame: %.1f of %.1f compute commands began without a barrier; %.1f queue barriers "
				"(%.1f render pass barriers, %.1f before a pass's first draw for what its vertex stage reads, %.1f after it), "
				"%.1f pass splits for vertex reads, %.1f intra-encoder barriers, %.1f full-barrier fallbacks (undeclared accesses); "
				"%.1f submissions waited for an earlier conflicting one, %.1f overlapped earlier work",
				per_frame(gpu.ordering_points_free), per_frame(gpu.ordering_points), per_frame(gpu.queue_barriers), per_frame(gpu.pass_barriers),
				per_frame(gpu.first_draw_barriers), per_frame(gpu.late_barriers),
				per_frame(gpu.splits_by_reason[static_cast<u32>(mtl::pass_split_reason::vertex_read)]),
				per_frame(gpu.encoder_barriers), per_frame(gpu.full_barriers), per_frame(gpu.submissions_waited), per_frame(gpu.submissions_overlapped));
		}

		// GPU -> CPU synchronization: time threads spent blocked on RSX timeline work (readbacks, zcull report reads,
		// hard syncs). Rising RSX-thread waits with falling GPU utilization mean the CPU and GPU take turns.
		const auto cpu_waits = mtl::get_cpu_wait_stats_and_reset();
		if (frames)
		{
			const auto per_frame = [frames](u64 value) { return static_cast<f64>(value) / frames; };
			rsx_log.notice("Metal: waits for GPU work per frame: RSX thread %.2f ms in %.2f waits (longest %.2f ms), other threads %.2f ms in %.2f waits (longest %.2f ms)",
				per_frame(cpu_waits.renderer.total_us) / 1000., per_frame(cpu_waits.renderer.count), cpu_waits.renderer.max_us / 1000.,
				per_frame(cpu_waits.others.total_us) / 1000., per_frame(cpu_waits.others.count), cpu_waits.others.max_us / 1000.);

			// Where the RSX thread waits (the pass_context it was in): readback = guest reads of GPU data (Write Color
			// Buffers, DMA), query = zcull reports, texture setup = texture cache, other = command list recycling etc.
			static constexpr std::array<const char*, mtl::cpu_wait_stats_t::context_count> context_names =
			{
				"other", "surface setup", "texture setup", "clear", "blit", "query", "readback", "present"
			};

			std::string by_context;
			for (u32 i = 0; i < mtl::cpu_wait_stats_t::context_count; i++)
			{
				if (const auto& b = cpu_waits.renderer_by_context[i]; b.count)
				{
					fmt::append(by_context, "%s%s %.2f ms in %.2f", by_context.empty() ? "" : ", ", context_names[i], per_frame(b.total_us) / 1000., per_frame(b.count));
				}
			}

			if (!by_context.empty())
			{
				rsx_log.notice("Metal: RSX thread waits for GPU work per frame by cause: %s", by_context);
			}

			if (const std::string sites = mtl::take_other_wait_sites(static_cast<u32>(frames)); !sites.empty())
			{
				rsx_log.notice("Metal: RSX thread waits for GPU work by site: %s", sites);
			}
		}

		// RSX thread load per guest frame: work (the RSX frame statistics, flip; "other" is FIFO command processing,
		// surface/texture cache work, readbacks, GPU waits) and idle time (waiting for commands or semaphores from the
		// guest, frame limiter), plus the PPU's waits for the RSX in HLE cellGcm. Idle near 0 with the PPU waiting: the
		// RSX thread limits the frame rate. Idle mostly "waiting for commands": the guest (PPU/SPU) does.
		const auto sync_waits = rsx::g_sync_wait_stats.snapshot();
		if (const auto& rsx_time = m_rsx_time_stats; rsx_time.frames && window_ms > 0.)
		{
			using enum rsx::sync_wait;
			const f64 frames_f = rsx_time.frames;
			const auto ms = [frames_f](f64 us) { return us / 1000. / frames_f; };
			const auto wait_ms = [&](rsx::sync_wait what) { return ms(static_cast<f64>(sync_waits.time_of(what) - rsx_time.waits.time_of(what))); };
			const auto waits = [&](rsx::sync_wait what) { return static_cast<f64>(sync_waits.count_of(what) - rsx_time.waits.count_of(what)) / frames_f; };

			const f64 frame_ms = window_ms / frames_f;
			const f64 idle_ms = wait_ms(fifo_empty) + wait_ms(flip_semaphore) + wait_ms(semaphore) + wait_ms(frame_limiter);
			const f64 work_ms = ms(static_cast<f64>(rsx_time.setup_us + rsx_time.vertex_upload_us + rsx_time.texture_upload_us + rsx_time.draw_exec_us)) +
				ms(static_cast<f64>(rsx_time.flip_us));

			rsx_log.notice("Metal: RSX thread per guest frame (%.2f ms, %.0f draw calls): busy %.2f ms (draw setup %.2f, vertex upload %.2f, texture upload %.2f, "
				"draw execution %.2f, flip %.2f incl. %.2f waiting for the display, other %.2f), idle %.2f ms (waiting for commands %.2f in %.1f waits, "
				"flip semaphore %.2f, other semaphores %.2f in %.1f waits, frame limiter %.2f). PPU waits for the RSX per frame: command buffer full %.2f ms "
				"in %.2f waits, flip status polled %.1f times while pending (%.2f ms)",
				frame_ms, rsx_time.draw_calls / frames_f, frame_ms - idle_ms, ms(static_cast<f64>(rsx_time.setup_us)), ms(static_cast<f64>(rsx_time.vertex_upload_us)),
				ms(static_cast<f64>(rsx_time.texture_upload_us)), ms(static_cast<f64>(rsx_time.draw_exec_us)), ms(static_cast<f64>(rsx_time.flip_us)),
				ms(static_cast<f64>(rsx_time.display_wait_us)), frame_ms - idle_ms - work_ms, idle_ms, wait_ms(fifo_empty), waits(fifo_empty),
				wait_ms(flip_semaphore), wait_ms(semaphore), waits(semaphore), wait_ms(frame_limiter),
				wait_ms(ppu_command_buffer), waits(ppu_command_buffer), waits(ppu_flip_status), wait_ms(ppu_flip_status));
		}

		m_rsx_time_stats = { .waits = sync_waits };

		// Flexible render pipeline states: how new render pipelines were created in this window (MTLPipelineCompiler.h)
		const auto pipelines = mtl::get_pipeline_creation_stats_and_reset();
		const auto average_ms = [](u64 total_us, u32 count) { return count ? total_us / 1000. / count : 0.; };
		rsx_log.notice("Metal: render pipelines over %us: %u created by specialization (%.2f ms on average, longest %.2f ms), %u with full state "
			"(%u listed in the pipeline archive, %u after a failed specialization), %u unspecialized built (%.2f ms on average); "
			"background full-state compiles: %u requested, %u built, %u swapped in",
			(now_us - pacing.stats_time) / 1'000'000, pipelines.specialized, average_ms(pipelines.specialization_us, pipelines.specialized),
			pipelines.specialization_max_us / 1000., pipelines.full_state, pipelines.full_state_archived, pipelines.fallbacks,
			pipelines.unspecialized, average_ms(pipelines.unspecialized_us, pipelines.unspecialized),
			pipelines.upgrades_requested, pipelines.upgrades_built, pipelines.upgrades_swapped);

		// Where compile time went in this window, all threads (the shader cache preload also logs its own totals)
		const auto timings = mtl::get_compile_timings_and_reset();
		rsx_log.notice("Metal: shader builds over %us: GLSL->MSL %u (%.1f ms on average), MTLLibrary %u (%.1f ms on average); pipelines (render "
			"and compute) with archive lookups %u (%.1f ms on average), compiled without %u (%.1f ms on average)",
			(now_us - pacing.stats_time) / 1'000'000, timings.translated, average_ms(timings.translate_us, timings.translated),
			timings.libraries, average_ms(timings.library_us, timings.libraries), timings.archive_pipelines,
			average_ms(timings.archive_pipeline_us, timings.archive_pipelines), timings.compiled_pipelines,
			average_ms(timings.compiled_pipeline_us, timings.compiled_pipelines));

		if (frames && (g_cfg.video.shadermode == shader_mode::async_with_interpreter || g_cfg.video.shadermode == shader_mode::interpreter_only))
		{
			// Shader interpreter: draws it drew instead of skipping them, and why the others were skipped
			using reason = mtl::shader_interpreter::skip_reason;
			const auto per_frame = [frames](u64 value) { return static_cast<f64>(value) / frames; };
			const auto skips = [this](reason r) { return m_interpreter_skips[static_cast<u32>(r)]; };
			const auto interpreter = m_shader_interpreter.get_stats_and_reset();

			rsx_log.notice("Metal: shader interpreter: %.1f draws per frame; skipped per frame: %.1f waiting for its pipeline, %.1f with programs "
				"it does not run, %.1f whose pipeline failed. Pipelines built: %u (+%u specialized for a program, %u failed), %llu in total",
				per_frame(m_interpreter_draws), per_frame(skips(reason::not_ready)), per_frame(skips(reason::unsupported) + skips(reason::inexact)),
				per_frame(skips(reason::failed)), interpreter.pipelines_built, interpreter.variants_built, interpreter.pipelines_failed,
				static_cast<unsigned long long>(m_shader_interpreter.get_pipeline_count()));
		}

		// GPU flows outside the draw passes (MSAA resolve/unresolve, blit/scale passes, compute kernels,
		// uploads, DMA readbacks, surface init/spill, depth copies, presents) plus the worst graphics issues
		// collected since the last window. This is the line to paste into a bug report.
		if (frames)
		{
			mtl::graphics_log_report(frames);
		}

		// What the GPU actually produced: black / partly black screens, NaN/Inf from shader math, a screen map
		if (m_frame_inspector)
		{
			m_frame_inspector->report();
		}

		m_skipped_draws = 0;
		m_pipeline_wait_us = 0;
		m_interpreter_draws = 0;
		m_interpreter_skips.fill(0);
		m_depth_bounds_draws = 0;
		m_depth_copies = 0;
		pacing.stats_time = now_us;
	}
}

void MTLGSRender::present(mtl::frame_context_t *ctx)
{
	ensure(ctx->drawable && ctx->present_image && ctx->swap_command_buffer);
	ensure(m_present_queue);

	// The frame's main-queue work (including the present passes) is complete once the RSX timeline reaches the value of
	// its last list. Deferred (MTRSX) submissions get that value once the offloader thread has committed the list.
	auto frame_cb = ctx->swap_command_buffer;
	frame_cb->flush();

	u64 frame_eid = 0;
	{
		reader_lock lock(frame_cb->guard_mutex);
		ctx->swap_timeline_value = frame_cb->get_fence().value; // 0: already retired (complete)
		frame_eid = frame_cb->eid_tag;
	}

	if (!ctx->present_command_buffer)
	{
		const auto index = static_cast<u32>(ctx - m_frame_context_storage.data());
		ctx->present_command_buffer = std::make_unique<mtl::command_buffer_chunk>();
		ctx->present_command_buffer->create(*m_device, m_present_queue, m_present_timeline, fmt::format("RSX present #%u", index));
		ctx->present_command_buffer->access_hint = mtl::command_list::access_type_hint::all;
	}

	auto cmd = ctx->present_command_buffer.get();
	cmd->reset(); // Idle: the frame context was cleaned up before it was reused
	cmd->begin();
	cmd->clear_flags();

	// Same size and format: flip() (re)creates present_image from this drawable's texture
	MTL::Texture* src = ctx->present_image->value;
	MTL::Texture* dst = ctx->drawable->texture();
	const NS::UInteger width = std::min(src->width(), dst->width());
	const NS::UInteger height = std::min(src->height(), dst->height());
	cmd->blit({ mtl::read_texture(src), mtl::write_texture(dst) })
		->copyFromTexture(src, 0, 0, MTL::Origin(0, 0, 0), MTL::Size(width, height, 1), dst, 0, 0, MTL::Origin(0, 0, 0));

	// Present queue: wait for the frame's work and for the drawable, copy, signal the drawable
	mtl::submit_info_t submit_info{};
	if (ctx->swap_timeline_value)
	{
		submit_info.wait_event = m_timeline.handle();
		submit_info.wait_value = ctx->swap_timeline_value;
	}
	submit_info.wait_drawable = ctx->drawable;
	submit_info.signal_drawable = ctx->drawable;

	// Completion of the present list implies completion of the frame's main-queue list: report that list's event id
	// (GC) when it completes, like the single-queue path did when retiring the swap list
	cmd->eid_tag = frame_eid;
	{
		std::lock_guard lock(cmd->guard_mutex);
		mtl::queue_submit_now(*cmd, submit_info);
	}

	// From now on the present list ends the frame (check_present_status / frame_context_cleanup poll it)
	ctx->swap_command_buffer = cmd;

	if (!swapchain_unavailable)
	{
		present_drawable(ctx);
	}

	// Presentation image released; the layer keeps the drawable alive until it has been displayed
	ctx->drawable->release();
	ctx->drawable = nullptr;
}

void MTLGSRender::advance_queued_frames()
{
	// Check all other frames for completion and clear resources
	check_present_status();

	// Run video memory balancer
	if (const auto load_severity = mtl::vmm_determine_memory_load_severity();
		load_severity >= rsx::problem_severity::moderate)
	{
		on_vram_exhausted(load_severity);
	}

	// m_rtts storage is double buffered and should be safe to tag on frame boundary
	m_rtts.trim(*m_current_command_buffer, mtl::vmm_determine_memory_load_severity());

	// Texture cache is also double buffered to prevent use-after-free
	m_texture_cache.on_frame_end();
	m_samplers_dirty.store(true);

	m_vertex_cache->purge();
	m_current_frame->tag_frame_end();

	m_queued_frames.push_back(m_current_frame);
	ensure(m_queued_frames.size() <= m_max_async_frames);

	m_current_queue_index = (m_current_queue_index + 1) % m_max_async_frames;
	m_current_frame = &m_frame_context_storage[m_current_queue_index];
	m_current_frame->flags |= frame_context_state::dirty;

	mtl::advance_frame_counter();

	report_resource_usage();
}

void MTLGSRender::queue_swap_request()
{
	ensure(!m_current_frame->swap_command_buffer);
	m_current_frame->swap_command_buffer = m_current_command_buffer;

	// The frame's last main-queue list. It does not wait for the drawable: the present passes rendered into the frame's
	// present image, which the present list copies into the drawable on the present queue.
	close_and_submit_command_buffer();

	// Set up a present request for this frame as well
	if (m_current_frame->drawable)
	{
		present(m_current_frame);
	}

	// Grab next cb in line and make it usable
	m_current_command_buffer = m_primary_cb_list.next();
	m_current_command_buffer->reset();
	m_current_command_buffer->clear_flags();

	if (m_occlusion_query_active)
	{
		m_current_command_buffer->flags |= mtl::command_list::cb_load_occluson_task;
	}

	m_current_command_buffer->begin();

	// Set up new pointers for the next frame
	advance_queued_frames();
}

void MTLGSRender::frame_context_cleanup(mtl::frame_context_t *ctx)
{
	ensure(ctx->swap_command_buffer);

	// Perform hard swap here
	mtl::wait_site_scope wait_site("frame context (an earlier frame still on the GPU)");
	if (!ctx->swap_command_buffer->wait(FRAME_PRESENT_TIMEOUT))
	{
		// GPU hang, stop presenting
		swapchain_unavailable = true;
	}

	// Resource cleanup.
	{
		if (m_overlay_manager && m_overlay_manager->has_dirty())
		{
			auto ui_renderer = mtl::get_overlay_pass<mtl::ui_overlay_renderer>();
			m_overlay_manager->lock_shared();

			std::vector<u32> uids_to_dispose;
			uids_to_dispose.reserve(m_overlay_manager->get_dirty().size());

			for (const auto& view : m_overlay_manager->get_dirty())
			{
				ui_renderer->remove_temp_resources(view->uid);
				uids_to_dispose.push_back(view->uid);
			}

			m_overlay_manager->unlock_shared();
			m_overlay_manager->dispose(uids_to_dispose);
		}

		mtl::get_resource_manager()->trim();

		mtl::reset_global_resources();

		if (ctx->last_frame_sync_time > m_last_heap_sync_time)
		{
			m_last_heap_sync_time = ctx->last_frame_sync_time;

			// Heap cleanup; deallocates memory consumed by the frame if it is still held
			mtl::data_heap_manager::restore_snapshot(ctx->heap_snapshot);
		}
	}

	ctx->swap_command_buffer = nullptr;
	ctx->swap_timeline_value = 0;

	// Remove from queued list
	while (!m_queued_frames.empty())
	{
		auto frame = m_queued_frames.front();
		m_queued_frames.pop_front();

		if (frame == ctx)
		{
			break;
		}
	}

	mtl::advance_completed_frame_counter();
}

void MTLGSRender::serve_guest_during_display_wait()
{
	// The pacing only discounts the RSX thread's own display waits (blocked_time). That is valid only if the guest keeps
	// running meanwhile, so the RSX thread must not sit on anything the guest waits for.

	if (m_queue_status & flush_queue_state::deadlock)
	{
		// Offloader fault (see do_local_task)
		on_invalidate_memory_range(m_offloader_fault_range, m_offloader_fault_cause);
		m_queue_status.clear(flush_queue_state::deadlock);
	}

	// PPU/SPU threads that fault on surfaces wait for the RSX thread to submit its work (on_access_violation). Same as
	// do_local_task, whose base part must not run here: it may start an emulated flip, and this can be one.
	if (!(m_queue_status & flush_queue_state::flushing) && m_flush_requests.pending() && m_flush_queue_mutex.try_lock())
	{
		flush_command_queue();

		m_flush_requests.clear_pending_flag();
		m_flush_requests.consumer_wait();
		m_flush_queue_mutex.unlock();
	}

	// Labels held back for zcull reports: write the ones whose reports are done (CPU threads may be polling them)
	if (zcull_ctrl && zcull_ctrl->has_deferred_labels())
	{
		zcull_ctrl->update(this);
	}
}

void MTLGSRender::wait_for_frame_context(mtl::frame_context_t* ctx)
{
	// Waiting for an older frame to leave the present queue is back-pressure from the display (see update_present_pacing)
	const u64 wait_start = get_system_time();

	if (auto cmd = ctx->swap_command_buffer; cmd && !cmd->poke())
	{
		// Committed from here on (deferred submissions), so its fence can be waited on
		cmd->flush();

		while (true)
		{
			serve_guest_during_display_wait();

			// Serving the guest may have retired the frame (flush_command_queue -> check_present_status)
			if (ctx->swap_command_buffer != cmd || cmd->poke() || get_system_time() - wait_start >= FRAME_PRESENT_TIMEOUT)
			{
				break;
			}

			// Up to 1 ms (the finest timeout MTLSharedEvent offers), wakes up as soon as the list completes
			cmd->get_fence().wait(1000);
		}
	}

	if (ctx->swap_command_buffer)
	{
		// Retires the frame (reports a GPU hang if the list never completed)
		frame_context_cleanup(ctx);
	}

	m_present_pacing.blocked_time += get_system_time() - wait_start;
}

mtl::viewable_image* MTLGSRender::get_present_source(/* inout */ mtl::present_surface_info* info, const rsx::avconf& avconfig)
{
	mtl::viewable_image* image_to_flip = nullptr;

	// @FIXME: This entire function needs to be rewritten to go through the texture cache's "upload_texture" routine.
	// That method is not a 1:1 replacement due to handling of insets that is done differently here.

	// Check the surface store first
	const auto format_bpp = rsx::get_format_block_size_in_bytes(info->format);
	const auto overlap_info = m_rtts.get_merged_texture_memory_region(*m_current_command_buffer,
		info->address, info->width, info->height, info->pitch, format_bpp, rsx::surface_access::transfer_read);

	if (!overlap_info.empty())
	{
		const auto& section = overlap_info.back();
		auto surface = mtl::as_rtt(section.surface);
		bool viable = false;

		if (section.base_address >= info->address)
		{
			const auto surface_width = surface->template get_surface_width<rsx::surface_metrics::samples>();
			const auto surface_height = surface->template get_surface_height<rsx::surface_metrics::samples>();

			if (section.base_address == info->address)
			{
				// Check for fit or crop
				viable = (surface_width >= info->width && surface_height >= info->height);
			}
			else
			{
				// Check for borders and letterboxing
				const u32 inset_offset = section.base_address - info->address;
				const u32 inset_y = inset_offset / info->pitch;
				const u32 inset_x = (inset_offset % info->pitch) / format_bpp;

				const u32 full_width = surface_width + inset_x + inset_x;
				const u32 full_height = surface_height + inset_y + inset_y;

				viable = (full_width == info->width && full_height == info->height);
			}

			if (viable)
			{
				image_to_flip = section.surface->get_surface(rsx::surface_access::transfer_read);

				std::tie(info->width, info->height) = rsx::apply_resolution_scale<true>(
					resolution_scaling_config,
					std::min(surface_width, info->width),
					std::min(surface_height, info->height));
			}
		}
	}
	else if (auto surface = m_texture_cache.find_texture_from_dimensions<true>(info->address, info->format);
			 surface && surface->get_width() >= info->width && surface->get_height() >= info->height)
	{
		// Hack - this should be the first location to check for output
		// The render might have been done offscreen or in software and a blit used to display
		image_to_flip = dynamic_cast<mtl::viewable_image*>(surface->get_raw_texture());
	}

	// The correct output format is determined by the AV configuration set in CellVideoOutConfigure by the game.
	// 99.9% of the time, this will match the backbuffer fbo format used in rendering/compositing the output.
	// But in some cases, let's just say some devs are creative.
	const auto expected_format = RSX_display_format_to_mtl_format(avconfig.format);

	if (!image_to_flip) [[ unlikely ]]
	{
		// Read from cell
		const auto range = utils::address_range32::start_length(info->address, info->pitch * info->height);
		const u32  lookup_mask = rsx::texture_upload_context::blit_engine_dst | rsx::texture_upload_context::framebuffer_storage;
		const auto overlap = m_texture_cache.find_texture_from_range<true>(range, 0, lookup_mask);

		for (const auto & section : overlap)
		{
			if (!section->is_synchronized())
			{
				section->copy_texture(*m_current_command_buffer, true);
			}
		}

		if (m_current_command_buffer->flags & mtl::command_list::cb_has_dma_transfer)
		{
			// Submit for processing to lower hard fault penalty
			flush_command_queue();
		}

		m_texture_cache.invalidate_range(*m_current_command_buffer, range, rsx::invalidation_cause::read);
		image_to_flip = m_texture_cache.upload_image_simple(*m_current_command_buffer, expected_format, info->address, info->width, info->height, info->pitch);
	}
	else if (image_to_flip->format() != expected_format)
	{
		// Devs are being creative. Force-cast this to the proper pixel layout.
		auto dst_img = m_texture_cache.create_temporary_subresource_storage(
			RSX_FORMAT_CLASS_COLOR, expected_format, static_cast<u16>(info->width), static_cast<u16>(info->height), 1, 1, 1,
			MTL::TextureType2D, 0, MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget);

		if (dst_img)
		{
			const areai src_rect = { 0, 0, static_cast<int>(info->width), static_cast<int>(info->height) };
			const areai dst_rect = src_rect;

			if (mtl::formats_are_bitcast_compatible(dst_img.get(), image_to_flip))
			{
				mtl::copy_image(*m_current_command_buffer, image_to_flip, dst_img.get(), src_rect, dst_rect);
			}
			else
			{
				mtl::copy_image_typeless(*m_current_command_buffer, image_to_flip, dst_img.get(), src_rect, dst_rect);
			}

			image_to_flip = dst_img.get();
			m_texture_cache.dispose_reusable_image(dst_img);
		}
	}

	return image_to_flip;
}

void MTLGSRender::flip(const rsx::display_flip_info_t& info)
{
	mtl::autorelease_scope pool;
	mtl::pass_context_scope pass_context(mtl::pass_context::present);

	// RSX thread time telemetry (present_drawable): the flipped frame's statistics and the time spent here
	struct flip_timer_t
	{
		u64& total_us;
		const u64 start_us = get_system_time();

		~flip_timer_t()
		{
			total_us += get_system_time() - start_us;
		}
	} flip_timer{ m_rsx_time_stats.flip_us };

	if (info.emu_flip)
	{
		mtl::graphics_log_next_frame();
		m_rsx_time_stats.frames++;
		m_rsx_time_stats.draw_calls += info.stats.draw_calls;
		m_rsx_time_stats.setup_us += info.stats.setup_time;
		m_rsx_time_stats.vertex_upload_us += info.stats.vertex_upload_time;
		m_rsx_time_stats.texture_upload_us += info.stats.textures_upload_time;
		m_rsx_time_stats.draw_exec_us += info.stats.draw_exec_time;
	}

	// New frame, new budget for waiting on pipelines that are still compiling (see load_program)
	m_async_compile_wait_spent_us = 0;
	m_unsupported_wait_spent_us = 0;
	update_shader_preload_notification();

	// Check surface condition/status. CAMetalLayer does not report resizes, poll the window size.
	if (m_swapchain_dims.width != m_frame->client_width() + 0u ||
		m_swapchain_dims.height != m_frame->client_height() + 0u)
	{
		should_reinitialize_swapchain = true;
	}

	if (m_vsync_mode != g_cfg.video.vsync)
	{
		should_reinitialize_swapchain = true;
	}

	if (swapchain_unavailable || should_reinitialize_swapchain)
	{
		// Reinitializing only fails for minimized windows (or a missing layer); the frame is then skipped below.
		reinitialize_swapchain();
	}

	m_profiler.start();

	// Screen properties and the guest frame interval (pacing inputs)
	update_present_pacing(info.emu_flip);
	m_present_pacing.flip_blocked_start = m_present_pacing.blocked_time;

	ensure(m_current_frame, "Invalid frame context setup");

	if (m_current_frame == &m_aux_frame_context)
	{
		m_current_frame = &m_frame_context_storage[m_current_queue_index];
		if (m_current_frame->swap_command_buffer)
		{
			// Its possible this flip request is triggered by overlays and the flip queue is in undefined state
			wait_for_frame_context(m_current_frame);
		}

		// Swap aux storage and current frame; aux storage should always be ready for use at all times
		m_current_frame->grab_resources(m_aux_frame_context);
	}
	else if (m_current_frame->swap_command_buffer)
	{
		if (info.stats.draw_calls > 0)
		{
			// This can be 'legal' if the window was being resized and no polling happened because of swapchain_unavailable flag
			rsx_log.error("Possible data corruption on frame context storage detected");
		}

		// There were no draws and back-to-back flips happened
		wait_for_frame_context(m_current_frame);
	}

	// Frame without presentation (skipped frame, minimized window, drawable acquisition timeout)
	const auto mini_flip = [&](bool skip_frame)
	{
		if (!skip_frame)
		{
			// Perform a mini-flip here without invoking present code
			m_current_frame->swap_command_buffer = m_current_command_buffer;
			flush_command_queue(true);
			mtl::advance_frame_counter();
			frame_context_cleanup(m_current_frame);
		}

		m_frame->flip(m_context);
		rsx::thread::flip(info);
	};

	if (info.skip_frame || swapchain_unavailable)
	{
		mini_flip(info.skip_frame);
		return;
	}

	u32 buffer_width = display_buffers[info.buffer].width;
	u32 buffer_height = display_buffers[info.buffer].height;
	u32 buffer_pitch = display_buffers[info.buffer].pitch;

	u32 av_format;
	const auto& avconfig = g_fxo->get<rsx::avconf>();

	if (!buffer_width)
	{
		buffer_width = avconfig.resolution_x;
		buffer_height = avconfig.resolution_y;
	}

	if (avconfig.state)
	{
		av_format = avconfig.get_compatible_gcm_format();
		if (!buffer_pitch)
			buffer_pitch = buffer_width * avconfig.get_bpp();

		const size2u video_frame_size = avconfig.video_frame_size();
		buffer_width = std::min(buffer_width, video_frame_size.width);
		buffer_height = std::min(buffer_height, video_frame_size.height);
	}
	else
	{
		av_format = CELL_GCM_TEXTURE_A8R8G8B8;
		if (!buffer_pitch)
			buffer_pitch = buffer_width * 4;
	}

	// Scan memory for required data. This is done early to optimize waiting for the drawable below.
	mtl::viewable_image* image_to_flip = nullptr;
	mtl::viewable_image* image_to_flip2 = nullptr;
	u32 present_address = 0;

	if (info.buffer < display_buffers_count && buffer_width && buffer_height)
	{
		mtl::present_surface_info present_info
		{
			.address = rsx::get_address(display_buffers[info.buffer].offset, CELL_GCM_LOCATION_LOCAL),
			.format = av_format,
			.width = buffer_width,
			.height = buffer_height,
			.pitch = buffer_pitch,
			.eye = 0
		};
		present_address = present_info.address;
		image_to_flip = get_present_source(&present_info, avconfig);

		if (avconfig.stereo_enabled) [[unlikely]]
		{
			const auto [unused, min_expected_height] = rsx::apply_resolution_scale<true>(resolution_scaling_config, RSX_SURFACE_DIMENSION_IGNORED, buffer_height + 30);
			if (image_to_flip->height() < min_expected_height)
			{
				// Get image for second eye
				const u32 image_offset = (buffer_height + 30) * buffer_pitch + display_buffers[info.buffer].offset;
				present_info.width = buffer_width;
				present_info.height = buffer_height;
				present_info.address = rsx::get_address(image_offset, CELL_GCM_LOCATION_LOCAL);
				present_info.eye = 1;

				image_to_flip2 = get_present_source(&present_info, avconfig);
			}
			else
			{
				// Account for possible insets
				const auto [unused2, scaled_buffer_height] = rsx::apply_resolution_scale<true>(resolution_scaling_config, RSX_SURFACE_DIMENSION_IGNORED, buffer_height);
				buffer_height = std::min<u32>(image_to_flip->height() - min_expected_height, scaled_buffer_height);
			}
		}

		buffer_width = present_info.width;
		buffer_height = present_info.height;
	}

	if (info.emu_flip)
	{
		evaluate_cpu_usage_reduction_limits();
	}

	const bool has_overlay = (m_overlay_manager && m_overlay_manager->has_visible());
	const bool user_asked_for_screenshot = g_user_asked_for_screenshot.exchange(false);
	const bool user_is_recording = (g_recording_mode != recording_mode::stopped && m_frame->can_consume_frame());
	const bool need_media_capture = user_asked_for_screenshot || user_is_recording;

	const auto render_overlays = [&](const mtl::overlay_target& target, const areau& area)
	{
		if (!has_overlay) return;

		// Lock to avoid modification during run-update chain
		auto ui_renderer = mtl::get_overlay_pass<mtl::ui_overlay_renderer>();
		std::lock_guard lock(*m_overlay_manager);

		const areau display_area = { 0, 0, target.width(), target.height() };
		for (const auto& view : m_overlay_manager->get_views())
		{
			const areau render_area = view->use_window_space ? display_area : area;
			ui_renderer->run(*m_current_command_buffer, render_area, target, m_texture_upload_buffer_ring_info, *view.get());
		}
	};

	// Screenshots / recording are captured before the drawable is acquired so that the hard sync does not delay it
	if (image_to_flip && need_media_capture)
	{
		const u32 bytes_per_pixel = (image_to_flip->format() == MTL::PixelFormatRGBA16Float) ? 8 : 4;

		if (bytes_per_pixel != 4)
		{
			rsx_log.error("Metal: screenshots and recording of 16-bit float display buffers are not supported");
		}
		else
		{
			const usz sshot_size = buffer_height * buffer_width * 4;
			mtl::buffer sshot_buf(*m_device, utils::align(sshot_size, 0x100000), mtl::memory_location::host_visible, "screenshot buffer");

			mtl::buffer_image_copy copy_info{};
			copy_info.buffer_offset = 0;
			copy_info.buffer_row_length = 0;
			copy_info.buffer_image_height = 0;
			copy_info.aspect = mtl::aspect_color;
			copy_info.base_layer = 0;
			copy_info.layer_count = 1;
			copy_info.mip_level = 0;
			copy_info.image_offset = MTL::Origin(0, 0, 0);
			copy_info.image_extent = MTL::Size(buffer_width, buffer_height, 1);

			mtl::image* image_to_copy = image_to_flip;

			if (g_cfg.video.record_with_overlays && has_overlay)
			{
				if (m_overlay_recording_img)
				{
					// Validate
					if (m_overlay_recording_img->format() != image_to_flip->format() ||
						m_overlay_recording_img->width() != image_to_flip->width() ||
						m_overlay_recording_img->height() != image_to_flip->height())
					{
						// Dispose correctly
						mtl::get_resource_manager()->dispose(m_overlay_recording_img);
					}
				}

				if (!m_overlay_recording_img)
				{
					mtl::image_create_info create_info{};
					create_info.type = MTL::TextureType2D;
					create_info.format = image_to_flip->format();
					create_info.width = image_to_flip->width();
					create_info.height = image_to_flip->height();
					create_info.usage = MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget;
					create_info.format_class = RSX_FORMAT_CLASS_COLOR;

					m_overlay_recording_img = std::make_unique<mtl::viewable_image>(*m_device, create_info);
					m_overlay_recording_img->set_debug_name("overlay recording image");
				}

				const areai rect = areai(0, 0, buffer_width, buffer_height);
				mtl::copy_image(*m_current_command_buffer, image_to_flip, m_overlay_recording_img.get(), rect, rect);

				render_overlays(mtl::overlay_target(m_overlay_recording_img.get()), areau(rect));

				image_to_copy = m_overlay_recording_img.get();
			}

			mtl::copy_image_to_buffer(*m_current_command_buffer, image_to_copy, &sshot_buf, copy_info);

			flush_command_queue(true);
			const auto src = sshot_buf.map(0);
			std::vector<u8> sshot_frame(sshot_size);
			std::memcpy(sshot_frame.data(), src, sshot_size);

			const bool is_bgra = image_to_copy->format() == MTL::PixelFormatBGRA8Unorm;

			if (user_asked_for_screenshot)
			{
				m_frame->take_screenshot(std::move(sshot_frame), buffer_width, buffer_height, is_bgra);
			}
			else
			{
				m_frame->present_frame(std::move(sshot_frame), buffer_width * 4, buffer_width, buffer_height, is_bgra);
			}
		}
	}

	if (!image_to_flip && !has_overlay && !g_cfg.video.debug_overlay)
	{
		// Nothing to show (no valid display buffer, e.g. while booting). Keep the previous image on screen instead of
		// presenting a black frame.
		mini_flip(false);
		return;
	}

	// Output scaling, set up before the drawable is acquired so that slow work (MetalFX kernels, mode switches) never
	// runs while a drawable is held. MetalFX builds its scalers on a worker thread (bilinear until ready) and keeps them
	// across resizes; the upscaler handles output size changes itself.
	const output_scaling_mode output_scaling = g_cfg.video.output_scaling.get();

	if (!m_upscaler || m_output_scaling != output_scaling)
	{
		m_output_scaling = output_scaling;
		m_upscaler = mtl::create_upscaler(m_output_scaling);
	}

	if (image_to_flip)
	{
		m_upscaler->prepare();
	}

	// Prepare surface for new frame. nextDrawable blocks until one is available (1 second timeout).
	ensure(!m_current_frame->drawable);
	ensure(m_current_frame->swap_command_buffer == nullptr);

	// Graphics self-check (MTLFrameInspector, opt-in: "Graphics Self-Check"): read back the checks the GPU has finished,
	// then record this frame's (a few small compute dispatches every 15th/30th frame, nothing waits for them, but they
	// end the frame's encoders and add a compute pass plus a surface-store walk to the present path)
	if (g_cfg.video.graphics_self_check && !m_frame_inspector)
	{
		m_frame_inspector = std::make_unique<mtl::frame_inspector>();
	}

	if (m_frame_inspector)
	{
		m_frame_inspector->poll();
	}

	if (m_frame_inspector && info.emu_flip && image_to_flip)
	{
		std::vector<mtl::frame_inspector::target_candidate> targets;

		if (m_frame_inspector->wants_targets())
		{
			// Float render targets written since the previous check (where shader math lands unclamped)
			const u64 since = m_frame_inspector_tag;
			m_rtts.for_each_color_surface([&](mtl::render_target* rtt)
			{
				if (!rtt || !rtt->value || rtt->last_use_tag <= since || !mtl::frame_inspector::is_float_format(rtt->format()))
				{
					return;
				}

				// A transfer wrote the single-sample image last (pending unresolve): that one holds the current data
				MTL::Texture* texture = rtt->value;
				if (rtt->samples() > 1 && (rtt->msaa_flags & rsx::surface_state_flags::require_unresolve) && rtt->resolve_surface)
				{
					texture = rtt->resolve_surface->value;
				}

				targets.push_back({ .texture = texture, .address = rtt->base_addr });
			});

			m_frame_inspector_tag = rsx::get_shared_tag();
		}

		m_frame_inspector->on_flip(*m_current_command_buffer, image_to_flip->value, present_address, info.stats.draw_calls, targets);
	}

	// Submit the frame's work so far: the GPU works on it while nextDrawable may block. Nothing on the main queue waits
	// for the drawable (see present()).
	flush_command_queue();

	// nextDrawable blocks without serving anything: hand out what the guest may be waiting for first
	serve_guest_during_display_wait();

	const u64 acquire_start = get_system_time();
	auto drawable = m_metal_layer->nextDrawable();
	m_present_pacing.blocked_time += get_system_time() - acquire_start;

	if (!drawable)
	{
		// The previous frame stays on screen
		rsx_log.warning("Metal: nextDrawable timed out. Frame skipped.");
		mini_flip(false);
		return;
	}

	drawable->retain();
	m_current_frame->drawable = drawable;

	MTL::Texture* drawable_texture = drawable->texture();
	const sizeu target_size = { static_cast<u32>(drawable_texture->width()), static_cast<u32>(drawable_texture->height()) };

	// The frame's output image. The present passes below render into it on the main queue; present() copies it into
	// the drawable on the present queue. Same size and format as the drawable. The frame context is idle here (its
	// previous present list has completed), so the image can be reused or replaced.
	auto& present_image = m_current_frame->present_image;
	if (!present_image ||
		present_image->width() != target_size.width ||
		present_image->height() != target_size.height ||
		present_image->format() != drawable_texture->pixelFormat())
	{
		if (present_image)
		{
			mtl::get_resource_manager()->dispose(present_image);
		}

		mtl::image_create_info create_info{};
		create_info.type = MTL::TextureType2D;
		create_info.format = drawable_texture->pixelFormat();
		create_info.width = target_size.width;
		create_info.height = target_size.height;
		create_info.usage = MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget;
		create_info.format_class = RSX_FORMAT_CLASS_COLOR;

		present_image = std::make_unique<mtl::viewable_image>(*m_device, create_info);
		present_image->set_debug_name("present image");
	}

	MTL::Texture* target_texture = present_image->value;
	const mtl::overlay_target target(target_texture);

	// Calculate output dimensions. Done after acquisition since the layer may have been resized.
	areai aspect_ratio;
	if (!g_cfg.video.stretch_to_display_area)
	{
		const auto converted = avconfig.aspect_convert_region({ buffer_width, buffer_height }, target_size);
		aspect_ratio = static_cast<areai>(converted);
	}
	else
	{
		aspect_ratio = { 0, 0, s32(target_size.width), s32(target_size.height) };
	}

	// The window background must be cleared to black (the present image holds an older frame, or nothing yet)
	const bool needs_letterbox_clear = !image_to_flip || aspect_ratio.x1 || aspect_ratio.y1 ||
		static_cast<u32>(aspect_ratio.x2) < target_size.width || static_cast<u32>(aspect_ratio.y2) < target_size.height;

	const bool use_full_rgb_range_output = g_cfg.video.full_rgb_range_output.get();
	const bool use_calibration_pass = image_to_flip &&
		(!use_full_rgb_range_output || !rsx::fcmp(avconfig.gamma, 1.f) || avconfig.stereo_enabled);

	if (needs_letterbox_clear && !image_to_flip)
	{
		// Nothing to draw: clear on its own. Otherwise the final pass (calibration pass or upscaler draw) clears on load.
		mtl::clear_color_texture(*m_current_command_buffer, target_texture, target_size.width, target_size.height, MTL::ClearColor::Make(0., 0., 0., 1.));
	}

	if (image_to_flip)
	{
		mtl::note_present(m_output_scaling == output_scaling_mode::fsr ? "metalfx" :
			m_output_scaling == output_scaling_mode::bilinear ? "bilinear" : "nearest");

		areai src_area = { 0, 0, s32(buffer_width), s32(buffer_height) };


		if (use_calibration_pass) [[unlikely]]
		{
			rsx::simple_array<mtl::viewable_image*> calibration_src;
			if (image_to_flip) calibration_src.push_back(image_to_flip);
			if (image_to_flip2) calibration_src.push_back(image_to_flip2);

			if (m_output_scaling == output_scaling_mode::fsr && !avconfig.stereo_enabled) // MetalFX: no stereo 3D (bilinear)
			{
				// Run upscaling pass before the rest of the output effects pipeline
				// This can be done with all upscalers but we already get bilinear upscaling for free if we just out the filters directly
				const areai dst_area = { 0, 0, aspect_ratio.width(), aspect_ratio.height() };

				for (unsigned i = 0; i < calibration_src.size(); ++i)
				{
					const rsx::flags32_t mode = (i == 0) ? UPSCALE_LEFT_VIEW : UPSCALE_RIGHT_VIEW;
					calibration_src[i] = m_upscaler->scale_output(*m_current_command_buffer, calibration_src[i], nullptr, src_area, dst_area, mode);
				}
			}

			// Letterbox clear folded into the pass (loadAction Clear) instead of a separate clear pass
			mtl::overlay_target calibration_target = target;
			calibration_target.clear_color_on_load = needs_letterbox_clear;
			calibration_target.clear_color_value = { 0.f, 0.f, 0.f, 1.f };

			mtl::get_overlay_pass<mtl::video_out_calibration_pass>()->run(
				*m_current_command_buffer, areau(aspect_ratio), calibration_target, calibration_src,
				avconfig.gamma, !use_full_rgb_range_output, avconfig.stereo_enabled);
		}
		else
		{
			// Scaled draw into the present image (Metal has no image blit). MetalFX (+ RCAS) upscales first when active.
			m_upscaler->scale_output(*m_current_command_buffer, image_to_flip, target_texture, src_area, aspect_ratio,
				UPSCALE_AND_COMMIT | UPSCALE_DEFAULT_VIEW | (needs_letterbox_clear ? UPSCALE_CLEAR_TARGET : 0));
		}
	}

	if (g_cfg.video.debug_overlay || has_overlay)
	{
		render_overlays(target, areau(aspect_ratio));

		if (g_cfg.video.debug_overlay)
		{
			m_render_pass_splits = mtl::g_feedback_loop_pass_splits.exchange(0);

			const auto num_dirty_textures = m_texture_cache.get_unreleased_textures_count();
			const auto texture_memory_size = m_texture_cache.get_texture_memory_in_use() / (1024 * 1024);
			const auto tmp_texture_memory_size = m_texture_cache.get_temporary_memory_in_use() / (1024 * 1024);
			const auto num_flushes = m_texture_cache.get_num_flush_requests();
			const auto num_mispredict = m_texture_cache.get_num_cache_mispredictions();
			const auto num_speculate = m_texture_cache.get_num_cache_speculative_writes();
			const auto num_misses = m_texture_cache.get_num_cache_misses();
			const auto num_unavoidable = m_texture_cache.get_num_unavoidable_hard_faults();
			const auto cache_miss_ratio = static_cast<u32>(ceil(m_texture_cache.get_cache_miss_ratio() * 100));
			const auto num_texture_upload = m_texture_cache.get_texture_upload_calls_this_frame();
			const auto num_texture_upload_miss = m_texture_cache.get_texture_upload_misses_this_frame();
			const auto texture_upload_miss_ratio = m_texture_cache.get_texture_upload_miss_percentage();
			const auto texture_copies_ellided = m_texture_cache.get_texture_copies_ellided_this_frame();
			const auto vertex_cache_hit_count = (info.stats.vertex_cache_request_count - info.stats.vertex_cache_miss_count);
			const auto vertex_cache_hit_ratio = info.stats.vertex_cache_request_count
				? (vertex_cache_hit_count * 100) / info.stats.vertex_cache_request_count
				: 0;
			const auto program_cache_lookups = info.stats.program_cache_lookups_total;
			const auto program_cache_ellided = info.stats.program_cache_lookups_ellided;
			const auto program_cache_ellision_rate = program_cache_lookups
				? (program_cache_ellided * 100) / program_cache_lookups
				: 0;

			rsx::overlays::set_debug_overlay_text(fmt::format(
				"Internal Resolution:      %s\n"
				"RSX Load:                 %3d%%\n"
				"draw calls: %17d\n"
				"submits: %20d\n"
				"render pass splits: %9d\n"
				"draw call setup: %12dus\n"
				"vertex upload time: %9dus\n"
				"texture upload time: %8dus\n"
				"draw call execution: %8dus\n"
				"submit and flip: %12dus\n"
				"Unreleased textures: %8d\n"
				"Texture cache memory: %7dM\n"
				"Temporary texture memory: %3dM\n"
				"Flush requests: %13d  = %2d (%3d%%) hard faults, %2d unavoidable, %2d misprediction(s), %2d speculation(s)\n"
				"Texture uploads: %12u (%u from CPU - %02u%%, %u copies avoided)\n"
				"Vertex cache hits: %10u/%u (%u%%)\n"
				"Program cache lookup ellision: %u/%u (%u%%)\n"
				"Present pacing: %.1f Hz %s %s, guest frame %.2f ms, %u refresh(es)/frame, min duration %.2f ms",
				info.stats.framebuffer_stats.to_string(resolution_scaling_config, !backend_config.supports_hw_msaa),
				get_load(), info.stats.draw_calls, info.stats.submit_count, m_render_pass_splits, info.stats.setup_time, info.stats.vertex_upload_time,
				info.stats.textures_upload_time, info.stats.draw_exec_time, info.stats.flip_time,
				num_dirty_textures, texture_memory_size, tmp_texture_memory_size,
				num_flushes, num_misses, cache_miss_ratio, num_unavoidable, num_mispredict, num_speculate,
				num_texture_upload, num_texture_upload_miss, texture_upload_miss_ratio, texture_copies_ellided,
				vertex_cache_hit_count, info.stats.vertex_cache_request_count, vertex_cache_hit_ratio,
				program_cache_ellided, program_cache_lookups, program_cache_ellision_rate,
				1. / m_present_pacing.refresh_interval, m_present_pacing.variable_refresh ? "variable" : "fixed",
				m_present_pacing.fullscreen ? "fullscreen" : "windowed", get_guest_frame_interval() * 1000.,
				m_present_pacing.slot_refreshes, m_present_pacing.min_duration * 1000.)
			);
		}
	}

	queue_swap_request();

	m_frame_stats.flip_time = m_profiler.duration();

	m_frame->flip(m_context);
	rsx::thread::flip(info);

	// Data sync
	const rsx::surface_scaling_config_t active_res_scaling_config =
	{
		.scale_percent = static_cast<u16>(g_cfg.video.resolution_scale_percent),
		.min_scalable_dimension = static_cast<u16>(g_cfg.video.min_scalable_dimension),
	};

	if (active_res_scaling_config != this->resolution_scaling_config)
	{
		// First, try to reclaim any memory since the res scale upgrade is so memory intensive
		if (const auto severity = mtl::vmm_determine_memory_load_severity();
			severity > rsx::problem_severity::low && m_rtts.handle_memory_pressure(*m_current_command_buffer, severity))
		{
			flush_command_queue(true);
		}

		// Then apply the change
		m_rtts.sync_scaling_config(*m_current_command_buffer, active_res_scaling_config);
		this->resolution_scaling_config = active_res_scaling_config;

		// Finally reclaim any unused resources
		if (const auto severity = mtl::vmm_determine_memory_load_severity();
			severity > rsx::problem_severity::low && m_rtts.handle_memory_pressure(*m_current_command_buffer, severity))
		{
			flush_command_queue(true);
		}
	}
}
