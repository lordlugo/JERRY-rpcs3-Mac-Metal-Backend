#pragma once

// Granular graphics-flow logging for the Metal backend.
//
// Goal: every graphical flow of a game (draws aside, which have their own
// counters) and every graphics error/issue must be explainable from RPCS3.log
// alone. Two mechanisms:
//
//  1. Flow counters + per-kernel trace lines. Each GPU dispatch outside the
//     draw passes (MSAA resolve/unresolve, blit/scale passes, compute kernels,
//     uploads, DMA readbacks, surface init/spill, depth copies, present calls)
//     bumps a counter and emits one rsx_log.trace line. Trace lines are free
//     when the trace channel is off (the log call early-outs internally), and
//     the counters are summarized per frame by graphics_log_report(), so a bug
//     report at default verbosity still shows *what the GPU did*.
//  2. Issue census. A log listener watches the RSX channel for warnings and
//     worse, deduplicates them, tags them with the presenting frame, and reports
//     the top offenders with counts from graphics_log_report(). The 100+ error
//     sites need no per-site changes; a newly firing error shows up automatically.
//
// Threading: the note_* helpers only bump atomic counters (safe from any
// thread; almost all call sites are the RSX thread). The collector's table is
// mutex-guarded because log messages can arrive from any thread.

namespace mtl
{
	// One counter per GPU flow outside the draw passes. Order matches
	// s_flow_names in MTLGraphicsLog.cpp.
	enum class gfx_flow : unsigned
	{
		resolve_color = 0,
		resolve_depth,
		resolve_depthstencil,
		unresolve_color,
		unresolve_depth,
		unresolve_depthstencil,
		blit_pass,
		scaled_copy,
		plain_copy,
		compute_detile,
		compute_gather,      // depth interleave/gather (readback packing)
		compute_shuffle,     // byteswap kernels
		compute_fconvert,    // float depth conversion kernels
		compute_deswizzle,   // texture upload deswizzle
		compute_other,
		upload_image,
		dma_readback,        // Write Color/Depth Buffers flushes to guest memory
		surface_read_init,   // surface initialized from guest memory (Read Color/Depth Buffers)
		surface_clear_init,  // surface cleared without a memory load
		surface_spill,
		surface_unspill,
		depth_copy,          // out-of-pass depth copy for shader reads of the depth attachment
		gather_build,        // texture gather images (re)built on the GPU
		gather_reuse,        // ... served from the reusable gather cache
		present_frame,
		count
	};

	// --- Flow recording (call from the dispatch sites) -----------------------------------------------

	void note_resolve(const char* kind, bool unresolve, unsigned w, unsigned h);
	void note_blit(unsigned dst_aspect, int output_type, int src_w, int src_h, int dst_w, int dst_h);
	void note_scaled_copy(int src_w, int src_h, int dst_w, int dst_h);
	void note_plain_copy();
	void note_compute(const char* task);
	void note_upload(unsigned w, unsigned h, unsigned layers);
	void note_readback(unsigned long long bytes);
	void note_surface_init(bool from_memory);
	void note_spill(bool unspill);
	void note_depth_copy();
	void note_gather(bool reused);
	void note_present(const char* upscaler);

	// --- Frame tracking ------------------------------------------------------------------------------

	// Call once per presented guest frame (MTLGSRender::flip). Returns the new frame serial.
	unsigned long long graphics_log_next_frame();
	unsigned long long graphics_log_current_frame();

	// --- Lifecycle/reporting -------------------------------------------------------------------------

	// Install the issue listener (once; intentionally never uninstalled, like gpu_stats_state).
	void graphics_log_install();

	// One notice block with the graphics-relevant configuration. Call once at renderer init.
	void graphics_log_boot_snapshot();

	// Periodic notice: per-frame flow rates since the last call plus any newly collected issues.
	// Resets the flow counters. `frames_in_window` is the presented-frame count of the window and
	// must be nonzero (matches the caller's stats window, which already guards on it).
	void graphics_log_report(unsigned long long frames_in_window);
}
