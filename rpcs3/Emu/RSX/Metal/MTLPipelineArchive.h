#pragma once

// Persistent Metal 4 pipeline archive: compiled pipeline binaries survive between sessions of a title, so repeat boots
// (shader cache preload) and effects seen in earlier sessions no longer pay for a full pipeline compile.
//
// How it works (see MTLPipelineArchive.cpp for the details and the reasoning):
//  - Every pipeline state (render and compute) is built by new_render_pipeline_state / new_compute_pipeline_state.
//    Those go through a "capturing" MTL4Compiler that has an MTL4PipelineDataSetSerializer attached, and pass the
//    archives written by earlier sessions as MTL4CompilerTaskOptions::lookupArchives, so the compiler reuses their
//    GPU binaries instead of compiling.
//  - A background thread periodically retires the capturing compiler (a new one takes over for later builds) and
//    writes the retired serializer with serializeAsArchiveAndFlushToURL (temp file + rename). Each serializer is
//    written exactly once, when no build uses it any more.
//  - Files live next to the title's RSX shader cache (removed with it by "Remove shader cache"):
//      <cache>/shaders_cache/metal_pipeline_archive/<fast|precise>/
//    (not inside pipelines/metal/v1.0: rsx::shaders_cache::load() deletes unknown files there and only creates raw/
//    when that directory does not exist yet).
//    identity.txt pins the OS build and GPU; any mismatch discards the archives (they would only miss).
//  - Render pipelines are flexible (MTLPipelineCompiler.cpp): the archives hold unspecialized pipelines and full-state
//    pipelines. Specializations are not archived (they have no task options, so no lookup, and are fast anyway). Each
//    archive file has a "<file>.keys" sidecar listing the keys of the full-state render pipelines it holds, so that a
//    later session builds those with full state (an archive hit) instead of specializing them.
//  - First lookups: at boot, the first builds with lookups (the shader interpreter's pipeline and the RSX thread's first
//    overlay pipeline) returned together only after 1.6 / 6.8 / 13.3 s with 26 / 44 / 64 MiB of archives, 1.5-2 s
//    without archives. So the archive thread makes the first lookup itself (a trivial compute pipeline, on a compiler of
//    its own); until it returns, builds compile without lookups (still recorded), no pipeline counts as archived, and the
//    shader cache preload waits (pipeline_archive_ready).
//  - Pipelines of functions specialized with function constants (only the shader interpreter's: its uber shader, by far
//    the largest function) are built without the archive: neither looked up nor recorded (use_archive = false).
//
// Disabled when the on-disk shader cache is disabled, when the title has no cache directory, or with the environment
// variable RPCS3_METAL_PIPELINE_ARCHIVE=0. Then pipelines are compiled with the device compiler exactly as before.

#include "mtlutils/mtl_api.h"

namespace mtl
{
	enum class render_pipeline_kind : u8
	{
		full_state,    // Every colour attachment configured: drawable as is
		unspecialized, // Colour attachment configuration unspecialized: only used to specialize from
	};

	// Renderer constructor, after the device exists and before any pipeline is built. Opens the archives of earlier
	// sessions and starts capturing. No-op when disabled.
	void initialize_pipeline_archive();

	// The shader cache preload has finished (the last cached pipeline was built in the background, or there was
	// nothing to preload) or was interrupted (the renderer stops first: `complete` = false). Writes everything built so
	// far in the background; if the preload was complete, that file supersedes the archives of earlier sessions.
	void on_pipeline_cache_preloaded(bool complete);

	// The RSX thread is exiting (MTLGSRender::on_exit, after the pipe compiler workers are gone). Starts writing what
	// is still pending in the background and returns immediately.
	void flush_pipeline_archive_async();

	// Stops the archive thread, waiting at most `timeout_ms` for a write in progress, and releases every Metal object
	// the archive owns. Called by render_device::destroy() (UI thread). Safe to call when never initialized.
	void shutdown_pipeline_archive(u32 timeout_ms = 3000);

	// Build a pipeline state. Owned (+1) result, or nullptr with `error` set (autoreleased) on failure. Thread-safe.
	// Uses the archive when enabled and `use_archive` (false: the pipeline's functions are specialized with function
	// constants), otherwise the device compiler without task options. Records the build time (record_compile_time).
	// `key` identifies the pipeline across sessions: a hash of everything the descriptor is derived from (0 = unknown).
	// The archive uses it to recognize a preload that recorded exactly the pipelines its newest file already holds, and
	// lists the keys of full-state render pipelines next to the file they are written to (is_full_state_archived).
	MTL::RenderPipelineState* new_render_pipeline_state(const MTL4::RenderPipelineDescriptor* descriptor, NS::Error** error, u64 key,
		render_pipeline_kind kind, bool use_archive);
	MTL::ComputePipelineState* new_compute_pipeline_state(const MTL4::ComputePipelineDescriptor* descriptor, NS::Error** error, u64 key,
		bool use_archive);

	// The archives of earlier sessions hold the full-state render pipeline with this key, so building it with
	// new_render_pipeline_state is a lookup rather than a compile. False while lookups are not ready. Thread-safe, no
	// Metal call.
	bool is_full_state_archived(u64 key);

	// Builds use the archives of earlier sessions for lookups (or there are none): their first lookup has returned. The
	// shader cache preload starts then (pipe_compiler::on_pipeline_archive_ready is called when it becomes true).
	bool pipeline_archive_ready();
}
