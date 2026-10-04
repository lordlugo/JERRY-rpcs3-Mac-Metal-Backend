#pragma once

#include "mtl_api.h"
#include "device.h"

namespace mtl
{
	class timeline;

	// Command memory telemetry (renderer's periodic resource line)
	struct command_allocator_stats_t
	{
		u64 allocated_bytes = 0; // allocatedSize() of every allocator alive, as last read (when handed back to the pool)
		u32 allocators = 0;      // Allocators alive (idle in the pool + recording + in flight)
		u32 peak_in_use = 0;     // Most allocators recording or in flight at once since the last report
		u32 created = 0;         // Created since the last report
		u32 released = 0;        // Released since the last report (idle, or ballooned by one large recording)
	};

	// MTL4CommandAllocators are shared by all command lists (Explore Metal 4 games: reset an allocator once the GPU
	// completed its work, reuse it for later encoding, release the ones you no longer need). An allocator keeps the
	// memory of its largest recording across reset(), so one allocator per command list made the ring of 128 lists
	// hold 128 high-water marks. A list takes an allocator when it begins recording and hands it back when it commits;
	// the allocator is reset and reused once the submission's timeline value has been reached. Only the allocators of
	// the recordings in flight exist, and the most recently used one is handed out first (its memory is already
	// allocated and warm). Allocators idle for 2 s are released, and once a second the largest idle allocator is
	// released if it is over 32 MiB and more than 4x the median size (an exceptional recording ballooned it).
	namespace command_allocators
	{
		// A reset allocator for one recording, owned by the caller until retire()/recycle()/discard()
		MTL4::CommandAllocator* acquire(const render_device& dev);

		// The recording made with `allocator` was committed; the allocator is reset and reused once `tl` reaches
		// `value`. Call on the thread that owns the allocator, after the commit, in commit order per timeline.
		void retire(MTL4::CommandAllocator* allocator, const timeline& tl, u64 value);

		// The recording made with `allocator` ended without being committed: the allocator is reusable right away
		void recycle(MTL4::CommandAllocator* allocator);

		// Releases an allocator that is still handed out (list destruction)
		void discard(MTL4::CommandAllocator* allocator);

		// Renderer teardown: every submission has completed (or was abandoned) and no list holds an allocator
		void release_all();

		command_allocator_stats_t get_stats_and_reset();
	}
}
