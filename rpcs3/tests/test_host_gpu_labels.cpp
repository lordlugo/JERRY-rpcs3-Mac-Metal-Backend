#include <gtest/gtest.h>

#include "Emu/RSX/Host/RSXDMAWriter.h"

// Protocol tests for the host GPU label event counters (rsx::host_gpu_context_t).
// The Metal label path (release_GCM_label / on_guest_texture_read / drain_label_queue)
// depends on exactly these transitions: acquire/release pairing, the completion
// predicates, and the shared monotonic counter. All logic here is header-inline.

namespace rsx
{
	TEST(HostGPULabels, FreshContextIsQuiescent)
	{
		host_gpu_context_t ctx{};
		EXPECT_EQ(ctx.magic, 0xCAFEBABE);
		EXPECT_TRUE(ctx.in_flight_commands_completed());
		EXPECT_TRUE(ctx.texture_loads_completed());
		EXPECT_FALSE(ctx.has_unflushed_texture_loads());
		EXPECT_FALSE(ctx.needs_label_release());
	}

	TEST(HostGPULabels, LabelAcquireReleaseCycle)
	{
		host_gpu_context_t ctx{};

		// CPU records a label write: acquire first (id must be nonzero and monotonic)
		const u64 id = ctx.on_label_acquire();
		EXPECT_EQ(id, 1);
		EXPECT_TRUE(ctx.needs_label_release());
		// Nothing released yet, and the GPU has not reported completion: still quiescent
		// from the drain's point of view (release2 == complete == 0)
		EXPECT_TRUE(ctx.in_flight_commands_completed());

		// CPU finished recording: release. Now the drain must wait for the GPU event.
		ctx.on_label_release();
		EXPECT_FALSE(ctx.needs_label_release());
		EXPECT_FALSE(ctx.in_flight_commands_completed());

		// GPU timeline catches up (commands_complete_event = release id): drain unblocks
		ctx.commands_complete_event = id;
		EXPECT_TRUE(ctx.in_flight_commands_completed());
	}

	TEST(HostGPULabels, TextureLoadTracking)
	{
		host_gpu_context_t ctx{};

		// Semaphore fast path requires all loads done: true when idle
		EXPECT_TRUE(ctx.texture_loads_completed());

		const u64 load = ctx.on_texture_load_acquire();
		EXPECT_EQ(load, 1);
		EXPECT_FALSE(ctx.texture_loads_completed());
		EXPECT_TRUE(ctx.has_unflushed_texture_loads());

		// GPU reports the load (texture_load_complete_event = request id)
		ctx.texture_load_complete_event = load;
		EXPECT_TRUE(ctx.texture_loads_completed());
		EXPECT_FALSE(ctx.has_unflushed_texture_loads());
	}

	TEST(HostGPULabels, SharedCounterIsMonotonicAcrossStreams)
	{
		host_gpu_context_t ctx{};

		const u64 load = ctx.on_texture_load_acquire();
		const u64 label = ctx.on_label_acquire();
		EXPECT_LT(load, label);

		// A load completing does not release the label stream and vice versa
		ctx.texture_load_complete_event = load;
		EXPECT_TRUE(ctx.texture_loads_completed());
		EXPECT_TRUE(ctx.needs_label_release());

		ctx.on_label_release();
		ctx.commands_complete_event = label;
		EXPECT_TRUE(ctx.in_flight_commands_completed());
	}

	TEST(HostGPULabels, DrainReturnsImmediatelyWhenQuiescent)
	{
		// drain_label_queue busy-waits only while work is in flight; a fresh writer
		// over zeroed memory must return without blocking (no GPU traffic here).
		alignas(64) u8 mem[sizeof(host_gpu_context_t)]{};
		RSXDMAWriter writer(mem);
		EXPECT_TRUE(writer.host_ctx()->in_flight_commands_completed());
		writer.drain_label_queue();

		// Queue machinery with no handlers registered just drops jobs on update()
		writer.enqueue(host_gpu_write_op_t{});
		writer.update();
		writer.drain_label_queue();
	}
}
