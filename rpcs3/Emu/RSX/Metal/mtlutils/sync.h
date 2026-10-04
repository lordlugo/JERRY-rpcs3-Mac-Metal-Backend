#pragma once

#include "mtl_api.h"
#include "device.h"
#include "util/atomic.hpp"

#include <array>
#include <string>

namespace mtl
{
	// Time CPU threads spent blocked in timeline::wait() on the timelines created with count_waits (telemetry for the
	// renderer's periodic statistics: GPU -> CPU synchronization stalls). Waits that did not block are not counted.
	struct cpu_wait_stats_t
	{
		struct bucket
		{
			u64 count = 0;
			u64 total_us = 0;
			u64 max_us = 0;
		};

		static constexpr u32 context_count = 8; // pass_context (commands.h)

		bucket renderer; // The RSX thread
		bucket others;   // Any other thread (e.g. access violation handlers flushing surfaces)
		std::array<bucket, context_count> renderer_by_context{}; // RSX thread waits by what it was doing (pass_context)
	};

	cpu_wait_stats_t get_cpu_wait_stats_and_reset();

	// The calling thread's pass_context (commands.h) as an index, for the wait statistics
	u32 current_pass_context_index();

	// RSX thread waits outside any pass_context ("other"), by the FIFO method being executed when they happened
	// (NV method offset, e.g. 0x1d6c = back end semaphore release): "0x1d6c 3.10 ms in 12, ..." (most time first). Resets.
	std::string take_other_wait_sites(u32 frames);

	// Names what the calling thread is about to wait for (telemetry for take_other_wait_sites; innermost wins)
	class wait_site_scope
	{
		const char* m_previous;

	public:
		explicit wait_site_scope(const char* site);
		~wait_site_scope();

		wait_site_scope(const wait_site_scope&) = delete;
		wait_site_scope& operator=(const wait_site_scope&) = delete;
	};

	// A monotonically increasing GPU timeline backed by MTLSharedEvent (Metal's timeline semaphore).
	// Queues signal it after each committed batch; the CPU polls or waits on values.
	class timeline
	{
		MTL::SharedEvent* m_event = nullptr;
		atomic_t<u64> m_next_value{ 0 };
		bool m_count_waits = false;

	public:
		timeline() = default;
		~timeline();

		timeline(const timeline&) = delete;
		timeline& operator=(const timeline&) = delete;

		// count_waits: blocking waits are added to the CPU wait statistics (get_cpu_wait_stats_and_reset)
		void create(const render_device& dev, std::string_view label, bool count_waits = false);
		void destroy();

		MTL::SharedEvent* handle() const { return m_event; }

		// Reserve the next value and enqueue a signal of it on `queue` (after previously committed work).
		u64 signal(MTL4::CommandQueue* queue);

		// Enqueue a GPU-side wait on `queue` until the timeline reaches `value`.
		void gpu_wait(MTL4::CommandQueue* queue, u64 value) const;

		u64 completed_value() const { return m_event ? m_event->signaledValue() : 0; }
		u64 last_signaled_value() const { return m_next_value.load(); }
		bool is_complete(u64 value) const { return completed_value() >= value; }

		// CPU wait. timeout_us == 0 means wait forever. Returns false on timeout.
		bool wait(u64 value, u64 timeout_us = 0) const;
	};

	// Host-side "fence" for a single submitted batch.
	struct fence
	{
		const timeline* owner = nullptr;
		u64 value = 0;
		atomic_t<bool> flushed = false;

		void reset() { value = 0; flushed = false; }
		bool signaled() const { return !value || (owner && owner->is_complete(value)); }
		bool wait(u64 timeout_us = 0) const { return !value || !owner || owner->wait(value, timeout_us); }
	};
}
