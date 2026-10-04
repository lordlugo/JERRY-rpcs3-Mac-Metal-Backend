#pragma once

#include <util/types.hpp>
#include "util/atomic.hpp"
#include "Emu/Cell/timers.hpp"

#include <array>

namespace rsx
{
	// Where the RSX thread and the guest CPU threads wait for each other. Cumulative (never reset): a renderer's periodic
	// telemetry reports the difference between two snapshots. RSX thread idle time is the sum of the RSX thread waits;
	// the rest of its time is work (command processing, draws, flips).
	enum class sync_wait : u32
	{
		fifo_empty,         // RSX thread: out of commands (FIFO empty, NOP, jump to self, fetch stalls): waits for the guest
		flip_semaphore,     // RSX thread: NV406E semaphore acquire of the flip label (cellGcmSetWaitFlip)
		semaphore,          // RSX thread: other NV406E semaphore acquires (waits for a PPU/SPU to release them)
		frame_limiter,      // RSX thread: frame limiter
		ppu_command_buffer, // PPU: cellGcmCallback, command buffer full, waiting for the RSX to leave the next segment
		ppu_flip_status,    // PPU: cellGcmGetFlipStatus found the flip pending (count: polls, time: first pending poll to done)

		count
	};

	struct sync_wait_snapshot
	{
		std::array<u64, static_cast<usz>(sync_wait::count)> time_us{};
		std::array<u64, static_cast<usz>(sync_wait::count)> count{};

		u64 time_of(sync_wait what) const { return time_us[static_cast<usz>(what)]; }
		u64 count_of(sync_wait what) const { return count[static_cast<usz>(what)]; }
	};

	struct sync_wait_stats
	{
		// One cache line per kind: written by different threads (the RSX thread, PPU threads polling in a loop)
		struct alignas(128) counter
		{
			atomic_t<u64> time_us{0};
			atomic_t<u64> count{0};
			atomic_t<u64> since{0}; // ppu_flip_status: get_system_time() of the first poll that found the flip pending
		};

		std::array<counter, static_cast<usz>(sync_wait::count)> counters{};

		// RSX thread only: time spent in flips run from do_local_task, which can happen inside a semaphore acquire wait
		// (the flip and its frame limiter wait are not time spent waiting for the semaphore)
		u64 rsx_async_flip_us = 0;

		void add(sync_wait what, u64 us)
		{
			auto& c = counters[static_cast<usz>(what)];
			c.time_us += us;
			c.count++;
		}

		void on_flip_status_poll(bool pending)
		{
			auto& c = counters[static_cast<usz>(sync_wait::ppu_flip_status)];

			if (pending)
			{
				c.count++;

				if (!c.since)
				{
					c.since.compare_and_swap(0, get_system_time());
				}
			}
			else if (c.since)
			{
				if (const u64 since = c.since.exchange(0))
				{
					c.time_us += get_system_time() - since;
				}
			}
		}

		sync_wait_snapshot snapshot() const
		{
			sync_wait_snapshot result;

			for (usz i = 0; i < counters.size(); i++)
			{
				result.time_us[i] = counters[i].time_us.load();
				result.count[i] = counters[i].count.load();
			}

			return result;
		}
	};

	inline sync_wait_stats g_sync_wait_stats;
}
