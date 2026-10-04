#pragma once

// Settle tracker for phase 2 of the SPU group-start boot barrier (see
// sys_spu_thread_group_start). Phase 1 proves every started thread executed
// its first block; phase 2 waits for each thread's first wait state, i.e.
// its first poll, so PPU submissions postdate it (GoW Ascension wedge).
//
// The wait flag alone cannot detect the first poll of compute-bound kernels:
// video decode (Bink), audio mix, and similar SPURS tasklets execute for
// seconds without touching a wait state. SVR 2011's logo movie wedged the
// game this way: every group start burned the full 2 s timeout (~2 s stall
// every ~2.2 s, 56 of them in one session), freezing the SPURS dispatcher so
// the movie stuttered and the title/login screen never arrived.
//
// Kernel init is short deterministic code (register setup, queue address,
// then poll); the only thing that stretches init-to-poll past milliseconds
// of wall time is LLVM (re)compilation, during which the thread does NOT
// execute. block_counter is incremented by JIT-emitted SPU code only, so any
// advance is genuine execution, never compiler activity. Therefore a thread
// that keeps executing block after block for ~100 ms of wall time is past
// init: its first poll is behind it (or it never polls by design). This
// tracker breaks the wait on such sustained execution while keeping the
// full timeout for stalled threads (still compiling/booting: no counter
// movement is the real hazard, and those keep the GoW protection intact).
//
// Dependency-free so unit tests can drive it without LV2 threads.

#include <array>
#include <cstdint>

namespace lv2_spu
{
	using u32 = std::uint32_t;
	using u64 = std::uint64_t;

	struct boot_settle_tracker
	{
		static constexpr u32 max_threads = 8;

		// Grace covering the kernel init tail past phase 1 (first block
		// executed: the thread is awake and running, so this is pure SPU
		// execution, microseconds for a dispatcher prologue), in
		// microseconds. Kept tiny on purpose: the game restarts SPURS groups
		// every frame (SVR 2011: 60 starts/s), so every microsecond here is
		// PPU stall at frame rate. SPURS kernels poll first by design (short
		// register setup, queue address, then poll), so sustained execution
		// past this bound is either past the first poll or a poll-free
		// compute kernel (video decode) whose reservation baseline was
		// established at boot. Compile or deschedule gaps show as no
		// advancement and keep waiting through the timeout below.
		static constexpr u64 grace_us = 500;

		// A thread counts as executing only while its counter advanced within
		// this trailing window. Compile and deschedule gaps are milliseconds
		// wide, so a gapped thread always looks stalled; executing threads
		// advance on every poll iteration.
		static constexpr u64 recency_us = 250;

		// Hard cap: stalled (compiling/booting) threads still wait this long.
		static constexpr u64 timeout_us = 2'000'000;

		enum class verdict
		{
			keep_waiting,
			settled,           // every thread waiting or stopped
			proceed_executing, // sustained execution past init, no wait in sight
			timed_out,
		};

		struct thread_snapshot
		{
			bool stopped = true;
			bool waiting = true;
			u64 counter = 0;
		};

		void begin(u32 count, u64 now_us, const u64* counters)
		{
			m_count = count > max_threads ? max_threads : count;
			m_start = now_us;

			for (u32 i = 0; i != m_count; i++)
			{
				m_last_counter[i] = counters[i];
				m_last_advance[i] = now_us;
				// Unseen until it advances inside this wait: a thread that
				// never moves after begin (compile gap from the first block
				// on) must not count as executing on recency alone. Required
				// now that the grace is shorter than the recency window.
				m_seen[i] = false;
			}
		}

		verdict update(u64 now_us, const thread_snapshot* states)
		{
			bool all_settled = true;
			bool sustained = (now_us - m_start) >= grace_us;

			for (u32 i = 0; i != m_count; i++)
			{
				if (states[i].stopped || states[i].waiting)
				{
					continue;
				}

				all_settled = false;

				if (states[i].counter != m_last_counter[i])
				{
					m_last_counter[i] = states[i].counter;
					m_last_advance[i] = now_us;
					m_seen[i] = true;
				}
				else if (!m_seen[i] || (now_us - m_last_advance[i]) >= recency_us)
				{
					// No execution observed (or none for a full window):
					// still booting or compiling (or wedged) — not
					// sustained execution.
					sustained = false;
				}
			}

			if (all_settled)
			{
				return verdict::settled;
			}

			if ((now_us - m_start) >= timeout_us)
			{
				return verdict::timed_out;
			}

			if (sustained)
			{
				return verdict::proceed_executing;
			}

			return verdict::keep_waiting;
		}

	private:
		u32 m_count = 0;
		u64 m_start = 0;
		std::array<u64, max_threads> m_last_counter{};
		std::array<u64, max_threads> m_last_advance{};
		std::array<bool, max_threads> m_seen{};
	};
} // namespace lv2_spu
