#include <gtest/gtest.h>

#include "Emu/Cell/lv2/spu_boot_settle.h"

// Phase 2 of the SPU group-start boot barrier waits for each thread's first
// wait state (first poll). Compute-bound kernels (SVR 2011's Bink decode)
// execute for seconds without one: every group start burned the full 2 s
// timeout (~2 s stall every ~2.2 s, 56 of them in one session), freezing the
// game's SPURS dispatcher so the logo movie stuttered and the title/login
// screen never arrived. The tracker proceeds early on sustained execution
// while keeping the full timeout for stalled (compiling/booting) threads,
// which is what the GoW Ascension wedge protection needs.
namespace
{
	using tracker = lv2_spu::boot_settle_tracker;
	using verdict = tracker::verdict;
	using u64 = lv2_spu::u64;

	tracker::thread_snapshot live(u64 counter)
	{
		return tracker::thread_snapshot{false, false, counter};
	}

	tracker::thread_snapshot waiting()
	{
		return tracker::thread_snapshot{false, true, 0};
	}

	tracker::thread_snapshot stopped()
	{
		return tracker::thread_snapshot{true, false, 0};
	}

	TEST(SpuBootSettle, ImmediateSettleWhenAllWaiting)
	{
		tracker settle;
		const u64 counters[1]{0};
		settle.begin(1, 0, counters);

		const tracker::thread_snapshot states[1]{waiting()};
		EXPECT_EQ(settle.update(0, states), verdict::settled);
	}

	TEST(SpuBootSettle, StoppedThreadsDoNotBlockSettle)
	{
		tracker settle;
		const u64 counters[2]{0, 0};
		settle.begin(2, 0, counters);

		const tracker::thread_snapshot states[2]{stopped(), waiting()};
		EXPECT_EQ(settle.update(0, states), verdict::settled);
	}

	TEST(SpuBootSettle, SustainedExecutionProceedsWellBeforeTimeout)
	{
		tracker settle;
		const u64 counters[1]{100};
		settle.begin(1, 0, counters);

		u64 counter = 100;
		verdict last = verdict::keep_waiting;
		u64 break_at = 0;

		// Advance the (execution-only) counter every production poll
		// iteration (200 us), never wait.
		for (u64 now = 200; now <= tracker::timeout_us; now += 200)
		{
			const tracker::thread_snapshot states[1]{live(++counter)};
			last = settle.update(now, states);

			if (last != verdict::keep_waiting)
			{
				break_at = now;
				break;
			}
		}

		EXPECT_EQ(last, verdict::proceed_executing);

		// Early: past the init grace, far short of the 2 s stall.
		EXPECT_GE(break_at, tracker::grace_us);
		EXPECT_LT(break_at, tracker::timeout_us / 10);
	}

	TEST(SpuBootSettle, StalledThreadsWaitOutTheFullTimeout)
	{
		tracker settle;
		const u64 counters[1]{100};
		settle.begin(1, 0, counters);

		// Frozen counter (still compiling/booting), never waiting.
		verdict last = verdict::keep_waiting;

		for (u64 now = 200; now <= tracker::timeout_us; now += 200)
		{
			const tracker::thread_snapshot states[1]{live(100)};
			last = settle.update(now, states);

			if (last != verdict::keep_waiting)
			{
				break;
			}
		}

		// Must not proceed early: no execution observed, keep GoW protection.
		EXPECT_EQ(last, verdict::timed_out);
	}

	TEST(SpuBootSettle, BriefAdvanceThenStallDoesNotProceedEarly)
	{
		tracker settle;
		const u64 counters[1]{100};
		settle.begin(1, 0, counters);

		// One block at the first poll, then freeze in a compile gap: must
		// keep waiting, never proceed early.
		verdict last = verdict::keep_waiting;
		u64 counter = 100;

		for (u64 now = 200; now <= tracker::timeout_us; now += 200)
		{
			if (now <= 200)
			{
				counter++;
			}

			const tracker::thread_snapshot states[1]{live(counter)};
			last = settle.update(now, states);

			if (last != verdict::keep_waiting)
			{
				break;
			}
		}

		EXPECT_EQ(last, verdict::timed_out);
	}

	TEST(SpuBootSettle, MixedWaitingAndExecutingProceedsOnSustainedExecution)
	{
		tracker settle;
		const u64 counters[2]{0, 50};
		settle.begin(2, 0, counters);

		u64 counter = 50;
		verdict last = verdict::keep_waiting;

		for (u64 now = 200; now <= tracker::timeout_us; now += 200)
		{
			const tracker::thread_snapshot states[2]{waiting(), live(++counter)};
			last = settle.update(now, states);

			if (last != verdict::keep_waiting)
			{
				break;
			}
		}

		EXPECT_EQ(last, verdict::proceed_executing);
	}

	TEST(SpuBootSettle, LateWaitStillSettles)
	{
		tracker settle;
		const u64 counters[1]{100};
		settle.begin(1, 0, counters);

		// Execute briefly, then reach the first poll before the grace expires.
		verdict last = verdict::keep_waiting;

		for (u64 now = 200; now <= tracker::timeout_us; now += 200)
		{
			tracker::thread_snapshot state = now < 400 ? live(100 + now / 200) : waiting();
			last = settle.update(now, &state);

			if (last != verdict::keep_waiting)
			{
				break;
			}
		}

		EXPECT_EQ(last, verdict::settled);
	}
} // namespace
