#include <gtest/gtest.h>

#include "Emu/RSX/RSXOffload.h"

// The RSX thread must never spin forever waiting for the offloader: poll_offload_drain
// is the pure decision behind dma_manager::sync()'s bounded wait.
TEST(OffloadSync, DrainsWhenNothingIsQueued)
{
	using rsx::dma_manager;
	using offload_drain_poll = dma_manager::offload_drain_poll;
	constexpr auto poll_offload_drain = dma_manager::poll_offload_drain;

	EXPECT_EQ(poll_offload_drain(0, 0, 0, 0, 2'000'000), offload_drain_poll::drained);
	EXPECT_EQ(poll_offload_drain(5, 5, 100, 900'000, 2'000'000), offload_drain_poll::drained);
	EXPECT_EQ(poll_offload_drain(3, 7, 100, 900'000, 2'000'000), offload_drain_poll::drained);
}

TEST(OffloadSync, WaitsWithinBudget)
{
	using rsx::dma_manager;
	using offload_drain_poll = dma_manager::offload_drain_poll;
	constexpr auto poll_offload_drain = dma_manager::poll_offload_drain;

	EXPECT_EQ(poll_offload_drain(7, 3, 100, 200, 2'000'000), offload_drain_poll::keep_waiting);
	EXPECT_EQ(poll_offload_drain(7, 3, 100, 100 + 2'000'000, 2'000'000), offload_drain_poll::keep_waiting);
}

TEST(OffloadSync, TimesOutPastBudget)
{
	using rsx::dma_manager;
	using offload_drain_poll = dma_manager::offload_drain_poll;
	constexpr auto poll_offload_drain = dma_manager::poll_offload_drain;

	EXPECT_EQ(poll_offload_drain(7, 3, 100, 100 + 2'000'001, 2'000'000), offload_drain_poll::timed_out);
	EXPECT_EQ(poll_offload_drain(851, 3, 0, 60'000'000, 2'000'000), offload_drain_poll::timed_out);
}
