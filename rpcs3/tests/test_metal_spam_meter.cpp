#include <gtest/gtest.h>

#include "Emu/RSX/Metal/mtlutils/spam_meter.h"

// The CB-chain "out of free entries" error fired 1126 times in 60 s during the
// GoW prologue wedge, burying every other signal in the log. The meter keeps the
// first burst fully visible, then one summary per window; the running total lets
// each emitted line say how bad the episode is.
namespace
{
	TEST(MetalSpamMeter, FirstBurstPasses)
	{
		mtl::spam_meter meter;

		for (unsigned i = 0; i < mtl::spam_meter::burst; i++)
		{
			EXPECT_TRUE(meter.allow());
		}

		EXPECT_EQ(meter.count(), mtl::spam_meter::burst);
	}

	TEST(MetalSpamMeter, SuppressesAfterBurstUntilWindow)
	{
		mtl::spam_meter meter;

		for (unsigned i = 0; i < mtl::spam_meter::burst; i++)
		{
			meter.allow();
		}

		EXPECT_FALSE(meter.allow());
		EXPECT_EQ(meter.count(), mtl::spam_meter::burst + 1);
	}

	TEST(MetalSpamMeter, WindowBoundarySummarizes)
	{
		mtl::spam_meter meter;

		unsigned allowed = 0;
		for (unsigned i = 0; i < mtl::spam_meter::window + 1; i++)
		{
			allowed += meter.allow() ? 1 : 0;
		}

		// First burst (32) plus the window summary at n=1024.
		EXPECT_EQ(allowed, mtl::spam_meter::burst + 1);
		EXPECT_EQ(meter.count(), mtl::spam_meter::window + 1);
	}
}
