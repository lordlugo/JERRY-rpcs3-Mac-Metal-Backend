#include <gtest/gtest.h>

#include "Emu/RSX/RSXThread.h"
#include "Utilities/File.h"

// The stall tripwire names the game-side wait from these lines; the format must
// stay exact so the wedged thread, its wait site and its program counter are all
// visible in one log line.
TEST(RSXStallReport, WaitingThreadNamesFunctionAndCIA)
{
	const std::string line = rsx::format_stall_thread_line(
		"PPU", 0x1000000, "main_thread", true, 0x1234, "cellMutexLock");

	EXPECT_EQ(line, "PPU[0x1000000] \"main_thread\": waiting in cellMutexLock at CIA 0x1234");
}

TEST(RSXStallReport, RunningThreadWithoutFunction)
{
	const std::string line = rsx::format_stall_thread_line(
		"PPU", 0x1000007, "voice thread", false, 0xa1560c, nullptr);

	EXPECT_EQ(line, "PPU[0x1000007] \"voice thread\": running at CIA 0xa1560c");
}

TEST(RSXStallReport, EmptyFunctionRendersWithoutIn)
{
	const std::string line = rsx::format_stall_thread_line(
		"PPU", 0x1000001, "worker", true, 0x0, "");

	EXPECT_EQ(line, "PPU[0x1000001] \"worker\": waiting at CIA 0x0");
}

TEST(RSXStallReport, SpuLineUsesPcLabel)
{
	const std::string line = rsx::format_stall_thread_line(
		"SPU", 0x100, "BigCellSpursKernel0", false, 0x4fc8, nullptr, "PC");

	EXPECT_EQ(line, "SPU[0x100] \"BigCellSpursKernel0\": running at PC 0x4fc8");
}

// The sidecar is what survives a force-quit + relaunch: the frozen session's
// RPCS3.log is gone as soon as the app opens again, so the stall report must be
// appended (never truncated) to its own file with its own timestamp.
TEST(RSXStallReport, SidecarAppendsTimestampedEntries)
{
	const std::string path = fs::get_temp_dir() + "rsx_stall_sidecar_test.log";
	fs::remove_file(path);

	EXPECT_TRUE(rsx::append_stall_sidecar(path, "tripwire first"));
	EXPECT_TRUE(rsx::append_stall_sidecar(path, "tripwire second"));

	fs::file f(path, fs::read);
	ASSERT_TRUE(!!f);
	const std::string content = f.to_string();

	// Two entries, each with its own header, nothing overwritten.
	EXPECT_EQ(content.find("tripwire first"), content.rfind("tripwire first"));
	EXPECT_NE(content.find("tripwire second"), std::string::npos);
	EXPECT_TRUE(content.starts_with("===== RSX stall report ("));
	EXPECT_NE(content.find("===== RSX stall report (", 1), std::string::npos);

	fs::remove_file(path);
}

TEST(RSXStallReport, SidecarRejectsEmptyInput)
{
	EXPECT_FALSE(rsx::append_stall_sidecar("", "text"));
	EXPECT_FALSE(rsx::append_stall_sidecar(fs::get_temp_dir() + "rsx_stall_sidecar_test.log", ""));
	EXPECT_FALSE(rsx::append_stall_sidecar("/nonexistent-dir-xyz/stall.log", "text"));
}

TEST(RSXStallReport, SidecarPathLivesNextToTheLog)
{
	EXPECT_TRUE(rsx::stall_sidecar_path().ends_with("RSXStallReports.log"));
}

// The SPU wait detail separates the three wedge suspects without a debugger:
// pure lock-line wait (LR mask, empty MFC), DMA completion wait (TG/SN mask
// with queued MFC commands), PPU signal/mailbox wait (S1/S2/MB mask).
TEST(RSXStallReport, SpuWaitDetailNamesLockLineWait)
{
	EXPECT_EQ(rsx::format_spu_wait_detail(0x400, 0x0, 0, 0xd0001000), " evmask=0x400 evpend=0x0 mfc=0 raddr=0xd0001000 outmb=0 outintr=0");
}

TEST(RSXStallReport, SpuWaitDetailNamesDmaWait)
{
	EXPECT_EQ(rsx::format_spu_wait_detail(0x1, 0x0, 3, 0x0), " evmask=0x1 evpend=0x0 mfc=3 raddr=0x0 outmb=0 outintr=0");
}

TEST(RSXStallReport, SpuWaitDetailNamesPendingSignal)
{
	EXPECT_EQ(rsx::format_spu_wait_detail(0x200, 0x200, 0, 0x0), " evmask=0x200 evpend=0x200 mfc=0 raddr=0x0 outmb=0 outintr=0");
}

TEST(RSXStallReport, SpuWaitDetailNamesPostedMailbox)
{
	// A completion sitting in the SPU outbox (posted, never read) points at the
	// PPU interrupt-delivery bridge instead of the producer.
	EXPECT_EQ(rsx::format_spu_wait_detail(0x400, 0x0, 0, 0x3001ad00, 0, 1),
		" evmask=0x400 evpend=0x0 mfc=0 raddr=0x3001ad00 outmb=0 outintr=1");
}

// A group join that never returns is the signature wedge: the dump appends the joined
// group id (read from the parked thread's GPR3) so the report names a dead group vs the
// live kernel group. The formatter takes it as a trailing extra; absent extra renders
// exactly as before.
TEST(RSXStallReport, JoinLineCarriesGroupId)
{
	const std::string line = rsx::format_stall_thread_line(
		"PPU", 0x1000002, "BigSpursHdlr0", true, 0xa9948c, "sys_spu_thread_group_join", "CIA", "group=0x4000100");

	EXPECT_EQ(line, "PPU[0x1000002] \"BigSpursHdlr0\": waiting in sys_spu_thread_group_join at CIA 0xa9948c group=0x4000100");
}

TEST(RSXStallReport, NullExtraRendersAsBefore)
{
	const std::string line = rsx::format_stall_thread_line(
		"PPU", 0x1000002, "BigSpursHdlr0", true, 0xa9948c, "sys_spu_thread_group_join", "CIA", nullptr);

	EXPECT_EQ(line, "PPU[0x1000002] \"BigSpursHdlr0\": waiting in sys_spu_thread_group_join at CIA 0xa9948c");
}

// The dump decodes WHAT each parked thread waits for from its syscall argument
// registers: join target, sleep duration, queue/cond/mutex ids and timeouts.
// Unknown waits decode empty (the line renders as before).
TEST(RSXStallReport, WaitArgsDecodeJoinTarget)
{
	EXPECT_EQ(rsx::format_stall_wait_args("sys_spu_thread_group_join", 0x4000100, 0, 0, 0), "group=0x4000100");
}

TEST(RSXStallReport, WaitArgsDecodeUsleepPoll)
{
	// A 16 ms frame-pacing sleep: the tight poll loop signature.
	EXPECT_EQ(rsx::format_stall_wait_args("sys_timer_usleep", 16000, 0, 0, 0), "sleep=0x3e80");
}

TEST(RSXStallReport, WaitArgsDecodeQueueAndCondWaits)
{
	EXPECT_EQ(rsx::format_stall_wait_args("sys_event_queue_receive", 0x25, 0, 0, 0), "queue=0x25 timeout=0x0");
	EXPECT_EQ(rsx::format_stall_wait_args("_sys_lwcond_queue_wait", 0x61, 0x62, 0, 0), "cond=0x61 mutex=0x62 timeout=0x0");
	EXPECT_EQ(rsx::format_stall_wait_args("_sys_lwmutex_lock", 0x63, 0, 1000000, 0), "mutex=0x63 timeout=0xf4240");
	EXPECT_EQ(rsx::format_stall_wait_args("sys_event_flag_wait", 0x12, 0x1, 0x2, 0), "flag=0x12 pattern=0x1 timeout=0x0");
}

TEST(RSXStallReport, WaitArgsUnknownWaitDecodesEmpty)
{
	EXPECT_TRUE(rsx::format_stall_wait_args("cellMutexLock", 1, 2, 3, 4).empty());
	EXPECT_TRUE(rsx::format_stall_wait_args(nullptr, 1, 2, 3, 4).empty());
	EXPECT_TRUE(rsx::format_stall_wait_args("", 1, 2, 3, 4).empty());
}

// Poll-loop context: raw code words around the sleep call site (ascending from
// base) and the GPR file (r0 first) let the next report name the polled flag:
// the loop's load displacement plus the base register give the exact address.
TEST(RSXStallReport, CodeWordsRenderAscendingFromBase)
{
	const u32 words[] = { 0x80a30000u, 0x2c1b0000u, 0x40820008u };
	EXPECT_EQ(rsx::format_stall_code_words(0x3e7a20, words, 3), "code=0x3e7a20:[80a30000 2c1b0000 40820008]");
}

TEST(RSXStallReport, GprsRenderR0First)
{
	u64 gpr[32]{};
	gpr[2] = 0xd0001000;
	gpr[13] = 0x2c0;

	const std::string line = rsx::format_stall_gprs(gpr);
	EXPECT_TRUE(line.rfind("gpr=[0 0 d0001000 ", 0) == 0);
	EXPECT_NE(line.find(" 2c0 "), std::string::npos);
	EXPECT_EQ(line.back(), ']');
}

// The GoW prologue wedge, decoded from the live report: main's loop ends in
// `lhz r3,0(r28)` with r28=0x99aac4 (halfword flag), Edge's in `lwz r3,0(r30)`
// with r30=0x99ac18 (progress counter). The decoder must resolve both.
TEST(RSXStallReport, PollAddrDecodesMainLoop)
{
	const u32 words[] = { 0xfb010080u, 0xfae10078u, 0xfac10070u, 0x2c030000u, 0x41820180u, 0x3c60009au,
		0x7bbf1764u, 0x3063aac0u, 0x7f83f814u, 0x7b9c0020u, 0xa07c0000u, 0x2c030000u, 0x4182001cu,
		0x3960008du, 0x3860001eu, 0x44000002u };
	u64 gpr[32]{};
	gpr[28] = 0x99aac4;
	gpr[31] = 0x4;

	u32 ea = 0;
	EXPECT_TRUE(rsx::decode_stall_poll_addr(words, 16, gpr, ea));
	EXPECT_EQ(ea, 0x99aac4u);
	EXPECT_EQ(rsx::format_stall_poll_value(ea, 0), "pollval=[0x99aac4]=0x0");
}

TEST(RSXStallReport, PollAddrDecodesEdgeLoop)
{
	const u32 words[] = { 0x30810078u, 0x7c65182eu, 0x48080221u, 0x60000000u, 0x337b0001u, 0x807e0000u,
		0x7b7b0020u, 0x7c1b1840u, 0x4180ffb0u, 0x3c60009au, 0x9383ac14u, 0x2c1f0000u, 0x41820010u,
		0x3960008du, 0x38600320u, 0x44000002u };
	u64 gpr[32]{};
	gpr[30] = 0x99ac18;

	u32 ea = 0;
	EXPECT_TRUE(rsx::decode_stall_poll_addr(words, 16, gpr, ea));
	EXPECT_EQ(ea, 0x99ac18u);
}

TEST(RSXStallReport, PollAddrHandlesEdgeCases)
{
	u32 ea = 0;

	// No loads at all.
	const u32 noload[] = { 0x3960008du, 0x3860001eu, 0x44000002u };
	u64 gpr[32]{};
	EXPECT_FALSE(rsx::decode_stall_poll_addr(noload, 3, gpr, ea));
	EXPECT_FALSE(rsx::decode_stall_poll_addr(nullptr, 0, gpr, ea));

	// lwarx with indexed address.
	const u32 ll[] = { 0x7c601828u }; // lwarx r3,0,r3 (XO=20)
	u64 gpr2[32]{};
	gpr2[3] = 0xd0002000;
	EXPECT_TRUE(rsx::decode_stall_poll_addr(ll, 1, gpr2, ea));
	EXPECT_EQ(ea, 0xd0002000u);

	// lwz with RA=0 uses displacement only.
	const u32 abs[] = { 0x80601234u }; // lwz r3,0x1234(0)
	EXPECT_TRUE(rsx::decode_stall_poll_addr(abs, 1, gpr2, ea));
	EXPECT_EQ(ea, 0x1234u);
}

// Event-queue backlog lines: a parked receiver with backlog>0 means the wakeup
// never fired (delivery bug); the summary counts queues and backlogged queues.
TEST(RSXStallReport, EventQueueLineNamesBacklog)
{
	EXPECT_EQ(rsx::format_stall_event_queue(0x8d000002, 0x5350555253574b57ull, 0x0, 3),
		"EVQ[0x8d000002] name=0x5350555253574b57 key=0x0 backlog=3");
	EXPECT_EQ(rsx::format_stall_event_queue(0x8d000001, 0x0, 0x0, 0),
		"EVQ[0x8d000001] name=0x0 key=0x0 backlog=0");
}
