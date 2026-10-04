#include <filesystem>
#include <gtest/gtest.h>

#include "Utilities/crash_report.h"

// The crash-report pipeline (format + summary-file round trip) must stay exact:
// dialogs and the next-launch notice render from these strings, and the log file
// report is the primary artifact of any crash investigation.
namespace
{
	utils::crash_report::context make_native_context()
	{
		utils::crash_report::context ctx{};
		ctx.kind = "native";
		ctx.signal_name = "SIGSEGV";
		ctx.signal_number = 11;
		ctx.fault_address = 0x449;
		ctx.has_fault_address = true;
		ctx.thread_id = 1234;
		ctx.thread_name = "rsx::thread";
		ctx.build_version = "test-build";
		ctx.extra = "RSX FIFO: GET=0x1234 PUT=0x5678\n";
		ctx.backtrace = { "frame_zero", "frame_one" };
		return ctx;
	}

	std::string unique_tmp_path(const char* name)
	{
		// Fixed names: the test removes strays first, and the test binary runs
		// its cases serially in one process.
		auto path = std::filesystem::temp_directory_path();
		path /= std::string("rpcs3_crash_test_") + name;
		return path.string();
	}
}

TEST(CrashReport, FullReportContainsEverySection)
{
	const std::string report = utils::crash_report::format_report(make_native_context());

	EXPECT_NE(report.find("RPCS3 CRASH REPORT"), std::string::npos);
	EXPECT_NE(report.find("SIGSEGV"), std::string::npos);
	EXPECT_NE(report.find("0x449"), std::string::npos);
	EXPECT_NE(report.find("1234"), std::string::npos);
	EXPECT_NE(report.find("rsx::thread"), std::string::npos);
	EXPECT_NE(report.find("test-build"), std::string::npos);
	EXPECT_NE(report.find("RSX FIFO: GET=0x1234 PUT=0x5678"), std::string::npos);
	EXPECT_NE(report.find("#0 frame_zero"), std::string::npos);
	EXPECT_NE(report.find("#1 frame_one"), std::string::npos);
	EXPECT_NE(report.find("END RPCS3 CRASH REPORT"), std::string::npos);
}

TEST(CrashReport, EmptyContextStillRendersSafely)
{
	// Guards the extra.back() edge and empty tag handling: must not crash and
	// must still frame the report with banner/footer.
	utils::crash_report::context ctx{};
	ctx.thread_id = 1;

	const std::string report = utils::crash_report::format_report(ctx);

	EXPECT_NE(report.find("RPCS3 CRASH REPORT"), std::string::npos);
	EXPECT_NE(report.find("END RPCS3 CRASH REPORT"), std::string::npos);
	EXPECT_EQ(report.find("Signal:"), std::string::npos);
	EXPECT_EQ(report.find("Backtrace"), std::string::npos);
}

TEST(CrashReport, SummaryNamesSignalAddressAndLog)
{
	const std::string summary = utils::crash_report::format_summary(make_native_context(), "/logs/RPCS3.log");

	EXPECT_NE(summary.find("SIGSEGV"), std::string::npos);
	EXPECT_NE(summary.find("0x449"), std::string::npos);
	EXPECT_NE(summary.find("rsx::thread"), std::string::npos);
	EXPECT_NE(summary.find("/logs/RPCS3.log"), std::string::npos);
}

TEST(CrashReport, SummaryFallsBackToReason)
{
	utils::crash_report::context ctx{};
	ctx.reason = "Something broke.";
	ctx.thread_id = 7;

	const std::string summary = utils::crash_report::format_summary(ctx, "");

	EXPECT_NE(summary.find("Something broke."), std::string::npos);
	EXPECT_EQ(summary.find("RPCS3.log"), std::string::npos);
}

TEST(CrashReport, SummaryFileRoundTrip)
{
	const std::string path = unique_tmp_path("summary.txt");
	::remove(path.c_str());

	std::string out;
	EXPECT_FALSE(utils::crash_report::read_last_summary(path, out));

	EXPECT_TRUE(utils::crash_report::write_last_summary(path, "The emulator crashed with SIGSEGV."));
	EXPECT_TRUE(utils::crash_report::read_last_summary(path, out));
	EXPECT_NE(out.find("SIGSEGV"), std::string::npos);

	EXPECT_TRUE(utils::crash_report::clear_last_summary(path));
	EXPECT_FALSE(utils::crash_report::read_last_summary(path, out));

	EXPECT_FALSE(utils::crash_report::write_last_summary("", "x"));
	EXPECT_FALSE(utils::crash_report::read_last_summary("", out));
}

TEST(CrashReport, NativeRecorderWritesSummaryAndLogCopy)
{
	// Exercises the real signal-handler seam (minus the async context): the same
	// function the installed handlers call must produce the next-launch summary
	// and a copy in the log file.
	const std::string summary_path = unique_tmp_path("last_crash.txt");
	const std::string log_path = unique_tmp_path("RPCS3.log");
	::remove(summary_path.c_str());
	::remove(log_path.c_str());

	utils::crash_report::set_crash_files(log_path, summary_path, "test-build");
	utils::crash_report::record_native_crash("SIGSEGV", 11, 0x449, true);

	std::string summary;
	EXPECT_TRUE(utils::crash_report::read_last_summary(summary_path, summary));
	EXPECT_NE(summary.find("SIGSEGV"), std::string::npos);
	EXPECT_NE(summary.find("0x449"), std::string::npos);

	std::string logged;
	EXPECT_TRUE(utils::crash_report::read_last_summary(log_path, logged));
	EXPECT_NE(logged.find("SIGSEGV"), std::string::npos);

	::remove(summary_path.c_str());
	::remove(log_path.c_str());
}
