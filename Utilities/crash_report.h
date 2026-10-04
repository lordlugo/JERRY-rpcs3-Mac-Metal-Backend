#pragma once

// Detailed crash reporting: if the emulator or a game crashes, the log file gets
// a full report (what, where, thread, addresses, backtrace) BEFORE any message is
// shown to the user, and the message itself points at the report.
//
// Two crash classes are covered:
//  - Orderly fatals (ensure/verify/throw): thread_ctrl::emergency_exit() logs the
//    enriched report through the normal channels and flushes, then the dialog shows.
//  - Native crashes (SIGSEGV/SIGBUS/SIGILL/SIGFPE/SIGABRT, Windows VEH): a signal-safe
//    handler appends the report straight to RPCS3.log with raw syscalls (no locks,
//    no heap), writes a one-page summary for the next-launch notice, and re-raises
//    so the system reporter still fires. No dialog can be shown in-handler.

#include <string>
#include <vector>

#include <util/types.hpp>

namespace utils::crash_report
{
	// All the facts one crash report is built from. Plain data: the formatter below
	// is pure and unit-tested, and both the orderly and the signal-safe paths fill it.
	struct context
	{
		// "native" for caught signals/structured exceptions, "fatal" for ensure/throw.
		const char* kind = "fatal";

		// Native crashes only: signal name/number and the faulting address (si_addr).
		const char* signal_name = nullptr;
		int signal_number = 0;
		u64 fault_address = 0;
		bool has_fault_address = false;

		u64 thread_id = 0;
		std::string thread_name;

		// The ensure/verify message (orderly fatals).
		std::string reason;

		std::string build_version;

		// Emulator-provided lines (RSX FIFO GET/PUT, PPU state, RAM usage...).
		std::string extra;

		// Symbolized backtrace, most-recent call first.
		std::vector<std::string> backtrace;
	};

	// Render the full log-file report (sections: header, reason, thread, emulator
	// context, backtrace, footer). Pure function, covered by unit tests.
	std::string format_report(const context& ctx);

	// Short user-facing summary for dialogs: what failed, where, and where the full
	// report lives. Pure function, covered by unit tests.
	std::string format_summary(const context& ctx, std::string_view log_path);

	// Remember where crash files live. Called once at startup, after the log file
	// exists; copies everything into static storage so the signal-safe recorder
	// below never touches the heap.
	//   log_path:     RPCS3.log (shown in dialogs so users can find the report).
	//   summary_path: one-page summary for the next-launch notice ("last_crash.txt").
	//   build_version: e.g. rpcs3::get_verbose_version().
	void set_crash_files(const std::string& log_path, const std::string& summary_path, const std::string& build_version);

	// Called on entry to the orderly fatal path (emergency_exit, before any dialog
	// or abort): a signal raised from there (e.g. the abort after the dialog) must
	// not produce a second, misleading native report.
	void notify_orderly_fatal();

	// True after notify_orderly_fatal(). The installed signal handlers consult it
	// to skip their summary write for orderly shutdowns.
	bool is_orderly_fatal();

	// Signal-safe: write the one-page summary for the next-launch notice with raw
	// syscalls only (no heap, no locks). Called from the installed native crash
	// handlers in Thread.cpp. Best-effort: silently does nothing without paths.
	void record_native_crash(const char* signal_name, int signal_number, u64 fault_address, bool has_fault_address);

	// Next-launch notice support (normal context, not signal-safe):
	// read the summary left by the signal handler; false when there is none.
	bool read_last_summary(const std::string& summary_path, std::string& out);

	// Remove a consumed summary file. Best-effort, returns success.
	bool clear_last_summary(const std::string& summary_path);

	// Write a summary file (used by the signal handler; also the unit-test seam for
	// the next-launch round trip). Best-effort, returns success.
	bool write_last_summary(const std::string& summary_path, std::string_view text);
}
