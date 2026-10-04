#pragma once

// Built-in frame-timestep patch for Ben 10 Ultimate Alien - Cosmic Destruction
// (BLES01110 EU), which steps its simulation once per vsync flip
// with a fixed dt of 1/60 s. At 120 fps (frame limiter at 120 or following a
// 120 Hz display) the guest flips at 120 Hz and the game runs at double speed
// with unstable per-frame physics (jump distance varies). Capping the guest at
// 60 fps avoids the speedup but does not deliver 120 fps. The true-120fps fix
// is to halve the fixed timestep (1/60 -> 1/120) so 120 steps/s advance game
// time at the correct rate.
//
// The patch is applied ONLY when the emulator is configured for >60 fps with
// the default 60 Hz vblank (see ben10_should_patch_timestep): at 60 fps the
// original 1/60 steps are already correct, and scaling them would run the game
// at half speed. PPUModule.cpp computes the effective frame limit the same way
// the RSX frame limiter does (primary limit, then the second frame limit which
// can only lower it) and skips the patch with an explanatory log line when the
// configuration keeps the guest at 60 fps.
//
// Discovery method (no offline addresses): after the module loads, scan guest
// memory for aligned 1/60 words. Words in non-executable load segments are data
// by construction and are always safe to patch. Words in executable segments
// can be instructions that merely encode to the same bits (0x3C888889 is also
// `addis r4, r8, -0x7777`, and a static file scan matched 17 such instructions
// - confirmed by per-site logging on a live boot, where every supposed float
// site held branch/load/store opcodes instead). Executable-segment candidates
// are therefore patched ONLY when proven to be float data: the word must be
// the target of an `lfs`/`lfsu` float load relative to the module TOC, or be
// referenced by a data-segment word holding its address (e.g. a TOC entry).
// Bare instructions are never loaded as floats nor pointed to as data.
//
// This header is dependency-free on purpose: scanning, gating, evidence
// selection and verify/apply logic are pure and unit-testable
// (rpcs3/tests/test_game_patches.cpp). The emulator wires them to guest memory
// in PPUModule.cpp.
//
// SAFETY: application is two-phase (verify ALL discovered sites first, then
// write). Any site that does not hold the expected 1/60 bits (already patched
// values are accepted for re-run safety; anything else aborts) stops the whole
// patch with nothing written, so subsystems can never be scaled inconsistently.
// A sane upper bound on the match count aborts the patch: a flood of matches
// means the pattern is coincidental data, not timestep literals.

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace game_patches
{
// The load-time Ben 10 patch was replaced by the automatic timestep governor
// (PPUModule.cpp, game_patches_runtime_on_flip), which retimes any fixed-step
// game at run time using the scanners below. The Ben 10 helpers stay for the
// unit tests of the scanning/verify logic.
constexpr bool kBen10_120fps_enabled = true;

// 1/60 s and 1/120 s as big-endian IEEE-754 float bits (guest byte order).
constexpr std::uint32_t kDt60Bits = 0x3C888889;
constexpr std::uint32_t kDt120Bits = 0x3C088889;

// Sanity bound on discovered literal count. A real fixed-timestep game holds a
// handful of such literals; hundreds would mean the bit pattern is coincidental
// payload, and scaling it would corrupt unrelated data.
constexpr std::uint32_t kMaxDtSites = 64;

inline bool ben10_applies_to(std::string_view title_id)
{
	// Cosmic Destruction (EU). Note: BLUS30621 is WWE SmackDown vs. Raw 2011, not Ben 10.
	return title_id == "BLES01110";
}

// Effective guest flip cap in Hz, mirroring the RSX frame limiter: the primary
// limit applies first, then the second frame limit, which can only lower it
// (a value below 0.1 disables its effect). 0 means uncapped/unknown.
inline double ben10_effective_frame_limit(double primary_hz, double second_hz)
{
	if (second_hz >= 0.1 && (second_hz < primary_hz || primary_hz <= 0.))
	{
		return second_hz;
	}

	return primary_hz;
}

// True only when the guest can flip faster than 60 Hz while game timing still
// runs on the default 60 Hz vblank. At 60 fps the original 1/60 steps are
// correct and must be left alone; a non-default vblank rate already re-times
// the game itself, so the fixed-step scaling does not apply there either.
inline bool ben10_should_patch_timestep(double effective_limit_hz, double vblank_hz)
{
	return effective_limit_hz > 65. && vblank_hz >= 59.5 && vblank_hz <= 60.5;
}

// An executable-segment 1/60 candidate plus the evidence collected for it:
// data_refs counts data-segment words holding the candidate's address, and
// float_loads counts r2-relative `lfs`/`lfsu` loads targeting it.
struct dt_candidate
{
	std::uint32_t vaddr = 0;
	std::uint32_t data_refs = 0;
	std::uint32_t float_loads = 0;
};

struct scan_result
{
	std::uint32_t sites[kMaxDtSites]{};
	std::uint32_t count = 0;
	bool overflow = false; // True when matches exceeded kMaxDtSites: sites[] is then empty and must not be used.
};

// Scan one non-executable memory span for 4-byte-aligned 1/60 words.
// `bytes` is the span content in guest (big-endian) byte order, `base_vaddr` is
// the guest address of bytes[0]. Only absolute-aligned words are matched, so a
// caller-supplied base needs no pre-alignment. Returns matches in ascending
// order; on overflow returns count == 0 with overflow == true.
inline scan_result scan_dt_literals(const std::uint8_t* bytes, std::uint32_t size, std::uint32_t base_vaddr)
{
	scan_result result{};

	if (!bytes || size < 4)
	{
		return result;
	}

	// First offset that is 4-aligned in absolute (guest-address) terms.
	std::uint32_t offset = (4 - (base_vaddr & 3u)) & 3u;

	for (; offset + 4u <= size; offset += 4u)
	{
		// offset <= size - 4 <= 0xFFFFFFFB, so offset + 4 cannot wrap.
		const std::uint32_t word = (static_cast<std::uint32_t>(bytes[offset]) << 24) |
			(static_cast<std::uint32_t>(bytes[offset + 1]) << 16) |
			(static_cast<std::uint32_t>(bytes[offset + 2]) << 8) |
			static_cast<std::uint32_t>(bytes[offset + 3]);

		if (word != kDt60Bits)
		{
			continue;
		}

		if (offset > UINT32_MAX - base_vaddr)
		{
			break; // Guest address would wrap; cannot happen for real segments.
		}

		if (result.count == kMaxDtSites)
		{
			result.count = 0;
			result.overflow = true;
			return result;
		}

		result.sites[result.count++] = base_vaddr + offset;
	}

	return result;
}

// Find D-form float loads (`lfs`/`lfsu`) with RA == r2 whose target (toc plus
// sign-extended disp) equals `target`. A hit proves the target word is loaded
// as a float; bare instructions never are. Returns the loader addresses in
// ascending order; on overflow returns count == 0 with overflow == true.
inline scan_result find_r2_float_loads(const std::uint8_t* bytes, std::uint32_t size, std::uint32_t base_vaddr, std::uint32_t toc, std::uint32_t target)
{
	scan_result result{};

	if (!bytes || size < 4)
	{
		return result;
	}

	std::uint32_t offset = (4 - (base_vaddr & 3u)) & 3u;

	for (; offset + 4u <= size; offset += 4u)
	{
		const std::uint32_t word = (static_cast<std::uint32_t>(bytes[offset]) << 24) |
			(static_cast<std::uint32_t>(bytes[offset + 1]) << 16) |
			(static_cast<std::uint32_t>(bytes[offset + 2]) << 8) |
			static_cast<std::uint32_t>(bytes[offset + 3]);

		if (const std::uint32_t op = word >> 26; (op != 48u && op != 49u) || ((word >> 16) & 31u) != 2u)
		{
			continue;
		}

		const std::int32_t disp = static_cast<std::int32_t>(static_cast<std::int16_t>(word & 0xFFFFu));

		if (toc + static_cast<std::uint32_t>(disp) != target)
		{
			continue;
		}

		if (offset > UINT32_MAX - base_vaddr)
		{
			break;
		}

		if (result.count == kMaxDtSites)
		{
			result.count = 0;
			result.overflow = true;
			return result;
		}

		result.sites[result.count++] = base_vaddr + offset;
	}

	return result;
}

// Count 4-byte-aligned words in a span whose value equals `target` (a guest
// address). Used to cross-reference a code-pool candidate: data words holding
// its address (e.g. TOC entries) prove it is referenced as data, while bare
// instructions are never pointed to. Same alignment rules as scan_dt_literals.
inline std::uint32_t count_address_refs(const std::uint8_t* bytes, std::uint32_t size, std::uint32_t base_vaddr, std::uint32_t target)
{
	std::uint32_t refs = 0;

	if (!bytes || size < 4)
	{
		return refs;
	}

	std::uint32_t offset = (4 - (base_vaddr & 3u)) & 3u;

	for (; offset + 4u <= size; offset += 4u)
	{
		const std::uint32_t word = (static_cast<std::uint32_t>(bytes[offset]) << 24) |
			(static_cast<std::uint32_t>(bytes[offset + 1]) << 16) |
			(static_cast<std::uint32_t>(bytes[offset + 2]) << 8) |
			static_cast<std::uint32_t>(bytes[offset + 3]);

		refs += (word == target);
	}

	return refs;
}

// Keep only candidates proven to be float data (referenced as data or loaded
// as a float). Unreferenced candidates stay instructions until proven otherwise
// and are never patched. The output bound matches scan_result, so callers can
// merge the result with data-segment hits under one safety bound; more than
// kMaxDtSites proven inputs overflow to empty, same as the scanners.
inline scan_result select_proven_dt_sites(const dt_candidate* cands, std::uint32_t count)
{
	scan_result result{};

	if (!cands || !count)
	{
		return result;
	}

	for (std::uint32_t i = 0; i < count; i++)
	{
		if (!cands[i].data_refs && !cands[i].float_loads)
		{
			continue;
		}

		if (result.count == kMaxDtSites)
		{
			result.count = 0;
			result.overflow = true;
			return result;
		}

		result.sites[result.count++] = cands[i].vaddr;
	}

	return result;
}

struct apply_result
{
	std::uint32_t verified = 0; // Sites holding the expected 1/60 bits
	std::uint32_t applied = 0;  // Sites rewritten to 1/120 bits
	std::uint32_t already = 0;  // Sites already holding 1/120 bits (re-run safe)
};

// Two-phase patch over discovered sites: verify every site first, then write.
// Any unexpected content (different game version, unreadable address) aborts
// with applied == 0: nothing is ever partially scaled. ReadFn:
// u32 be_bits = read(vaddr, ok); WriteFn: write(vaddr, be_bits).
template <typename ReadFn, typename WriteFn>
apply_result apply_dt_sites(ReadFn&& read_be32, WriteFn&& write_be32, const std::uint32_t* sites, std::uint32_t count)
{
	apply_result result{};

	for (std::uint32_t i = 0; i < count; i++)
	{
		bool ok = false;
		const std::uint32_t cur = read_be32(sites[i], ok);

		if (!ok || (cur != kDt60Bits && cur != kDt120Bits))
		{
			// Wrong version or unreadable address: abort with nothing written.
			return apply_result{};
		}

		result.verified++;
		result.already += (cur == kDt120Bits);
	}

	for (std::uint32_t i = 0; i < count; i++)
	{
		bool ok = false;
		if (read_be32(sites[i], ok) == kDt60Bits && ok)
		{
			write_be32(sites[i], kDt120Bits);
			result.applied++;
		}
	}

	return result;
}
// ---------------------------------------------------------------------------
// Fast single-pass discovery (used by PPUModule.cpp at boot).
//
// The per-candidate helpers above rescan every segment once per candidate:
// with ~9 MB of code and up to 64 candidates that is ~600 MB of reads at boot.
// The functions below read every segment exactly once and resolve all
// candidates at the same time (sorted targets + binary search, with a min/max
// range check that rejects almost every word before the search).
// ---------------------------------------------------------------------------

// 1/60 s and 1/120 s as big-endian IEEE-754 double bits. Engines that keep the
// timestep in double precision store these instead of the float literals.
constexpr std::uint64_t kDt60Bits64 = 0x3F91111111111111ull;
constexpr std::uint64_t kDt120Bits64 = 0x3F81111111111111ull;

inline std::uint32_t load_be32(const std::uint8_t* p)
{
	return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) |
		(static_cast<std::uint32_t>(p[2]) << 8) | static_cast<std::uint32_t>(p[3]);
}

// Index of `value` in ascending `sorted[0..count)`, or -1.
inline int find_sorted(const std::uint32_t* sorted, std::uint32_t count, std::uint32_t value)
{
	std::uint32_t lo = 0, hi = count;

	while (lo < hi)
	{
		const std::uint32_t mid = lo + (hi - lo) / 2;

		if (sorted[mid] < value)
		{
			lo = mid + 1;
		}
		else
		{
			hi = mid;
		}
	}

	return lo < count && sorted[lo] == value ? static_cast<int>(lo) : -1;
}

// Scan one non-executable span for 8-byte-aligned (guest address) 1/60 doubles.
// Same contract as scan_dt_literals (ascending, overflow -> empty).
inline scan_result scan_dt64_literals(const std::uint8_t* bytes, std::uint32_t size, std::uint32_t base_vaddr)
{
	scan_result result{};

	if (!bytes || size < 8)
	{
		return result;
	}

	std::uint32_t offset = (8 - (base_vaddr & 7u)) & 7u;
	constexpr std::uint32_t hi_bits = static_cast<std::uint32_t>(kDt60Bits64 >> 32);
	constexpr std::uint32_t lo_bits = static_cast<std::uint32_t>(kDt60Bits64);

	for (; offset + 8u <= size; offset += 8u)
	{
		if (load_be32(bytes + offset) != hi_bits || load_be32(bytes + offset + 4) != lo_bits)
		{
			continue;
		}

		if (offset > UINT32_MAX - base_vaddr)
		{
			break;
		}

		if (result.count == kMaxDtSites)
		{
			result.count = 0;
			result.overflow = true;
			return result;
		}

		result.sites[result.count++] = base_vaddr + offset;
	}

	return result;
}

// One pass over a data span: for every aligned word equal to one of the sorted
// candidate addresses, increments refs[index]. Adds to refs (call per segment).
inline void accumulate_address_refs(const std::uint8_t* bytes, std::uint32_t size, std::uint32_t base_vaddr,
	const std::uint32_t* sorted_targets, std::uint32_t count, std::uint32_t* refs)
{
	if (!bytes || size < 4 || !sorted_targets || !count || !refs)
	{
		return;
	}

	const std::uint32_t lo = sorted_targets[0], hi = sorted_targets[count - 1];

	for (std::uint32_t offset = (4 - (base_vaddr & 3u)) & 3u; offset + 4u <= size; offset += 4u)
	{
		const std::uint32_t word = load_be32(bytes + offset);

		if (word < lo || word > hi)
		{
			continue;
		}

		if (const int idx = find_sorted(sorted_targets, count, word); idx >= 0)
		{
			refs[idx]++;
		}
	}
}

// One pass over a code span: every `lfs`/`lfsu`/`lfd`/`lfdu` with RA == r2 whose
// target (toc + disp) is one of the sorted candidates increments loads[index]
// and records the first loader address. Adds to the outputs (call per segment).
inline void accumulate_r2_float_loads(const std::uint8_t* bytes, std::uint32_t size, std::uint32_t base_vaddr, std::uint32_t toc,
	const std::uint32_t* sorted_targets, std::uint32_t count, std::uint32_t* loads, std::uint32_t* first_loader)
{
	if (!bytes || size < 4 || !sorted_targets || !count || !loads)
	{
		return;
	}

	for (std::uint32_t offset = (4 - (base_vaddr & 3u)) & 3u; offset + 4u <= size; offset += 4u)
	{
		const std::uint32_t word = load_be32(bytes + offset);

		// Primary opcode 48..51 (lfs, lfsu, lfd, lfdu) and RA == r2
		if ((word >> 28) != 0xCu || ((word >> 16) & 31u) != 2u)
		{
			continue;
		}

		const std::uint32_t target = toc + static_cast<std::uint32_t>(static_cast<std::int32_t>(static_cast<std::int16_t>(word & 0xFFFFu)));

		if (const int idx = find_sorted(sorted_targets, count, target); idx >= 0)
		{
			if (!loads[idx]++ && first_loader)
			{
				first_loader[idx] = base_vaddr + offset;
			}
		}
	}
}

struct apply_result_all
{
	std::uint32_t verified = 0;
	std::uint32_t applied = 0;
	std::uint32_t already = 0;
};

// Two-phase patch over float and double sites together: every site of both
// kinds is verified before anything is written, so the game can never end up
// with a mix of 1/60 and 1/120 steps. Doubles are read/written as two
// big-endian words. Same ReadFn/WriteFn contract as apply_dt_sites.
template <typename ReadFn, typename WriteFn>
apply_result_all apply_dt_sites_all(ReadFn&& read_be32, WriteFn&& write_be32,
	const std::uint32_t* f_sites, std::uint32_t f_count, const std::uint32_t* d_sites, std::uint32_t d_count)
{
	apply_result_all result{};
	constexpr std::uint32_t d60_hi = static_cast<std::uint32_t>(kDt60Bits64 >> 32), d60_lo = static_cast<std::uint32_t>(kDt60Bits64);
	constexpr std::uint32_t d120_hi = static_cast<std::uint32_t>(kDt120Bits64 >> 32), d120_lo = static_cast<std::uint32_t>(kDt120Bits64);

	for (std::uint32_t i = 0; i < f_count; i++)
	{
		bool ok = false;
		const std::uint32_t cur = read_be32(f_sites[i], ok);

		if (!ok || (cur != kDt60Bits && cur != kDt120Bits))
		{
			return {};
		}

		result.verified++;
		result.already += (cur == kDt120Bits);
	}

	for (std::uint32_t i = 0; i < d_count; i++)
	{
		bool ok1 = false, ok2 = false;
		const std::uint32_t hi = read_be32(d_sites[i], ok1);
		const std::uint32_t lo = read_be32(d_sites[i] + 4, ok2);
		const bool is60 = hi == d60_hi && lo == d60_lo;
		const bool is120 = hi == d120_hi && lo == d120_lo;

		if (!ok1 || !ok2 || (!is60 && !is120))
		{
			return {};
		}

		result.verified++;
		result.already += is120;
	}

	for (std::uint32_t i = 0; i < f_count; i++)
	{
		bool ok = false;
		if (read_be32(f_sites[i], ok) == kDt60Bits && ok)
		{
			write_be32(f_sites[i], kDt120Bits);
			result.applied++;
		}
	}

	for (std::uint32_t i = 0; i < d_count; i++)
	{
		bool ok = false;
		if (read_be32(d_sites[i], ok) == d60_hi && ok)
		{
			// Low word is identical for 1/60 and 1/120 (same mantissa): only the exponent word changes
			write_be32(d_sites[i], d120_hi);
			result.applied++;
		}
	}

	return result;
}

} // namespace game_patches
