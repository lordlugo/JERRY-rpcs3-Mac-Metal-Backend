#include <gtest/gtest.h>

#include "Emu/Cell/GamePatches.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <vector>

// Built-in true-120fps timestep patch for Ben 10 Ultimate Alien: Cosmic
// Destruction (BLES01110 EU):
// the discovery (scan of non-executable memory for aligned 1/60 words) and the
// verify/apply logic are pure, so the safety properties can be tested without a
// running emulator.

namespace
{
	float bits_to_float(std::uint32_t bits)
	{
		float value = 0.f;
		std::memcpy(&value, &bits, sizeof(value));
		return value;
	}

	void put60(std::vector<std::uint8_t>& buf, std::size_t offset)
	{
		buf[offset] = 0x3C;
		buf[offset + 1] = 0x88;
		buf[offset + 2] = 0x88;
		buf[offset + 3] = 0x89;
	}
}

TEST(GamePatches, TimestepConstantsDecode)
{
	EXPECT_FLOAT_EQ(bits_to_float(game_patches::kDt60Bits), 1.f / 60.f);
	EXPECT_FLOAT_EQ(bits_to_float(game_patches::kDt120Bits), 1.f / 120.f);
	EXPECT_EQ(game_patches::kDt60Bits, 0x3C888889u);
	EXPECT_EQ(game_patches::kDt120Bits, 0x3C088889u);
}

TEST(GamePatches, TitleGate)
{
	// Cosmic Destruction (EU). BLUS30621 is WWE SmackDown vs. Raw 2011 (a frame-counting game that must not match).
	EXPECT_TRUE(game_patches::ben10_applies_to("BLES01110"));
	EXPECT_FALSE(game_patches::ben10_applies_to("BLUS30621"));
	EXPECT_FALSE(game_patches::ben10_applies_to(""));
	EXPECT_FALSE(game_patches::ben10_applies_to("BLES01110 "));
	EXPECT_FALSE(game_patches::ben10_applies_to("bles01110"));
	EXPECT_FALSE(game_patches::ben10_applies_to("BLUS30622"));
}

TEST(GamePatches, EffectiveLimitMirrorsFrameLimiter)
{
	// Primary limit applies first; the second limit can only lower it.
	EXPECT_DOUBLE_EQ(game_patches::ben10_effective_frame_limit(120., 0.), 120.);
	EXPECT_DOUBLE_EQ(game_patches::ben10_effective_frame_limit(60., 0.), 60.);
	EXPECT_DOUBLE_EQ(game_patches::ben10_effective_frame_limit(120., 60.), 60.);
	EXPECT_DOUBLE_EQ(game_patches::ben10_effective_frame_limit(60., 120.), 60.);
	EXPECT_DOUBLE_EQ(game_patches::ben10_effective_frame_limit(0., 120.), 120.);
	EXPECT_DOUBLE_EQ(game_patches::ben10_effective_frame_limit(0., 0.), 0.);
	// A second limit below 0.1 disables its effect.
	EXPECT_DOUBLE_EQ(game_patches::ben10_effective_frame_limit(120., 0.05), 120.);
}

TEST(GamePatches, PatchGatedOnHighFrameRate)
{
	// 120 fps at the default vblank: patch applies.
	EXPECT_TRUE(game_patches::ben10_should_patch_timestep(120., 60.));
	// 60 fps (Auto/vblank default, explicit 60, or lowered by second limit):
	// the original 1/60 steps are already correct, patch must stay off or the
	// game would run at half speed.
	EXPECT_FALSE(game_patches::ben10_should_patch_timestep(60., 60.));
	EXPECT_FALSE(game_patches::ben10_should_patch_timestep(30., 60.));
	EXPECT_FALSE(game_patches::ben10_should_patch_timestep(0., 60.));
	// Boundary: just above 60 still counts, 65 does not.
	EXPECT_TRUE(game_patches::ben10_should_patch_timestep(65.5, 60.));
	EXPECT_FALSE(game_patches::ben10_should_patch_timestep(65., 60.));
	// A non-default vblank already re-times the game itself.
	EXPECT_FALSE(game_patches::ben10_should_patch_timestep(120., 30.));
	EXPECT_FALSE(game_patches::ben10_should_patch_timestep(120., 120.));
}

TEST(GamePatches, SelectProvenDtSites)
{
	const game_patches::dt_candidate cands[] = {
		{0x1000, 2, 0}, // data referenced: proven
		{0x2000, 0, 1}, // float loaded: proven
		{0x3000, 0, 0}, // no evidence: stays an instruction, never patched
		{0x4000, 1, 3}, // both: proven once
	};

	const auto result = game_patches::select_proven_dt_sites(cands, 4);

	ASSERT_EQ(result.count, 3u);
	EXPECT_FALSE(result.overflow);
	EXPECT_EQ(result.sites[0], 0x1000u);
	EXPECT_EQ(result.sites[1], 0x2000u);
	EXPECT_EQ(result.sites[2], 0x4000u);
}

TEST(GamePatches, SelectProvenDtSitesEmpty)
{
	EXPECT_EQ(game_patches::select_proven_dt_sites(nullptr, 0).count, 0u);

	const game_patches::dt_candidate bare[] = {{0x1000, 0, 0}};
	const auto result = game_patches::select_proven_dt_sites(bare, 1);
	EXPECT_EQ(result.count, 0u);
	EXPECT_FALSE(result.overflow);
}

TEST(GamePatches, ScanFindsAlignedHit)
{
	std::vector<std::uint8_t> buf(64, 0);
	put60(buf, 8);
	put60(buf, 60);

	const auto result = game_patches::scan_dt_literals(buf.data(), 64, 0x10000);

	ASSERT_EQ(result.count, 2u);
	EXPECT_FALSE(result.overflow);
	EXPECT_EQ(result.sites[0], 0x10008u);
	EXPECT_EQ(result.sites[1], 0x1003Cu);
}

TEST(GamePatches, ScanAlignsToGuestAddress)
{
	// Base is 1 past alignment: the word at buffer offset 0 (guest ...001) must
	// not match, while the word at buffer offset 3 (guest ...004) must.
	std::vector<std::uint8_t> buf(16, 0);
	put60(buf, 0);
	put60(buf, 3);

	const auto result = game_patches::scan_dt_literals(buf.data(), 16, 0x10001);

	ASSERT_EQ(result.count, 1u);
	EXPECT_EQ(result.sites[0], 0x10004u);
}

TEST(GamePatches, ScanIgnoresMisalignedPattern)
{
	// The 1/60 byte run sits at offset 1: neither aligned word contains it.
	std::vector<std::uint8_t> buf(12, 0);
	put60(buf, 1);

	const auto result = game_patches::scan_dt_literals(buf.data(), 12, 0x20000);

	EXPECT_EQ(result.count, 0u);
	EXPECT_FALSE(result.overflow);
}

TEST(GamePatches, ScanOverflowAborts)
{
	// More literals than the safety bound: no partial site list may be used.
	std::vector<std::uint8_t> buf((game_patches::kMaxDtSites + 1) * 4, 0);
	for (std::size_t off = 0; off < buf.size(); off += 4)
	{
		put60(buf, off);
	}

	const auto result = game_patches::scan_dt_literals(buf.data(), static_cast<std::uint32_t>(buf.size()), 0x30000);

	EXPECT_TRUE(result.overflow);
	EXPECT_EQ(result.count, 0u);
}

TEST(GamePatches, ScanNullAndShortAreSafe)
{
	const auto null_result = game_patches::scan_dt_literals(nullptr, 1024, 0x40000);
	EXPECT_EQ(null_result.count, 0u);
	EXPECT_FALSE(null_result.overflow);

	const std::uint8_t tiny[3] = {0x3C, 0x88, 0x88};
	const auto short_result = game_patches::scan_dt_literals(tiny, 3, 0x40000);
	EXPECT_EQ(short_result.count, 0u);
	EXPECT_FALSE(short_result.overflow);
}

TEST(GamePatches, ScanCountsAddressRefs)
{
	// Words holding the target guest address (e.g. TOC entries pointing at a
	// pool float); misaligned and non-matching words must not count.
	std::vector<std::uint8_t> buf(32, 0);
	buf[4] = 0x00; buf[5] = 0x01; buf[6] = 0x26; buf[7] = 0x14;
	buf[12] = 0x00; buf[13] = 0x01; buf[14] = 0x26; buf[15] = 0x14;
	buf[20] = 0x01; buf[21] = 0x26; buf[22] = 0x14; buf[23] = 0x00;
	buf[28] = 0xDE; buf[29] = 0xAD; buf[30] = 0xBE; buf[31] = 0xEF;

	EXPECT_EQ(game_patches::count_address_refs(buf.data(), 32, 0x90000, 0x12614), 2u);
	EXPECT_EQ(game_patches::count_address_refs(buf.data(), 32, 0x90000, 0xDEADBEEF), 1u);
	EXPECT_EQ(game_patches::count_address_refs(nullptr, 32, 0x90000, 0x12614), 0u);
}

TEST(GamePatches, ScanFindsR2FloatLoads)
{
	// lfs f1, -256(r2) with TOC=0x90000 targets 0x8FF00. The RA=r3 decoy and
	// the wrong-displacement decoy must not match.
	std::vector<std::uint8_t> buf(32, 0);
	buf[8] = 0xC0; buf[9] = 0x22; buf[10] = 0xFF; buf[11] = 0x00;
	buf[16] = 0xC0; buf[17] = 0x23; buf[18] = 0xFF; buf[19] = 0x00;
	buf[24] = 0xC0; buf[25] = 0x22; buf[26] = 0x00; buf[27] = 0x10;

	const auto result = game_patches::find_r2_float_loads(buf.data(), 32, 0x70000, 0x90000, 0x8FF00);

	ASSERT_EQ(result.count, 1u);
	EXPECT_FALSE(result.overflow);
	EXPECT_EQ(result.sites[0], 0x70008u);
}

namespace
{
	struct fake_mem
	{
		std::map<std::uint32_t, std::uint32_t> cells;
		bool fail_reads = false;
		std::uint32_t writes = 0;

		std::uint32_t read(std::uint32_t vaddr, bool& ok)
		{
			if (fail_reads)
			{
				ok = false;
				return 0;
			}

			const auto it = cells.find(vaddr);
			ok = it != cells.end();
			return ok ? it->second : 0;
		}

		void write(std::uint32_t vaddr, std::uint32_t value)
		{
			cells[vaddr] = value;
			writes++;
		}
	};

	const std::uint32_t kSiteA = 0x101008;
	const std::uint32_t kSiteB = 0x202014;
	const std::uint32_t kSiteC = 0x303330;
	const std::uint32_t kSites[] = {kSiteA, kSiteB, kSiteC};

	fake_mem pristine()
	{
		fake_mem mem;
		for (const auto site : kSites)
		{
			mem.cells[site] = game_patches::kDt60Bits;
		}
		return mem;
	}
}

TEST(GamePatches, AppliesDiscoveredSites)
{
	fake_mem mem = pristine();

	const auto result = game_patches::apply_dt_sites(
		[&](std::uint32_t vaddr, bool& ok) { return mem.read(vaddr, ok); },
		[&](std::uint32_t vaddr, std::uint32_t value) { mem.write(vaddr, value); },
		kSites, 3);

	EXPECT_EQ(result.verified, 3u);
	EXPECT_EQ(result.applied, 3u);
	EXPECT_EQ(result.already, 0u);

	for (const auto site : kSites)
	{
		EXPECT_EQ(mem.cells[site], game_patches::kDt120Bits);
	}
}

TEST(GamePatches, SingleMismatchAbortsWithoutWriting)
{
	fake_mem mem = pristine();
	mem.cells[kSiteB] = 0xDEADBEEFu;

	const auto result = game_patches::apply_dt_sites(
		[&](std::uint32_t vaddr, bool& ok) { return mem.read(vaddr, ok); },
		[&](std::uint32_t vaddr, std::uint32_t value) { mem.write(vaddr, value); },
		kSites, 3);

	// Nothing may be partially scaled: all-or-nothing.
	EXPECT_EQ(result.verified, 0u);
	EXPECT_EQ(result.applied, 0u);
	EXPECT_EQ(result.already, 0u);
	EXPECT_EQ(mem.writes, 0u);
	EXPECT_EQ(mem.cells[kSiteA], game_patches::kDt60Bits);
}

TEST(GamePatches, RerunIsSafe)
{
	fake_mem mem = pristine();

	auto read = [&](std::uint32_t vaddr, bool& ok) { return mem.read(vaddr, ok); };
	auto write = [&](std::uint32_t vaddr, std::uint32_t value) { mem.write(vaddr, value); };

	const auto first = game_patches::apply_dt_sites(read, write, kSites, 3);
	EXPECT_EQ(first.applied, 3u);

	const auto second = game_patches::apply_dt_sites(read, write, kSites, 3);
	EXPECT_EQ(second.applied, 0u);
	EXPECT_EQ(second.already, 3u);
}

TEST(GamePatches, UnreadableAddressAbortsWithoutWriting)
{
	fake_mem mem = pristine();
	mem.fail_reads = true;

	const auto result = game_patches::apply_dt_sites(
		[&](std::uint32_t vaddr, bool& ok) { return mem.read(vaddr, ok); },
		[&](std::uint32_t vaddr, std::uint32_t value) { mem.write(vaddr, value); },
		kSites, 3);

	EXPECT_EQ(result.applied, 0u);
	EXPECT_EQ(mem.writes, 0u);
}

// ---- Fast single-pass discovery (boot path) ----

namespace
{
	void put_be32(std::vector<std::uint8_t>& buf, std::size_t offset, std::uint32_t v)
	{
		buf[offset] = static_cast<std::uint8_t>(v >> 24);
		buf[offset + 1] = static_cast<std::uint8_t>(v >> 16);
		buf[offset + 2] = static_cast<std::uint8_t>(v >> 8);
		buf[offset + 3] = static_cast<std::uint8_t>(v);
	}

	double bits_to_double(std::uint64_t bits)
	{
		double value = 0.;
		std::memcpy(&value, &bits, sizeof(value));
		return value;
	}
}

TEST(GamePatches, DoubleConstantsDecode)
{
	EXPECT_DOUBLE_EQ(bits_to_double(game_patches::kDt60Bits64), 1. / 60.);
	EXPECT_DOUBLE_EQ(bits_to_double(game_patches::kDt120Bits64), 1. / 120.);
	// Same mantissa: only the high (exponent) word differs, which apply_dt_sites_all relies on
	EXPECT_EQ(static_cast<std::uint32_t>(game_patches::kDt60Bits64), static_cast<std::uint32_t>(game_patches::kDt120Bits64));
}

TEST(GamePatches, ScanFindsAlignedDoubles)
{
	std::vector<std::uint8_t> buf(64, 0);
	put_be32(buf, 8, 0x3F911111);
	put_be32(buf, 12, 0x11111111);
	// Misaligned (4 mod 8) copy must be ignored
	put_be32(buf, 36, 0x3F911111);
	put_be32(buf, 40, 0x11111111);

	const auto result = game_patches::scan_dt64_literals(buf.data(), 64, 0x10000);
	ASSERT_EQ(result.count, 1u);
	EXPECT_EQ(result.sites[0], 0x10008u);
}

TEST(GamePatches, AccumulateAddressRefsMatchesPerCandidateScan)
{
	std::vector<std::uint8_t> buf(64, 0);
	put_be32(buf, 0, 0x12614);
	put_be32(buf, 8, 0x12700);
	put_be32(buf, 12, 0x12614);
	put_be32(buf, 60, 0x99999);

	const std::uint32_t targets[] = {0x12614, 0x12700, 0x13000};
	std::uint32_t refs[3]{};
	game_patches::accumulate_address_refs(buf.data(), 64, 0x90000, targets, 3, refs);

	for (int i = 0; i < 3; i++)
	{
		EXPECT_EQ(refs[i], game_patches::count_address_refs(buf.data(), 64, 0x90000, targets[i]));
	}

	EXPECT_EQ(refs[0], 2u);
	EXPECT_EQ(refs[1], 1u);
	EXPECT_EQ(refs[2], 0u);
}

TEST(GamePatches, AccumulateR2FloatLoads)
{
	std::vector<std::uint8_t> buf(32, 0);
	// lfs f1, -0x100(r2) -> 0x8FF00 ; lfd f2, 0x10(r2) -> 0x90010 ; lfs f1, -0x100(r3) (not r2)
	put_be32(buf, 0, (48u << 26) | (1u << 21) | (2u << 16) | 0xFF00u);
	put_be32(buf, 8, (50u << 26) | (2u << 21) | (2u << 16) | 0x0010u);
	put_be32(buf, 12, (48u << 26) | (1u << 21) | (3u << 16) | 0xFF00u);
	put_be32(buf, 16, (48u << 26) | (1u << 21) | (2u << 16) | 0xFF00u);

	const std::uint32_t targets[] = {0x8FF00, 0x90010, 0x95000};
	std::uint32_t loads[3]{}, first[3]{};
	game_patches::accumulate_r2_float_loads(buf.data(), 32, 0x70000, 0x90000, targets, 3, loads, first);

	EXPECT_EQ(loads[0], 2u);
	EXPECT_EQ(first[0], 0x70000u);
	EXPECT_EQ(loads[1], 1u);
	EXPECT_EQ(first[1], 0x70008u);
	EXPECT_EQ(loads[2], 0u);
}

TEST(GamePatches, ApplyAllPatchesFloatsAndDoublesTogether)
{
	std::map<std::uint32_t, std::uint32_t> mem{{0x100, game_patches::kDt60Bits}, {0x200, 0x3F911111}, {0x204, 0x11111111}};

	auto read = [&](std::uint32_t a, bool& ok) -> std::uint32_t { auto it = mem.find(a); ok = it != mem.end(); return ok ? it->second : 0; };
	auto write = [&](std::uint32_t a, std::uint32_t v) { mem[a] = v; };

	const std::uint32_t f[] = {0x100};
	const std::uint32_t d[] = {0x200};
	const auto r = game_patches::apply_dt_sites_all(read, write, f, 1, d, 1);

	EXPECT_EQ(r.verified, 2u);
	EXPECT_EQ(r.applied, 2u);
	EXPECT_EQ(mem[0x100], game_patches::kDt120Bits);
	EXPECT_EQ((static_cast<std::uint64_t>(mem[0x200]) << 32) | mem[0x204], game_patches::kDt120Bits64);

	// Re-run is a no-op
	const auto again = game_patches::apply_dt_sites_all(read, write, f, 1, d, 1);
	EXPECT_EQ(again.applied, 0u);
	EXPECT_EQ(again.already, 2u);
}

TEST(GamePatches, ApplyAllAbortsOnBadDoubleWithoutWriting)
{
	std::map<std::uint32_t, std::uint32_t> mem{{0x100, game_patches::kDt60Bits}, {0x200, 0x3F911111}, {0x204, 0x22222222}};

	auto read = [&](std::uint32_t a, bool& ok) -> std::uint32_t { auto it = mem.find(a); ok = it != mem.end(); return ok ? it->second : 0; };
	auto write = [&](std::uint32_t a, std::uint32_t v) { mem[a] = v; };

	const std::uint32_t f[] = {0x100};
	const std::uint32_t d[] = {0x200};
	const auto r = game_patches::apply_dt_sites_all(read, write, f, 1, d, 1);

	EXPECT_EQ(r.applied, 0u);
	EXPECT_EQ(mem[0x100], game_patches::kDt60Bits);
}
