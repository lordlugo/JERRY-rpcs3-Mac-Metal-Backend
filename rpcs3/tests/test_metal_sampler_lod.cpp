#include <gtest/gtest.h>

#include <set>

#include "Emu/RSX/Metal/mtlutils/sampler.h"

// GoW Ascension varies min_lod continuously per frame (40 distinct values in the last 60
// sampler creations before the RSX 0x448 crash). The pool dedupes by exact float equality,
// so each distinct value minted a new MTLSamplerState — and a new driver GPU resource ID —
// until the ~1024-live-sampler driver table overflowed and setSamplerState segfaulted on
// ID 0x400/0x401. The fill sites now snap LOD params with quantize_sampler_lod(); these
// tests pin the snap contract: jitter dedupes, exact values survive, and pool keys compare
// equal after quantization.
namespace
{
	TEST(MetalSamplerLOD, KillerAndHealthyMergeToSameBucket)
	{
		// Killer id=0x400 (min_lod 1.457031) vs healthy id=0x3fa (min_lod 1.523438): same
		// flags, neighbouring LODs. Post-fix both must resolve to one pooled sampler.
		EXPECT_FLOAT_EQ(mtl::quantize_sampler_lod(1.457031f), mtl::quantize_sampler_lod(1.523438f));
		EXPECT_FLOAT_EQ(mtl::quantize_sampler_lod(1.457031f), 1.5f);
	}

	TEST(MetalSamplerLOD, ObservedJitterCollapsesToFewBuckets)
	{
		// Distinct min_lod values from the 60 creations before the crash; 0.5 steps must
		// collapse the jitter cluster instead of minting one state per value.
		const float observed[] = { 0.031250f, 0.062500f, 0.066406f, 0.097656f, 0.101562f,
			0.132812f, 0.164062f, 0.167969f, 0.199219f, 0.230469f, 0.265625f };
		std::set<float> buckets;
		for (float v : observed)
		{
			buckets.insert(mtl::quantize_sampler_lod(v));
		}
		EXPECT_LE(buckets.size(), 2u);
	}

	TEST(MetalSamplerLOD, ExactValuesSurviveUnchanged)
	{
		EXPECT_FLOAT_EQ(mtl::quantize_sampler_lod(0.f), 0.f);
		EXPECT_FLOAT_EQ(mtl::quantize_sampler_lod(8.f), 8.f);
		EXPECT_FLOAT_EQ(mtl::quantize_sampler_lod(-1.f), -1.f);
		EXPECT_FLOAT_EQ(mtl::quantize_sampler_lod(0.5f), 0.5f);
		// Idempotent: quantizing twice changes nothing.
		EXPECT_FLOAT_EQ(mtl::quantize_sampler_lod(mtl::quantize_sampler_lod(1.457031f)), 1.5f);
	}

	TEST(MetalSamplerLOD, QuantizedInfosDedupeInPool)
	{
		// Simulate the fill sites: both infos carry quantized LODs, so the pool's exact
		// operator==/hash find the shared entry instead of creating a second sampler.
		mtl::sampler_create_info a{}, b{};
		a.min_lod = mtl::quantize_sampler_lod(1.457031f);
		b.min_lod = mtl::quantize_sampler_lod(1.523438f);
		a.max_lod = b.max_lod = mtl::quantize_sampler_lod(8.f);
		a.mip_lod_bias = b.mip_lod_bias = mtl::quantize_sampler_lod(-1.f);
		EXPECT_TRUE(a == b);
		EXPECT_EQ(a.hash(), b.hash());
	}
}
