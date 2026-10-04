R"(
#define _select mix
#define _saturate(x) clamp(x, 0., 1.)
#define _get_bits(x, off, count) bitfieldExtract(x, off, count)
#define _set_bits(x, y, off, count) bitfieldInsert(x, y, off, count)
#define _test_bit(x, y) (_get_bits(x, y, 1) != 0)
#define _rand(seed) _hash_noise(seed.xy)

// Per-pixel noise in [0, 1). Integer hash: exact on every backend. The classic fract(sin(dot(p, k)) * 43758.5)
// needs sin() of arguments in the tens of thousands, which fast-math sin() (Metal) evaluates so coarsely that the
// "noise" comes out in clumps and blocks (visible as blotchy alpha-to-coverage edges and particles).
float _hash_noise(const in vec2 p)
{
	uvec2 q = uvec2(ivec2(floor(p)));
	uint h = (q.x * 0x8da6b343u) ^ (q.y * 0xd8163841u);
	h ^= h >> 16;
	h *= 0x7feb352du;
	h ^= h >> 15;
	h *= 0x846ca68bu;
	h ^= h >> 16;
	return float(h >> 8) * (1.0 / 16777216.0);
}

#ifdef _GPU_LOW_PRECISION_COMPARE
#define CMP_FIXUP(a) (sign(a) * 16. + a)
#else
#define CMP_FIXUP(a) (a)
#endif

#define _builtin_approx_pow(a, b) exp2((b) * log2(a))

#ifdef _ENABLE_LIT_EMULATION
// LIT is well documented in NV extension documents. See https://registry.khronos.org/OpenGL/extensions/NV/NV_vertex_program.txt
// In RSXFP the LIT instruction is unimplemented in hw. In it's place we have LIF and full LIT is emulated using at least 3 instructions.
vec4 _builtin_lit(const in vec4 values)
{
    // We clamp t.y to 1e-10 to avoid NaN in approx_pow when y=0 and w=0. This matches the spec's pow(x, 0) == 1 requirement.
    vec4 t = vec4(max(values.xy, vec2(0.f, 1e-10)), values.zw);
    return vec4(1.f,
        t.x,
        t.x > 0.f ? _builtin_approx_pow(t.y, t.w) : 0.f,
        1.f);
}

// LIT d, t on RSXFP becomes:
// t.xy = max(t.xy, 0)
// t.w = t.w * log2(t.y)
// LIF d, t
vec4 _builtin_lif(const in vec4 t)
{
    return vec4(1.f,
        t.y,
        t.y > 0.f ? exp2(t.w) : 0.f,
        1.f);
}
#endif

)"
