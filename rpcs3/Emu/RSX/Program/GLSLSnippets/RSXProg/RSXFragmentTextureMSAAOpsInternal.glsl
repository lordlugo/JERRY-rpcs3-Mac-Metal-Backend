R"(
vec4 texelFetch2DMS(in _MSAA_SAMPLER_TYPE_ tex, const in ivec2 clamp_bounds, const in vec2 sample_count, const in ivec2 icoords, const in ivec2 offset)
{
	const vec2 resolve_coords = vec2(clamp(icoords + offset, ivec2(0), clamp_bounds)); // Clamp to edge. Input offset is always zero or positive.
	const vec2 aa_coords = floor(resolve_coords / sample_count);                       // AA coords = real_coords / sample_count
	const vec2 sample_loc = fma(aa_coords, -sample_count, resolve_coords);             // Sample ID = real_coords % sample_count
	const float sample_index = fma(sample_loc.y, sample_count.y, sample_loc.x);

	// TODO: Hack. Filtering will break when sampling sub-pixel sample ids in wrap mode.
	return texelFetch(tex, ivec2(aa_coords), int(sample_index));
}

vec4 sampleTexture2DMS(in _MSAA_SAMPLER_TYPE_ tex, const in vec2 coords, const in sampler_info tex_params)
{
	const uint flags = tex_params.flags;
	const vec2 scaled_coords = _texcoord_xform(coords, tex_params);
	const vec2 normalized_coords = texture2DMSCoord(scaled_coords, flags);
	const vec2 sample_count = vec2(2., textureSamples(tex) * 0.5);
	const ivec2 image_size = ivec2(textureSize(tex) * sample_count);
	const ivec2 clamp_bounds = image_size - ivec2(1);

	// Position in the sample-expanded image, snapped to 1/128 texel like the fixed-point texel addressing of a texture
	// unit. A lookup at a pixel centre lands exactly on the boundary between the two samples of that pixel; without the
	// snap, float rounding (interpolation, fast math) picks either sample from one lookup to the next, so an edge pixel
	// could mix the depth of one surface with the normal or the light of another (outlines along geometry edges in
	// light pre-pass / deferred games). Multiplying by a power of two keeps the snapped value exact.
	const vec2 texel_coords = floor(fma(normalized_coords, vec2(image_size) * 128., vec2(0.5))) * (1. / 128.);
	const ivec2 icoords = ivec2(texel_coords);
	const vec4 sample0 = texelFetch2DMS(tex, clamp_bounds, sample_count, icoords, ivec2(0));

	if (_get_bits(flags, FILTERED_MAG_BIT, 2) == 0)
	{
		return sample0;
	}

	// Bilinear scaling, with upto 2x2 downscaling with simple weights
	const vec2 uv_step = 1.0 / vec2(image_size);
	const vec2 actual_step = vec2(dFdx(normalized_coords.x), dFdy(normalized_coords.y));

	const bvec2 no_filter = lessThan(abs(uv_step - actual_step), vec2(0.000001));
	if (no_filter.x && no_filter.y)
	{
		return sample0;
	}

	// Filtered read. Positions are in sample-expanded texels (texel_coords, snapped as above).
	//  - Minification (the common case: an MSAA surface drawn at its pixel size, i.e. the game's own downsample or
	//    resolve): box filter over the screen pixel's footprint, CENTRED on the lookup position. A lookup at a pixel
	//    centre then averages exactly that pixel's own samples, which is what the PS3's texture unit returns. The
	//    footprint used to start at the lookup position, which averaged the last sample of one pixel with the first
	//    sample of the next: edge pixels picked up a neighbour's sample, alternately per row (stair-stepped,
	//    checkerboard-like edges, e.g. WWE SmackDown vs. Raw 2011).
	//  - Magnification: bilinear between the two nearest texel centres (texel centre = index + 0.5).
	//  - 1:1 on an axis: the texel under the position.
	const vec2 footprint = actual_step * vec2(image_size);
	vec2 start = texel_coords;

	if (!no_filter.x)
	{
		start.x -= (actual_step.x > uv_step.x) ? 0.5 * footprint.x : 0.5;
	}

	if (!no_filter.y)
	{
		start.y -= (actual_step.y > uv_step.y) ? 0.5 * footprint.y : 0.5;
	}

	const ivec2 base = ivec2(floor(start));
	vec4 a, b;

	if (no_filter.x)
	{
		// No scaling, 1:1
		a = texelFetch2DMS(tex, clamp_bounds, sample_count, base, ivec2(0, 0));
		b = texelFetch2DMS(tex, clamp_bounds, sample_count, base, ivec2(0, 1));
	}
	else if (actual_step.x > uv_step.x)
	{
		// Downscale in X: box filter over up to 3 texels
		const vec3 weights = compute2x2DownsampleWeights(start.x, 1.0, footprint.x);
		const vec4 s00 = texelFetch2DMS(tex, clamp_bounds, sample_count, base, ivec2(0, 0));
		const vec4 s10 = texelFetch2DMS(tex, clamp_bounds, sample_count, base, ivec2(1, 0));
		const vec4 s20 = texelFetch2DMS(tex, clamp_bounds, sample_count, base, ivec2(2, 0));
		a = fma(s00, weights.xxxx, s10 * weights.y) + (s20 * weights.z);

		if (!no_filter.y)
		{
			const vec4 s01 = texelFetch2DMS(tex, clamp_bounds, sample_count, base, ivec2(0, 1));
			const vec4 s11 = texelFetch2DMS(tex, clamp_bounds, sample_count, base, ivec2(1, 1));
			const vec4 s21 = texelFetch2DMS(tex, clamp_bounds, sample_count, base, ivec2(2, 1));
			b = fma(s01, weights.xxxx, s11 * weights.y) + (s21 * weights.z);
		}
		else
		{
			b = a;
		}
	}
	else
	{
		// Upscale in X: bilinear between texel centres
		const float fx = fract(start.x);
		a = mix(texelFetch2DMS(tex, clamp_bounds, sample_count, base, ivec2(0, 0)), texelFetch2DMS(tex, clamp_bounds, sample_count, base, ivec2(1, 0)), fx);
		b = mix(texelFetch2DMS(tex, clamp_bounds, sample_count, base, ivec2(0, 1)), texelFetch2DMS(tex, clamp_bounds, sample_count, base, ivec2(1, 1)), fx);
	}

	if (no_filter.y)
	{
		// 1:1 no scale
		return a;
	}
	else if (actual_step.y > uv_step.y)
	{
		// Downscale in Y. Only 2 rows are fetched for performance reasons, so the third row's weight goes to row 2.
		const vec3 weights = compute2x2DownsampleWeights(start.y, 1.0, footprint.y);
		return a * weights.x + b * (weights.y + weights.z);
	}
	else
	{
		// Upscale in Y: bilinear between texel centres
		return mix(a, b, fract(start.y));
	}
}

)"
