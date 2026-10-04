#pragma once

// Whole-GPU utilization as macOS reports it (Activity Monitor, Redline / mac-resource-monitor): the GPU driver's
// "PerformanceStatistics" dictionary on every IOAccelerator service in the IORegistry. The keys are driver specific:
// "Device Utilization %", "Renderer Utilization %" and "Tiler Utilization %" are alternate views of the same engines on
// different drivers, so the busiest one is the honest answer, taken across all accelerators.

#include "util/types.hpp"

namespace mtl
{
	// 0..100, or a negative value when no accelerator exposes the statistics. Not thread-safe: call from one thread.
	f32 read_iokit_gpu_utilization();
}
