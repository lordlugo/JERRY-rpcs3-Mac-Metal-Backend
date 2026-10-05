#include "stdafx.h"
#include "commands.h"
#include "buffer_object.h"
#include "image.h"

#include <bit>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mtl
{
	namespace
	{
		struct gpu_stats_state
		{
			std::mutex mutex;
			f64 busy_until = 0.;
			u64 busy_ns = 0;
			atomic_t<u64> draw_render_passes = 0;
			std::array<std::array<atomic_t<u64>, gpu_stats_t::context_count>, gpu_stats_t::reason_count> draw_pass_ends{};
			std::array<atomic_t<u64>, gpu_stats_t::context_count> other_passes{};
			std::array<atomic_t<u64>, static_cast<u32>(pass_event::count)> pass_events{};
			atomic_t<u64> attachment_load_bytes = 0;
			atomic_t<u64> attachment_store_bytes = 0;
			atomic_t<u64> feedback_splits = 0;
			std::array<atomic_t<u64>, static_cast<u32>(pass_split_reason::count)> splits_by_reason{};
			atomic_t<u64> feedback_reads_in_pass = 0;
			atomic_t<u64> uploads_ahead = 0;
			atomic_t<u64> uploads_inline = 0;
			atomic_t<u64> uploads_inline_split = 0;
			atomic_t<u64> submissions = 0;
			atomic_t<u64> table_writes = 0;
			atomic_t<u64> table_writes_skipped = 0;
			atomic_t<u64> state_sets = 0;
			atomic_t<u64> state_sets_skipped = 0;
			atomic_t<u64> ordering_points = 0;
			atomic_t<u64> ordering_points_free = 0;
			atomic_t<u64> queue_barriers = 0;
			atomic_t<u64> pass_barriers = 0;
			atomic_t<u64> first_draw_barriers = 0;
			atomic_t<u64> late_barriers = 0;
			atomic_t<u64> encoder_barriers = 0;
			atomic_t<u64> full_barriers = 0;
			atomic_t<u64> submissions_waited = 0;
			atomic_t<u64> submissions_overlapped = 0;
			atomic_t<u32> errors_logged = 0;
		};

		// Never destroyed: commit feedback can still arrive on a Metal thread while the process exits
		gpu_stats_state& gpu_stats()
		{
			static gpu_stats_state* s_state = new gpu_stats_state();
			return *s_state;
		}

		atomic_t<u64> g_pass_serial = 0;

		// What this thread is doing (pass_context_scope)
		thread_local pass_context g_pass_context = pass_context::other;

		// Bytes per pixel of an attachment plane (telemetry estimate: formats the backend renders to)
		u32 get_attachment_texel_size(MTL::PixelFormat format, bool stencil_plane)
		{
			switch (format)
			{
			case MTL::PixelFormatR8Unorm:
			case MTL::PixelFormatA8Unorm:
			case MTL::PixelFormatStencil8:
				return 1;
			case MTL::PixelFormatRG8Unorm:
			case MTL::PixelFormatR16Unorm:
			case MTL::PixelFormatR16Float:
			case MTL::PixelFormatB5G6R5Unorm:
			case MTL::PixelFormatA1BGR5Unorm:
			case MTL::PixelFormatBGR5A1Unorm:
			case MTL::PixelFormatABGR4Unorm:
			case MTL::PixelFormatDepth16Unorm:
				return 2;
			case MTL::PixelFormatRGBA16Float:
			case MTL::PixelFormatRGBA16Unorm:
			case MTL::PixelFormatRG32Float:
				return 8;
			case MTL::PixelFormatRGBA32Float:
				return 16;
			case MTL::PixelFormatDepth32Float_Stencil8:
				return stencil_plane ? 1 : 4;
			default:
				return 4;
			}
		}

		// Called by Metal when committed work finishes (any thread)
		void on_commit_feedback(MTL4::CommitFeedback* feedback)
		{
			if (!feedback)
			{
				return;
			}

			auto& state = gpu_stats();

			if (const NS::Error* error = feedback->error(); error && state.errors_logged++ < 16)
			{
				autorelease_scope pool;
				rsx_log.error("Metal: GPU error in committed work: %s", to_string(error));
			}

			const f64 start = feedback->GPUStartTime();
			const f64 end = feedback->GPUEndTime();
			if (!(end > start))
			{
				return;
			}

			// Union of the intervals (approximate when feedback arrives out of order across queues)
			std::lock_guard lock(state.mutex);
			const f64 from = std::max(start, state.busy_until);
			if (end > from)
			{
				state.busy_ns += static_cast<u64>((end - from) * 1'000'000'000.);
			}
			state.busy_until = std::max(state.busy_until, end);
		}

		// Waits + commit + timeline signal must be atomic across threads, otherwise two submitters could interleave and
		// signal timeline values out of order relative to their work. Also guards the per-queue submission records.
		std::mutex& submit_mutex()
		{
			static std::mutex* s_mutex = new std::mutex();
			return *s_mutex;
		}

		// ---- Hazard tracking ------------------------------------------------------------------------------------
		// Stage classes: the stages a barrier can name, grouped. Coverage is tracked per (producer, consumer) class.
		enum stage_class : u32
		{
			class_vertex = 0,
			class_fragment,
			class_tile,
			class_dispatch,
			class_blit,
			class_other, // object, mesh, resource state, acceleration structure, machine learning
			class_count
		};

		using class_mask = u32;
		constexpr class_mask all_classes = (1u << class_count) - 1;
		constexpr MTL::Stages other_stages = stages_all & ~(stages_render | stages_compute);

		constexpr class_mask to_classes(MTL::Stages stages)
		{
			class_mask result = 0;
			if (stages & MTL::StageVertex) result |= 1u << class_vertex;
			if (stages & MTL::StageFragment) result |= 1u << class_fragment;
			if (stages & MTL::StageTile) result |= 1u << class_tile;
			if (stages & MTL::StageDispatch) result |= 1u << class_dispatch;
			if (stages & MTL::StageBlit) result |= 1u << class_blit;
			if (stages & ~(stages_render | stages_compute)) result |= 1u << class_other;
			return result;
		}

		constexpr MTL::Stages to_stages(class_mask classes)
		{
			MTL::Stages result = 0;
			if (classes & (1u << class_vertex)) result |= MTL::StageVertex;
			if (classes & (1u << class_fragment)) result |= MTL::StageFragment;
			if (classes & (1u << class_tile)) result |= MTL::StageTile;
			if (classes & (1u << class_dispatch)) result |= MTL::StageDispatch;
			if (classes & (1u << class_blit)) result |= MTL::StageBlit;
			if (classes & (1u << class_other)) result |= other_stages;
			return result;
		}

		constexpr class_mask render_classes = to_classes(stages_render);
		constexpr class_mask attachment_classes = to_classes(stages_attachment);
		constexpr class_mask compute_classes = to_classes(stages_compute);

		template <typename F>
		void for_each_class(class_mask mask, F&& func)
		{
			for (; mask; mask &= mask - 1)
			{
				func(static_cast<u32>(std::countr_zero(mask)));
			}
		}

		// cover[a][b]: accesses of producer class a with a key (encoder serial, command index or submission serial) below
		// this value are ordered before every later work of consumer class b
		using cover_matrix = std::array<std::array<u64, class_count>, class_count>;

		// Accesses of one kind (reads or writes) to one resource, merged: the union of their ranges and stage classes
		struct range_set
		{
			u64 lo = 0;           // buffer: first byte, texture: subresource mask
			u64 hi = 0;           // buffer: end byte
			class_mask stages = 0; // 0: empty
			u64 key = 0;          // encoder serial / command index / submission serial of the latest access

			bool empty() const { return !stages; }
			void clear() { *this = {}; }
		};

		bool overlaps(const range_set& set, u64 lo, u64 hi, bool texture)
		{
			return texture ? (set.lo & lo) != 0 : (set.lo < hi && lo < set.hi);
		}

		void add(range_set& set, u64 lo, u64 hi, bool texture, class_mask stages, u64 key)
		{
			if (set.empty())
			{
				set.lo = lo;
				set.hi = hi;
			}
			else if (texture)
			{
				set.lo |= lo;
			}
			else
			{
				set.lo = std::min(set.lo, lo);
				set.hi = std::max(set.hi, hi);
			}

			set.stages |= stages;
			set.key = std::max(set.key, key);
		}

		void add(range_set& set, const range_set& other, bool texture, u64 key)
		{
			if (!other.empty())
			{
				add(set, other.lo, other.hi, texture, other.stages, key);
			}
		}

		// The producer classes of `set` that are not yet ordered before every later work of the `consumers` classes
		class_mask uncovered(const range_set& set, class_mask consumers, const cover_matrix& cover)
		{
			class_mask result = 0;
			for_each_class(set.stages, [&](u32 a)
			{
				for_each_class(consumers, [&](u32 b)
				{
					if (set.key >= cover[a][b])
					{
						result |= 1u << a;
					}
				});
			});
			return result;
		}

		// Drops the classes of `set` that are ordered before every later work of every class
		void prune(range_set& set, const cover_matrix& cover)
		{
			for_each_class(set.stages, [&](u32 a)
			{
				bool covered = true;
				for (u32 b = 0; b < class_count && covered; ++b)
				{
					covered = set.key < cover[a][b];
				}

				if (covered)
				{
					set.stages &= ~(1u << a);
				}
			});

			if (set.empty())
			{
				set.clear();
			}
		}

		void cover_pairs(cover_matrix& cover, class_mask producers, class_mask consumers, u64 key)
		{
			for_each_class(producers, [&](u32 a)
			{
				for_each_class(consumers, [&](u32 b)
				{
					cover[a][b] = std::max(cover[a][b], key);
				});
			});
		}

		u64 pair_bits(class_mask producers, class_mask consumers)
		{
			u64 bits = 0;
			for_each_class(producers, [&](u32 a)
			{
				for_each_class(consumers, [&](u32 b)
				{
					bits |= u64{1} << (a * class_count + b);
				});
			});
			return bits;
		}

		// Whether some (producer class of `set`, consumer class) pair of `pairs` (pair_bits) is not yet ordered
		bool uncovered_pairs(const range_set& set, u64 pairs, const cover_matrix& cover)
		{
			bool result = false;
			for_each_class(set.stages, [&](u32 a)
			{
				for (u32 b = 0; b < class_count && !result; ++b)
				{
					result = (pairs & (u64{1} << (a * class_count + b))) && set.key >= cover[a][b];
				}
			});
			return result;
		}

		// Pairs of an access made by a draw of a render pass (consumer classes `consumers`) that the barriers the pass began
		// with do not order after every earlier access of the queue, earlier submissions included: the vertex stage after
		// fragment and tile work. Its fragment and tile work (attachments, fragment reads, visibility results) is ordered
		// after everything.
		u64 draw_external_pairs(class_mask consumers)
		{
			return (consumers & (1u << class_vertex)) ? pair_bits(attachment_classes, 1u << class_vertex) : 0;
		}

		// Accesses of one recording to one resource (or, for the `unknown` entry, to everything)
		struct hazard_entry
		{
			const void* resource = nullptr;
			bool texture = false;
			u64 encoder = 0;  // encoder serial of `current` and `group`
			u64 command = 0;  // command index of `group`

			// Keys: encoder serial (prior), command index (current, group)
			range_set read_prior, write_prior;     // accesses of earlier encoders
			range_set read_current, write_current; // accesses of earlier commands of encoder `encoder`
			range_set read_group, write_group;     // accesses of command `command` of encoder `encoder`
			range_set read_all, write_all;         // everything the recording did (submission check; key unused)

			// Submission check: (producer class of an earlier submission, consumer class of this recording) pairs of the
			// recording's reads / writes that its own barriers do not order (pair_bits). Compute commands: every pair;
			// draws: see draw_external_pairs(); attachments: none (the pass barriers order them after everything).
			u64 external_read = 0, external_write = 0;
		};

		struct barrier_needs
		{
			class_mask queue = 0;   // producer classes of earlier encoders to wait for
			class_mask encoder = 0; // producer classes of earlier commands of the open compute encoder to wait for
		};

		// ---- Submissions ------------------------------------------------------------------------------------------
		// What earlier submissions to a queue accessed, until they completed
		struct queue_record
		{
			bool texture = false;
			range_set read, write; // key: submission serial of the latest access
		};

		struct queue_state
		{
			const MTL4::CommandQueue* queue = nullptr;
			u64 serial = 0;    // submissions so far
			u64 completed = 0; // every submission up to this one completed, or a later one waited for it
			u64 swept = 0;     // `completed` when `records` were last cleaned up

			struct submission
			{
				u64 serial;
				const timeline* tl;
				u64 value;
			};
			std::deque<submission> in_flight;

			cover_matrix cover{}; // serial of the latest submission with a queue barrier a -> b
			std::unordered_map<const void*, queue_record> records;
			range_set unknown;    // undeclared accesses: writes of everything
		};

		// Guarded by submit_mutex()
		std::vector<std::unique_ptr<queue_state>>& queue_states()
		{
			static auto* s_states = new std::vector<std::unique_ptr<queue_state>>();
			return *s_states;
		}

		queue_state& get_queue_state(const MTL4::CommandQueue* queue)
		{
			for (auto& state : queue_states())
			{
				if (state->queue == queue)
				{
					return *state;
				}
			}

			auto& state = queue_states().emplace_back(std::make_unique<queue_state>());
			state->queue = queue;
			return *state;
		}

		void retire_completed(queue_state& qs)
		{
			while (!qs.in_flight.empty())
			{
				const auto& front = qs.in_flight.front();
				if (front.serial > qs.completed && !front.tl->is_complete(front.value))
				{
					break;
				}

				qs.completed = std::max(qs.completed, front.serial);
				qs.in_flight.pop_front();
			}

			if (qs.swept == qs.completed)
			{
				return;
			}

			qs.swept = qs.completed;

			auto clean = [&](range_set& set)
			{
				if (!set.empty() && set.key <= qs.completed)
				{
					set.clear();
				}

				prune(set, qs.cover);
			};

			for (auto it = qs.records.begin(); it != qs.records.end();)
			{
				clean(it->second.read);
				clean(it->second.write);
				it = (it->second.read.empty() && it->second.write.empty()) ? qs.records.erase(it) : std::next(it);
			}

			clean(qs.unknown);
		}

		// Subresource bits of one axis: 8 bits, index % 8
		u8 axis_bits(u32 base, u32 count)
		{
			if (count >= 8)
			{
				return 0xFF;
			}

			const u32 bits = ((1u << count) - 1) << (base & 7);
			return static_cast<u8>(bits | (bits >> 8));
		}

		gpu_access buffer_access(const void* buf, u64 offset, u64 length, u64 size, bool write)
		{
			if (!buf || !length)
			{
				return {};
			}

			const u64 end = (length == umax || offset + length > size) ? size : offset + length;
			return { .resource = buf, .lo = offset, .hi = std::max(end, offset + 1), .kind = gpu_access::buffer_range, .write = write };
		}

		gpu_access texture_access(const MTL::Texture* tex, bool write)
		{
			if (!tex)
			{
				return {};
			}

			// Views and textures made from buffers alias the memory of what they were made from
			if (const MTL::Buffer* buf = tex->buffer())
			{
				return { .resource = buf, .lo = 0, .hi = u64{umax}, .kind = gpu_access::buffer_range, .write = write };
			}

			while (const MTL::Texture* parent = tex->parentTexture())
			{
				tex = parent;
			}

			if (const MTL::Buffer* buf = tex->buffer())
			{
				return { .resource = buf, .lo = 0, .hi = u64{umax}, .kind = gpu_access::buffer_range, .write = write };
			}

			return { .resource = tex, .lo = u64{umax}, .kind = gpu_access::texture_subresources, .write = write };
		}

		// An attachment of a render pass descriptor (level/slice relative to `tex`, which may be a view)
		gpu_access attachment_access(const MTL::Texture* tex, NS::UInteger level, NS::UInteger slice)
		{
			if (!tex)
			{
				return {};
			}

			if (tex->parentTexture() || tex->buffer())
			{
				// A view: the whole texture it was made from
				return texture_access(tex, true);
			}

			const bool is_3d = tex->textureType() == MTL::TextureType3D;
			return
			{
				.resource = tex,
				.lo = subresource_mask(static_cast<u32>(level), 1, is_3d ? 0 : static_cast<u32>(slice), is_3d ? u32{umax} : 1u),
				.kind = gpu_access::texture_subresources,
				.write = true
			};
		}
	}

	// ---- Hazard state of one recording ----------------------------------------------------------------------------

	struct command_list::hazard_state
	{
		std::vector<hazard_entry> entries;
		std::vector<u32> index; // Open addressing: entry index + 1, 0 = free. Size: power of two, at most half full.
		hazard_entry unknown;   // Undeclared accesses: writes of everything

		cover_matrix queue_cover{};   // Encoder serial of the latest queue barrier a -> b (covers prior sets below it)
		cover_matrix encoder_cover{}; // Command index of the latest intra-encoder barrier a -> b in the open compute encoder
		u64 barrier_pairs = 0;        // Bit a * class_count + b: a queue barrier a -> b was encoded (submission coverage)

		u64 encoder_serial = 0; // Serial of the open (or last) encoder; external work gets one too. 0: none yet.
		u64 command = 0;        // Current command of the open compute encoder (0: none yet)
		bool render = false;    // The open encoder is a render pass
		bool split_required = false; // A draw of the open draw pass needs a barrier the pass can no longer get

		// Argument table slots declared in the open render pass (vertex, fragment)
		struct table_declarations
		{
			std::array<gpu_access, gpu_capabilities::max_buffers_per_stage> buffers{};
			std::array<gpu_access, argument_table_shadow::max_textures> textures{};
			u32 buffers_known = 0;
			u64 textures_known = 0;
		};
		std::array<table_declarations, 2> tables{};

		// Telemetry since the list last submitted
		u64 ordering_points = 0;
		u64 ordering_points_free = 0;
		u64 queue_barriers = 0;
		u64 pass_barriers = 0;
		u64 first_draw_barriers = 0;
		u64 late_barriers = 0;
		u64 encoder_barriers = 0;
		u64 full_barriers = 0;

		void reset()
		{
			entries.clear();
			std::fill(index.begin(), index.end(), 0u);
			unknown = {};
			queue_cover = {};
			encoder_cover = {};
			barrier_pairs = 0;
			encoder_serial = 0;
			command = 0;
			render = false;
			split_required = false;
			reset_tables();
		}

		void reset_tables()
		{
			for (auto& table : tables)
			{
				table.buffers_known = 0;
				table.textures_known = 0;
			}
		}

		static u64 hash(const void* resource)
		{
			return (reinterpret_cast<uptr>(resource) >> 4) * 0x9E3779B97F4A7C15ull;
		}

		u32 find(const void* resource, bool texture)
		{
			if (entries.size() * 2 + 2 > index.size())
			{
				// Grow (and rehash)
				index.assign(std::max<usz>(1024, index.size() * 2), 0u);
				const u64 mask = index.size() - 1;
				for (u32 i = 0; i < entries.size(); ++i)
				{
					u64 h = hash(entries[i].resource) & mask;
					while (index[h])
					{
						h = (h + 1) & mask;
					}
					index[h] = i + 1;
				}
			}

			const u64 mask = index.size() - 1;
			u64 h = hash(resource) & mask;
			while (const u32 slot = index[h])
			{
				if (entries[slot - 1].resource == resource)
				{
					return slot - 1;
				}

				h = (h + 1) & mask;
			}

			entries.push_back({ .resource = resource, .texture = texture, .encoder = encoder_serial, .command = command });
			index[h] = ::size32(entries);
			return ::size32(entries) - 1;
		}

		// Moves the accesses of earlier commands / encoders to the sets they are checked in now
		void fold(hazard_entry& e)
		{
			if (e.encoder != encoder_serial)
			{
				// Everything of an earlier encoder: ordered by queue barriers from now on
				const bool texture = e.texture;
				for (auto [prior, current, group] : { std::tuple{ &e.read_prior, &e.read_current, &e.read_group }, std::tuple{ &e.write_prior, &e.write_current, &e.write_group } })
				{
					if (!current->empty() || !group->empty())
					{
						prune(*prior, queue_cover);
						add(*prior, *current, texture, e.encoder);
						add(*prior, *group, texture, e.encoder);
						current->clear();
						group->clear();
					}
				}

				e.encoder = encoder_serial;
				e.command = command;
				return;
			}

			if (e.command != command)
			{
				// Accesses of an earlier command of the open encoder: ordered by intra-encoder barriers
				add(e.read_current, e.read_group, e.texture, e.command);
				add(e.write_current, e.write_group, e.texture, e.command);
				e.read_group.clear();
				e.write_group.clear();
				e.command = command;
			}
		}

		// What `access` (by consumer classes `consumers`) must wait for among the accesses recorded in `e`
		void check(const hazard_entry& e, const gpu_access& access, bool always_overlaps, class_mask consumers, barrier_needs& needs) const
		{
			auto test = [&](const range_set& set) -> bool
			{
				return !set.empty() && (always_overlaps || overlaps(set, access.lo, access.hi, e.texture));
			};

			auto check_sets = [&](const range_set& prior, const range_set& current)
			{
				if (test(prior))
				{
					needs.queue |= uncovered(prior, consumers, queue_cover);
				}

				// Inside a render pass, accesses of the pass itself cannot be ordered (the renderer ends the pass for
				// feedback loops)
				if (!render && test(current))
				{
					needs.encoder |= uncovered(current, consumers, encoder_cover);
				}
			};

			check_sets(e.write_prior, e.write_current);
			if (access.write)
			{
				check_sets(e.read_prior, e.read_current);
			}
		}

		// Undeclared accesses of earlier commands: everything conflicts with them
		void check_unknown(class_mask consumers, barrier_needs& needs)
		{
			fold(unknown);
			check(unknown, gpu_access{}, true, consumers, needs);
		}

		// `external_pairs`: see hazard_entry::external_read
		void record(hazard_entry& e, const gpu_access& access, class_mask producers, u64 external_pairs)
		{
			// The `unknown` entry (no resource) records writes of everything
			const bool write = access.write || !e.resource;
			const u64 lo = e.resource ? access.lo : 0;
			const u64 hi = e.resource ? access.hi : u64{umax};
			add(write ? e.write_group : e.read_group, lo, hi, e.texture, producers, command);
			add(write ? e.write_all : e.read_all, lo, hi, e.texture, producers, 0);
			(write ? e.external_write : e.external_read) |= external_pairs;
		}

		// The two queue barriers every render pass begins with (encoder `encoder_serial`). A tile-based GPU bins all draws
		// of a pass before it shades them tile by tile, so the order of a pass against earlier work is fixed when it
		// begins: its fragment and tile work (attachment loads, shading, stores) waits for every stage of all earlier work,
		// its vertex work for every earlier stage except fragment and tile work, so that binning can overlap the shading
		// of the previous pass. What remains, the vertex stage reading what earlier fragment work wrote, gets a barrier
		// before the first draw (command_list::draw_access).
		void begin_pass(MTL4::RenderCommandEncoder* encoder)
		{
			constexpr class_mask non_fragment_classes = all_classes & ~attachment_classes;
			encoder->barrierAfterQueueStages(to_stages(non_fragment_classes), stages_render, MTL4::VisibilityOptionDevice);
			encoder->barrierAfterQueueStages(stages_attachment, stages_attachment, MTL4::VisibilityOptionDevice);

			cover_pairs(queue_cover, non_fragment_classes, render_classes, encoder_serial);
			cover_pairs(queue_cover, attachment_classes, attachment_classes, encoder_serial);
			barrier_pairs |= pair_bits(non_fragment_classes, render_classes) | pair_bits(attachment_classes, attachment_classes);
			queue_barriers += 2;
			pass_barriers += 2;
		}

		// Producer classes of every earlier encoder that are not yet ordered before later work of `consumers` (an
		// undeclared access conflicts with everything)
		class_mask uncovered_earlier(class_mask consumers) const
		{
			if (encoder_serial <= 1)
			{
				return 0; // No earlier encoder in this recording
			}

			const range_set earlier{ .stages = all_classes, .key = encoder_serial - 1 };
			return uncovered(earlier, consumers, queue_cover);
		}

		// Encodes the barriers `needs` asks for before work of the `consumers` classes
		void emit(MTL4::CommandEncoder* encoder, const barrier_needs& needs, class_mask consumers)
		{
			if (needs.queue)
			{
				encoder->barrierAfterQueueStages(to_stages(needs.queue), to_stages(consumers), MTL4::VisibilityOptionDevice);
				cover_pairs(queue_cover, needs.queue, consumers, encoder_serial);
				barrier_pairs |= pair_bits(needs.queue, consumers);
				queue_barriers++;
			}

			if (needs.encoder)
			{
				encoder->barrierAfterEncoderStages(to_stages(needs.encoder), to_stages(consumers), MTL4::VisibilityOptionDevice);
				cover_pairs(encoder_cover, needs.encoder, consumers, command);
				encoder_barriers++;
			}
		}

		// Entry indices of the accesses of the command being declared (reused)
		std::vector<u32> ids;
	};

	// ---- Telemetry --------------------------------------------------------------------------------------------------

	gpu_stats_t get_gpu_stats_and_reset()
	{
		auto& state = gpu_stats();
		gpu_stats_t stats{};
		{
			std::lock_guard lock(state.mutex);
			stats.busy_ns = std::exchange(state.busy_ns, 0);
		}
		stats.draw_render_passes = state.draw_render_passes.exchange(0);
		for (u32 reason = 0; reason < gpu_stats_t::reason_count; reason++)
		{
			for (u32 context = 0; context < gpu_stats_t::context_count; context++)
			{
				stats.draw_pass_ends[reason][context] = state.draw_pass_ends[reason][context].exchange(0);
			}
		}
		for (u32 context = 0; context < gpu_stats_t::context_count; context++)
		{
			stats.other_passes[context] = state.other_passes[context].exchange(0);
		}
		for (u32 event = 0; event < stats.pass_events.size(); event++)
		{
			stats.pass_events[event] = state.pass_events[event].exchange(0);
		}
		stats.attachment_load_bytes = state.attachment_load_bytes.exchange(0);
		stats.attachment_store_bytes = state.attachment_store_bytes.exchange(0);
		stats.feedback_splits = state.feedback_splits.exchange(0);
		for (u32 i = 0; i < stats.splits_by_reason.size(); i++)
		{
			stats.splits_by_reason[i] = state.splits_by_reason[i].exchange(0);
		}
		stats.feedback_reads_in_pass = state.feedback_reads_in_pass.exchange(0);
		stats.uploads_ahead = state.uploads_ahead.exchange(0);
		stats.uploads_inline = state.uploads_inline.exchange(0);
		stats.uploads_inline_split = state.uploads_inline_split.exchange(0);
		stats.submissions = state.submissions.exchange(0);
		stats.table_writes = state.table_writes.exchange(0);
		stats.table_writes_skipped = state.table_writes_skipped.exchange(0);
		stats.state_sets = state.state_sets.exchange(0);
		stats.state_sets_skipped = state.state_sets_skipped.exchange(0);
		stats.ordering_points = state.ordering_points.exchange(0);
		stats.ordering_points_free = state.ordering_points_free.exchange(0);
		stats.queue_barriers = state.queue_barriers.exchange(0);
		stats.pass_barriers = state.pass_barriers.exchange(0);
		stats.first_draw_barriers = state.first_draw_barriers.exchange(0);
		stats.late_barriers = state.late_barriers.exchange(0);
		stats.encoder_barriers = state.encoder_barriers.exchange(0);
		stats.full_barriers = state.full_barriers.exchange(0);
		stats.submissions_waited = state.submissions_waited.exchange(0);
		stats.submissions_overlapped = state.submissions_overlapped.exchange(0);
		return stats;
	}

	u64 peek_gpu_busy_ns()
	{
		auto& state = gpu_stats();
		std::lock_guard lock(state.mutex);
		return state.busy_ns;
	}

	void count_feedback_split(pass_split_reason reason)
	{
		auto& state = gpu_stats();
		state.feedback_splits++;
		state.splits_by_reason[static_cast<u32>(reason)]++;
	}

	void count_feedback_read_in_pass()
	{
		gpu_stats().feedback_reads_in_pass++;
	}

	void count_pass_event(pass_event event)
	{
		gpu_stats().pass_events[static_cast<u32>(event)]++;
	}

	u32 current_pass_context_index()
	{
		static_assert(static_cast<u32>(pass_context::count) == cpu_wait_stats_t::context_count);
		return static_cast<u32>(g_pass_context);
	}

	pass_context_scope::pass_context_scope(pass_context context)
		: m_previous(g_pass_context)
	{
		g_pass_context = context;
	}

	pass_context_scope::~pass_context_scope()
	{
		g_pass_context = m_previous;
	}

	std::string describe_render_passes(const gpu_stats_t& stats, u32 frames)
	{
		static constexpr std::array<const char*, gpu_stats_t::context_count> context_names =
		{
			"other", "surface setup", "texture setup", "clear", "blit", "query", "readback", "present"
		};

		static constexpr std::array<const char*, gpu_stats_t::reason_count> reason_names =
		{
			"framebuffer change", "clear", "feedback", "copy/compute", "other pass", "query pool", "submit"
		};

		const f64 scale = frames ? 1. / frames : 0.;

		// "a 1.0, b 2.0" for the non-zero contexts of `counts`
		const auto by_context = [&](const std::array<u64, gpu_stats_t::context_count>& counts)
		{
			std::string result;
			for (u32 context = 0; context < gpu_stats_t::context_count; context++)
			{
				if (counts[context])
				{
					fmt::append(result, "%s%s %.1f", result.empty() ? "" : ", ", context_names[context], counts[context] * scale);
				}
			}

			return result.empty() ? std::string("none") : result;
		};

		u64 other_total = 0;
		for (const u64 count : stats.other_passes)
		{
			other_total += count;
		}

		std::string ends;
		for (u32 reason = 0; reason < gpu_stats_t::reason_count; reason++)
		{
			u64 total = 0;
			for (const u64 count : stats.draw_pass_ends[reason])
			{
				total += count;
			}

			if (!total)
			{
				continue;
			}

			fmt::append(ends, "%s%s %.1f", ends.empty() ? "" : ", ", reason_names[reason], total * scale);

			if (const auto& counts = stats.draw_pass_ends[reason]; total != counts[static_cast<u32>(pass_context::other)])
			{
				fmt::append(ends, " (%s)", by_context(counts));
			}
		}

		const auto event = [&](pass_event e) { return stats.pass_events[static_cast<u32>(e)] * scale; };
		const auto split = [&](pass_split_reason r) { return stats.splits_by_reason[static_cast<u32>(r)] * scale; };

		constexpr f64 mib = 1024. * 1024.;
		return fmt::format("%.1f per frame: %.1f draw passes and %.1f other passes (%s); draw passes ended by %s; "
			"feedback splits %.1f (read after write %.1f, through a copy %.1f, write after read %.1f, "
			"vertex read of earlier fragment output %.1f, copy of the depth buffer for shader reads %.1f), "
			"%.1f feedback reads kept in the pass; "
			"no readback speculation for %.1f surfaces that stayed bound; "
			"clears: %.1f drawn as quads in the draw pass (scissored or masked; %.1f of them full-frame clears of some planes that kept the pass open), %.1f folded into a load action, "
			"%.1f clear-only passes; "
			"attachments retained across %.1f RSX layout changes (%.1f dropped again for a draw sampling them); "
			"attachment memory traffic %.0f MiB loaded and %.0f MiB stored per frame",
			(stats.draw_render_passes + other_total) * scale, stats.draw_render_passes * scale, other_total * scale, by_context(stats.other_passes),
			ends.empty() ? std::string("nothing") : ends,
			stats.feedback_splits * scale, split(pass_split_reason::read_after_write), split(pass_split_reason::read_through_copy),
			split(pass_split_reason::write_after_read), split(pass_split_reason::vertex_read),
			split(pass_split_reason::depth_copy),
			stats.feedback_reads_in_pass * scale,
			event(pass_event::readback_not_speculated),
			event(pass_event::clear_in_pass), event(pass_event::clear_kept_pass_open), event(pass_event::clear_folded), event(pass_event::clear_pass),
			event(pass_event::attachments_retained), event(pass_event::retained_attachment_sampled),
			stats.attachment_load_bytes * scale / mib, stats.attachment_store_bytes * scale / mib);
	}

	void count_image_upload(bool ahead_of_pass, bool ended_pass)
	{
		auto& state = gpu_stats();
		if (ahead_of_pass)
		{
			state.uploads_ahead++;
			return;
		}

		state.uploads_inline++;
		if (ended_pass)
		{
			state.uploads_inline_split++;
		}
	}

	// ---- Access descriptions --------------------------------------------------------------------------------------

	u64 subresource_mask(u32 base_level, u32 level_count, u32 base_layer, u32 layer_count)
	{
		if (!level_count || !layer_count)
		{
			return 0;
		}

		const u8 levels = axis_bits(base_level, level_count);
		const u64 layers = axis_bits(base_layer, layer_count);
		u64 mask = 0;
		for (u32 level = 0; level < 8; ++level)
		{
			if (levels & (1u << level))
			{
				mask |= layers << (level * 8);
			}
		}

		return mask;
	}

	gpu_access read_buffer(const mtl::buffer* buf, u64 offset, u64 length)
	{
		return buf ? buffer_access(buf->value(), offset, length, buf->size(), false) : gpu_access{};
	}

	gpu_access write_buffer(const mtl::buffer* buf, u64 offset, u64 length)
	{
		return buf ? buffer_access(buf->value(), offset, length, buf->size(), true) : gpu_access{};
	}

	gpu_access read_buffer(const MTL::Buffer* buf, u64 offset, u64 length)
	{
		return buf ? buffer_access(buf, offset, length, buf->length(), false) : gpu_access{};
	}

	gpu_access write_buffer(const MTL::Buffer* buf, u64 offset, u64 length)
	{
		return buf ? buffer_access(buf, offset, length, buf->length(), true) : gpu_access{};
	}

	gpu_access read_buffer(const mtl::buffer_view* view)
	{
		return view ? buffer_access(view->parent, view->offset, view->size, view->offset + view->size, false) : gpu_access{};
	}

	static gpu_access image_access(const mtl::image* img, u32 base_level, u32 level_count, u32 base_layer, u32 layer_count, bool write)
	{
		if (!img || !img->value)
		{
			return {};
		}

		const bool is_3d = img->type() == MTL::TextureType3D;
		return
		{
			.resource = img->value,
			.lo = subresource_mask(base_level, level_count, is_3d ? 0 : base_layer, is_3d ? u32{umax} : layer_count),
			.kind = gpu_access::texture_subresources,
			.write = write
		};
	}

	gpu_access read_image(const mtl::image* img, u32 base_level, u32 level_count, u32 base_layer, u32 layer_count)
	{
		return image_access(img, base_level, level_count, base_layer, layer_count, false);
	}

	gpu_access write_image(const mtl::image* img, u32 base_level, u32 level_count, u32 base_layer, u32 layer_count)
	{
		return image_access(img, base_level, level_count, base_layer, layer_count, true);
	}

	gpu_access read_image(const mtl::image_view* view)
	{
		if (!view || !view->parent_texture())
		{
			return {};
		}

		// The view's own fields only: the image object may have given the texture and its views to a clone
		const bool is_3d = view->info.type == MTL::TextureType3D;
		return
		{
			.resource = view->parent_texture(),
			.lo = subresource_mask(view->info.base_level, view->info.level_count, is_3d ? 0 : view->info.base_layer, is_3d ? u32{umax} : view->info.layer_count),
			.kind = gpu_access::texture_subresources
		};
	}

	gpu_access read_texture(const MTL::Texture* tex)
	{
		return texture_access(tex, false);
	}

	gpu_access write_texture(const MTL::Texture* tex)
	{
		return texture_access(tex, true);
	}

	gpu_access write_object(const void* object)
	{
		return object ? gpu_access{ .resource = object, .lo = 0, .hi = 1, .kind = gpu_access::buffer_range, .write = true } : gpu_access{};
	}

	// ---- command_list -----------------------------------------------------------------------------------------------

	command_list::command_list()
		: m_hazards(std::make_unique<hazard_state>())
	{
	}

	command_list::~command_list()
	{
		destroy();
	}

	void command_list::create(const render_device& dev, MTL4::CommandQueue* queue, timeline& tl, std::string_view label)
	{
		ensure(!m_commands);
		autorelease_scope pool;

		m_device = &dev;
		m_queue = queue;
		m_timeline = &tl;
		m_label = std::string(label);

		m_commands = dev.handle()->newCommandBuffer();
		ensure(m_commands, "Metal: failed to create MTL4CommandBuffer");

		auto desc = ref(MTL4::ArgumentTableDescriptor::alloc()->init());
		desc->setMaxBufferBindCount(gpu_capabilities::max_buffers_per_stage);
		desc->setMaxTextureBindCount(argument_table_shadow::max_textures);
		desc->setMaxSamplerStateBindCount(gpu_capabilities::max_samplers_per_stage);
		desc->setInitializeBindings(true);
		desc->setSupportAttributeStrides(false);

		for (u32 i = 0; i < table_count; ++i)
		{
			NS::Error* error = nullptr;
			m_argument_tables[i] = dev.handle()->newArgumentTable(desc.get(), &error);
			if (!(m_argument_tables[i]))
			{
				fmt::throw_exception("Metal: failed to create argument table: %s", to_string(error));
			}
		}

		m_submit_fence.owner = m_timeline;
	}

	void command_list::destroy()
	{
		if (!m_commands)
		{
			return;
		}

		if (m_is_pending && !wait(5'000'000))
		{
			rsx_log.error("Metal: command list '%s' still in flight at destruction (GPU hang?)", m_label);
		}

		// Committed together with this list: the wait above covered it
		m_prologue.reset();
		m_prologue_recorded = false;
		m_deferred_clears.clear();

		for (auto& table : m_argument_tables)
		{
			if (table)
			{
				table->release();
				table = nullptr;
			}
		}

		m_commands->release();
		m_commands = nullptr;

		// A recording that was never committed (the allocators of committed ones are in the pool)
		if (m_allocator)
		{
			command_allocators::discard(m_allocator);
			m_allocator = nullptr;
		}
	}

	void command_list::begin()
	{
		ensure(!m_is_open);

		if (m_is_pending)
		{
			// Recording state (fence, reset ids) describes one submission at a time
			wait();
		}

		if (m_allocator)
		{
			// The previous recording was ended and dropped without being committed
			command_allocators::recycle(m_allocator);
		}

		m_allocator = command_allocators::acquire(*m_device);
		m_commands->beginCommandBuffer(m_allocator);

		// A label is an NSString and a message per recording; only worth it when a debugging tool shows it
		if (!m_label.empty() && debug_labels_enabled())
		{
			autorelease_scope pool;
			m_commands->setLabel(ns_str(m_label));
		}

		m_is_open = true;
		m_deferred_clears.clear(); // Only a recording that was dropped without end() can leave some
		m_folded_clears = 0;

		// Nothing recorded yet. Accesses of earlier submissions are checked when this one is submitted.
		m_hazards->reset();

		// The tables keep their contents, but every recording starts from a known state (cheap: one full write each)
		for (auto& shadow : m_argument_table_shadows)
		{
			shadow.invalidate();
		}
		m_render_bindings = {};
		m_compute_bindings = {};

		// A prologue of a recording that was ended but never submitted is dropped with it (end() closed it)
		ensure(!m_prologue || !m_prologue->m_is_open);
		m_prologue_recorded = false;
	}

	void command_list::end()
	{
		ensure(m_is_open);
		end_encoder(pass_end_reason::submit);
		flush_deferred_clears();
		m_commands->endCommandBuffer();
		m_is_open = false;

		if (m_prologue && m_prologue->m_is_open)
		{
			m_prologue->end();
		}
	}

	command_list* command_list::prologue()
	{
		if (!can_record_prologue())
		{
			return nullptr;
		}

		if (!m_prologue)
		{
			m_prologue = std::make_unique<command_list>();
			m_prologue->create(*m_device, m_queue, *m_timeline, m_label + " prologue");
			m_prologue->m_is_prologue = true;
		}

		if (!m_prologue_recorded)
		{
			// The prologue's previous recording was committed with this list's previous submission, which begin()
			// waited for before this recording started: its allocator can be reset.
			m_prologue->begin();
			m_prologue_recorded = true;
		}

		return m_prologue.get();
	}

	void command_list::wait_for_conflicting_submissions()
	{
		// Under the submit mutex. Everything this recording (and its prologue) accessed, against what earlier submissions
		// to the queue that may still run accessed: a conflict not ordered by a queue barrier of a submission in between
		// makes the queue wait for the conflicting submission (every later submission then runs after it too). Only the
		// (producer, consumer) pairs the recording's own barriers do not order are tested (hazard_entry::external_read):
		// every render pass begins with barriers on all earlier work of the queue, earlier submissions included, so its
		// attachments and fragment reads need no wait; compute commands and vertex-stage reads do.
		auto& qs = get_queue_state(m_queue);
		retire_completed(qs);

		if (qs.in_flight.empty())
		{
			return;
		}

		u64 wait_serial = 0;
		auto check_recording = [&](const hazard_state& hs)
		{
			u64 external_all = 0;

			for (const auto& e : hs.entries)
			{
				external_all |= e.external_read | e.external_write;

				const auto found = qs.records.find(e.resource);
				if (found == qs.records.end())
				{
					continue;
				}

				const queue_record& rec = found->second;
				auto test = [&](const range_set& set, const range_set& access)
				{
					return !set.empty() && !access.empty() && (e.texture ? (set.lo & access.lo) != 0 : (set.lo < access.hi && access.lo < set.hi));
				};

				// Read or write after write
				const u64 after_write = (test(rec.write, e.read_all) ? e.external_read : 0) | (test(rec.write, e.write_all) ? e.external_write : 0);
				if (after_write && uncovered_pairs(rec.write, after_write, qs.cover))
				{
					wait_serial = std::max(wait_serial, rec.write.key);
				}

				// Write after read
				if (test(rec.read, e.write_all) && uncovered_pairs(rec.read, e.external_write, qs.cover))
				{
					wait_serial = std::max(wait_serial, rec.read.key);
				}
			}

			// Undeclared accesses of an earlier submission conflict with everything, and undeclared accesses of this
			// one with everything earlier
			if (!qs.unknown.empty() && external_all && uncovered_pairs(qs.unknown, external_all, qs.cover))
			{
				wait_serial = std::max(wait_serial, qs.unknown.key);
			}

			if (!hs.unknown.write_all.empty())
			{
				wait_serial = qs.serial;
			}
		};

		check_recording(*m_hazards);
		if (m_prologue_recorded)
		{
			check_recording(*m_prologue->m_hazards);
		}

		auto& stats = gpu_stats();
		if (wait_serial <= qs.completed)
		{
			stats.submissions_overlapped++;
			return;
		}

		for (const auto& submission : qs.in_flight)
		{
			if (submission.serial >= wait_serial)
			{
				// Queue-ordered: every later submission waits too, so everything up to it counts as completed
				m_queue->wait(submission.tl->handle(), submission.value);
				qs.completed = submission.serial;
				break;
			}
		}

		stats.submissions_waited++;
		retire_completed(qs);
	}

	void command_list::record_submission(u64 value, const command_list* prologue)
	{
		// Under the submit mutex, after the commit
		auto& qs = get_queue_state(m_queue);
		const u64 serial = ++qs.serial;
		qs.in_flight.push_back({ serial, m_timeline, value });

		auto merge = [&](const hazard_state& hs)
		{
			// A queue barrier a -> b of this submission orders a-work of every earlier submission before b-work of this and
			// every later one
			for (u64 bits = hs.barrier_pairs; bits; bits &= bits - 1)
			{
				const u32 bit = static_cast<u32>(std::countr_zero(bits));
				auto& cover = qs.cover[bit / class_count][bit % class_count];
				cover = std::max(cover, serial);
			}

			for (const auto& e : hs.entries)
			{
				if (e.read_all.empty() && e.write_all.empty())
				{
					continue;
				}

				auto& rec = qs.records[e.resource];
				rec.texture = e.texture;

				for (auto [set, access] : { std::pair{ &rec.read, &e.read_all }, std::pair{ &rec.write, &e.write_all } })
				{
					if (!access->empty())
					{
						prune(*set, qs.cover);
						add(*set, *access, e.texture, serial);
					}
				}
			}

			if (!hs.unknown.write_all.empty())
			{
				prune(qs.unknown, qs.cover);
				add(qs.unknown, 0, u64{umax}, false, hs.unknown.write_all.stages, serial);
			}
		};

		merge(*m_hazards);
		if (prologue)
		{
			merge(*prologue->m_hazards);
		}
	}

	u64 command_list::submit(const submit_info_t& info)
	{
		if (m_is_open)
		{
			end();
		}

		ensure(!m_is_pending);

		// Residency changes made while recording must be visible before the GPU runs this work.
		g_render_device->commit_residency();

		std::lock_guard lock(submit_mutex());

		if (info.wait_drawable)
		{
			m_queue->wait(info.wait_drawable);
		}

		if (info.wait_event)
		{
			m_queue->wait(info.wait_event, info.wait_value);
		}

		// Earlier submissions to this queue that may still run and access what this one accesses
		wait_for_conflicting_submissions();

		// Commit feedback: GPU time (telemetry) and GPU errors (page faults, timeouts), which are silent otherwise
		auto options = ref(MTL4::CommitOptions::alloc()->init());
		options->addFeedbackHandler(MTL4::CommitFeedbackHandlerFunction(&on_commit_feedback));

		// The prologue goes first in the same commit: the waits above, the timeline signal below (fence, GC event id of
		// the renderer's lists) and the commit feedback cover both buffers, and the producer barriers that end every
		// encoder of the prologue order all of this list's work after it.
		const MTL4::CommandBuffer* buffers[2] = {};
		u32 buffer_count = 0;
		command_list* committed_prologue = nullptr;

		if (m_prologue_recorded)
		{
			ensure(!m_prologue->m_is_open);
			buffers[buffer_count++] = m_prologue->m_commands;
			committed_prologue = m_prologue.get();
			m_prologue_recorded = false;
		}

		ensure(m_allocator);
		buffers[buffer_count++] = m_commands;
		m_queue->commit(buffers, buffer_count, options.get());

		if (info.signal_drawable)
		{
			m_queue->signalDrawable(info.signal_drawable);
		}

		const u64 value = m_timeline->signal(m_queue);

		// What this submission accesses, for the submissions that follow it
		record_submission(value, committed_prologue);

		// The allocators are reused once the GPU has executed this submission (retired in commit order: the submit lock
		// is still held)
		if (committed_prologue)
		{
			command_allocators::retire(std::exchange(committed_prologue->m_allocator, nullptr), *m_timeline, value);
			committed_prologue->flush_encoding_counters();
		}

		flush_encoding_counters();
		gpu_stats().submissions++;

		command_allocators::retire(std::exchange(m_allocator, nullptr), *m_timeline, value);
		m_submit_fence.value = value;
		m_submit_fence.flushed = true;
		m_is_pending = true;
		return value;
	}

	void command_list::flush_encoding_counters()
	{
		auto& state = gpu_stats();
		u64 writes = 0, writes_skipped = 0;
		for (auto& shadow : m_argument_table_shadows)
		{
			writes += std::exchange(shadow.writes, 0);
			writes_skipped += std::exchange(shadow.writes_skipped, 0);
		}

		state.table_writes += writes;
		state.table_writes_skipped += writes_skipped;
		state.state_sets += std::exchange(m_state_counters.sets, 0);
		state.state_sets_skipped += std::exchange(m_state_counters.skipped, 0);

		auto& hs = *m_hazards;
		state.ordering_points += std::exchange(hs.ordering_points, 0);
		state.ordering_points_free += std::exchange(hs.ordering_points_free, 0);
		state.queue_barriers += std::exchange(hs.queue_barriers, 0);
		state.pass_barriers += std::exchange(hs.pass_barriers, 0);
		state.first_draw_barriers += std::exchange(hs.first_draw_barriers, 0);
		state.late_barriers += std::exchange(hs.late_barriers, 0);
		state.encoder_barriers += std::exchange(hs.encoder_barriers, 0);
		state.full_barriers += std::exchange(hs.full_barriers, 0);
	}
	void command_list::end_encoding(MTL4::CommandEncoder* encoder, MTL::Stages stages)
	{
		if (m_is_prologue)
		{
			// Producer barrier: every later pass of the queue (all work of the list this prologue is committed with,
			// including work recorded before the prologue work) waits for this pass
			encoder->barrierAfterStages(stages, stages_all, MTL4::VisibilityOptionDevice);
		}

		encoder->endEncoding();
	}

	u32 command_list::fold_deferred_clears(MTL4::RenderPassDescriptor* desc, std::array<std::pair<MTL::RenderPassAttachmentDescriptor*, MTL::LoadAction>, 10>& restore, u32& restore_count)
	{
		const u32 area_width = static_cast<u32>(desc->renderTargetWidth());
		const u32 area_height = static_cast<u32>(desc->renderTargetHeight());
		u32 folded = 0;

		// The attachment clears the plane of a pending clear of its texture if it covers exactly the cleared area
		const auto fold = [&](MTL::RenderPassAttachmentDescriptor* attachment, u32 plane, u32 bit)
		{
			const MTL::Texture* texture = attachment->texture();
			if (!texture || attachment->level() != 0 || attachment->slice() != 0 || attachment->depthPlane() != 0)
			{
				return;
			}

			// Render area: renderTargetWidth/Height, or the attachment size when they are not set
			const u32 width = area_width ? area_width : static_cast<u32>(texture->width());
			const u32 height = area_height ? area_height : static_cast<u32>(texture->height());

			for (auto& entry : m_deferred_clears)
			{
				if (entry.texture.get() != texture || !(entry.planes & plane) || entry.width != width || entry.height != height)
				{
					continue;
				}

				entry.planes &= ~plane;

				if (const auto load = attachment->loadAction(); load != MTL::LoadActionClear)
				{
					// Load or DontCare: the pass starts from the cleared contents. A pass that clears on its own
					// overwrites the pending clear.
					restore[restore_count++] = { attachment, load };
					attachment->setLoadAction(MTL::LoadActionClear);

					switch (plane)
					{
					case aspect_depth: static_cast<MTL::RenderPassDepthAttachmentDescriptor*>(attachment)->setClearDepth(entry.value.depth); break;
					case aspect_stencil: static_cast<MTL::RenderPassStencilAttachmentDescriptor*>(attachment)->setClearStencil(entry.value.stencil); break;
					default: static_cast<MTL::RenderPassColorAttachmentDescriptor*>(attachment)->setClearColor(entry.value.color); break;
					}
				}

				folded |= bit;
				count_pass_event(pass_event::clear_folded);
				return;
			}
		};

		// Colour attachments are dense in every descriptor the backend builds
		for (u32 index = 0; index < 8; ++index)
		{
			const auto attachment = desc->colorAttachments()->object(index);
			if (!attachment->texture())
			{
				break;
			}

			fold(attachment, aspect_color, 1u << index);
		}

		fold(desc->depthAttachment(), aspect_depth, 1u << 8);
		fold(desc->stencilAttachment(), aspect_stencil, 1u << 9);

		std::erase_if(m_deferred_clears, [](const deferred_clear& entry) { return !entry.planes; });
		return folded;
	}

	MTL4::RenderCommandEncoder* command_list::begin_render_pass(MTL4::RenderPassDescriptor* desc, bool draw_pass)
	{
		ensure(m_is_open);
		end_encoder(pass_end_reason::other_pass);

		std::array<std::pair<MTL::RenderPassAttachmentDescriptor*, MTL::LoadAction>, 10> restore{};
		u32 restore_count = 0;
		u32 folded = 0;

		if (!m_deferred_clears.empty())
		{
			folded = fold_deferred_clears(desc, restore, restore_count);

			// Clears this pass does not fold (other textures, planes or areas) go first, as clear-only passes
			flush_deferred_clears();
		}

		auto encoder = open_render_encoder(desc, draw_pass);
		m_folded_clears = folded;

		// Metal copied the descriptor
		for (u32 i = 0; i < restore_count; ++i)
		{
			restore[i].first->setLoadAction(restore[i].second);
		}

		return encoder;
	}

	MTL4::RenderCommandEncoder* command_list::open_render_encoder(MTL4::RenderPassDescriptor* desc, bool draw_pass)
	{
		// Encoders are returned autoreleased (+0). Keep our own reference so the encoder survives any autorelease pool
		// drained while it is open; end_render_pass() drops it. Passes are never suspended/resumed (DESIGN.md §4).
		m_render_encoder = m_commands->renderCommandEncoder(desc);
		ensure(m_render_encoder, "Metal: failed to begin render pass");
		m_render_encoder->retain();
		m_pass_serial = ++g_pass_serial;
		m_render_bindings = {}; // A new encoder starts without pipeline state or argument tables
		m_draw_pass = draw_pass;

		auto& stats = gpu_stats();
		(draw_pass ? stats.draw_render_passes : stats.other_passes[static_cast<u32>(g_pass_context)])++;

		auto& hs = *m_hazards;
		hs.render = true;
		hs.split_required = false;
		hs.reset_tables();
		hs.encoder_serial++;
		hs.command = 0;

		// Before anything else of the pass: the pass barriers order it against all earlier work of the queue (every stage
		// of the attachment work; the vertex stage against all but fragment work). Nothing recorded after the pass's first
		// draw could (tile-based GPU).
		hs.begin_pass(m_render_encoder);

		// The attachments (and resolve targets): loaded (or cleared), written and stored by the pass. `desc` is the
		// descriptor as begun: attachments whose load action is a folded deferred clear are written like any other, and
		// the clear-only passes of flush_deferred_clears() come through here too. In-pass clear quads only write
		// attachments of the open pass. Unused colour attachment descriptors are created (autoreleased) on access.
		// One walk over the descriptor serves both the hazard declarations and the attachment traffic telemetry (each
		// property is an objc_msgSend; passes begin tens to hundreds of times per frame).
		std::array<gpu_access, 20> attachments{};
		u32 count = 0;
		u64 load_bytes = 0, store_bytes = 0;
		{
			autorelease_scope pool;
			const u64 area = u64{ desc->renderTargetWidth() } * desc->renderTargetHeight();

			// Returns false for an unused attachment slot
			auto add_attachment = [&](const MTL::RenderPassAttachmentDescriptor* attachment, bool stencil_plane)
			{
				if (!attachment)
				{
					return false;
				}

				const MTL::Texture* texture = attachment->texture();
				if (!texture)
				{
					return false;
				}

				const NS::UInteger level = attachment->level();
				if (const auto access = attachment_access(texture, level, attachment->slice()); access.kind != gpu_access::none)
				{
					attachments[count++] = access;
				}

				if (const MTL::Texture* resolve = attachment->resolveTexture())
				{
					if (const auto access = attachment_access(resolve, attachment->resolveLevel(), attachment->resolveSlice()); access.kind != gpu_access::none)
					{
						attachments[count++] = access;
					}
				}

				// Memory traffic of the pass: loads at its start, stores at its end
				const u64 pixels = area ? area : u64{ std::max<NS::UInteger>(texture->width() >> level, 1) } *
					std::max<NS::UInteger>(texture->height() >> level, 1);
				const u64 bytes = pixels * texture->sampleCount() * get_attachment_texel_size(texture->pixelFormat(), stencil_plane);

				if (attachment->loadAction() == MTL::LoadActionLoad)
				{
					load_bytes += bytes;
				}

				if (const auto store = attachment->storeAction(); store == MTL::StoreActionStore || store == MTL::StoreActionStoreAndMultisampleResolve)
				{
					store_bytes += bytes;
				}

				return true;
			};

			// Colour attachments are dense in every descriptor the backend builds; stop at the first unused slot (an
			// unused attachment descriptor is created, autoreleased, on access)
			const auto colors = desc->colorAttachments();
			for (u32 i = 0; i < 8 && add_attachment(colors->object(i), false); ++i)
			{
			}

			add_attachment(desc->depthAttachment(), false);
			add_attachment(desc->stencilAttachment(), true);
		}

		stats.attachment_load_bytes += load_bytes;
		stats.attachment_store_bytes += store_bytes;

		// Earlier accesses of the attachments are ordered by the pass barriers (fragment/tile work after everything).
		// Written by every render stage of the pass, as far as later work is concerned.
		for (u32 i = 0; i < count; ++i)
		{
			auto& entry = hs.entries[hs.find(attachments[i].resource, attachments[i].kind == gpu_access::texture_subresources)];
			hs.fold(entry);
			hs.record(entry, attachments[i], render_classes, 0);
		}

		return m_render_encoder;
	}

	void command_list::end_render_pass(pass_end_reason reason)
	{
		if (m_render_encoder)
		{
			if (m_draw_pass)
			{
				gpu_stats().draw_pass_ends[static_cast<u32>(reason)][static_cast<u32>(g_pass_context)]++;
				m_draw_pass = false;
			}

			end_encoding(m_render_encoder, stages_render);
			m_render_encoder->release();
			m_render_encoder = nullptr;
			m_hazards->render = false;
		}
	}

	void command_list::draw_access(const gpu_access& access, MTL::Stages stages)
	{
		if (access.kind == gpu_access::none)
		{
			return;
		}

		ensure(m_render_encoder, "Draw access declared outside of a render pass");

		auto& hs = *m_hazards;
		const class_mask consumers = to_classes(stages);
		barrier_needs needs{};
		u32 id = umax;

		if (access.kind == gpu_access::unknown)
		{
			// Everything earlier, in every stage (what the pass barriers do not order yet)
			needs.queue = hs.uncovered_earlier(consumers);
			hs.full_barriers++;
		}
		else
		{
			id = hs.find(access.resource, access.kind == gpu_access::texture_subresources);
			hs.fold(hs.entries[id]);
			hs.check(hs.entries[id], access, false, consumers, needs);
		}

		hs.check_unknown(consumers, needs);

		if (needs.queue)
		{
			// The pass barriers order the fragment stage after everything earlier: this is the vertex stage waiting for
			// earlier fragment work (or an undeclared access). A tile-based GPU bins every draw of the pass before it
			// shades any, so a barrier only orders the pass while no draw was recorded in it (no pipeline state set yet,
			// render_encoder_bindings). After that, the renderer's draw pass is split: the draw goes to a new pass, whose
			// first bind declares the access again and gets the barrier.
			if (!m_render_bindings.program_uid)
			{
				hs.emit(m_render_encoder, needs, consumers);
				hs.first_draw_barriers++;
			}
			else if (m_draw_pass)
			{
				hs.split_required = true;
			}
			else
			{
				// Other passes bind nothing that fragment work writes to their vertex stage (overlays, copies, clears):
				// a programming error. The barrier is the best that can still be done.
				static atomic_t<bool> s_reported = false;
				if (!s_reported.exchange(true))
				{
					rsx_log.error("Metal: a draw of a helper render pass reads, after the pass's first draw, what earlier work wrote "
						"and the pass barriers do not order (the barrier may not order the pass on a tile-based GPU)");
				}

				hs.emit(m_render_encoder, needs, consumers);
				hs.late_barriers++;
			}
		}

		if (id != umax)
		{
			hs.record(hs.entries[id], access, consumers, draw_external_pairs(consumers));
		}
		else
		{
			hs.record(hs.unknown, access, consumers, draw_external_pairs(consumers));
		}
	}

	bool command_list::pass_split_required() const
	{
		return m_render_encoder && m_hazards->split_required;
	}

	void command_list::table_access(argument_table_slot table, u32 index, bool texture, const gpu_access& access)
	{
		ensure(table == table_vertex || table == table_fragment);

		auto& decl = m_hazards->tables[table];
		if (texture)
		{
			ensure(index < argument_table_shadow::max_textures);
			const u64 bit = u64{1} << index;
			if ((decl.textures_known & bit) && decl.textures[index] == access)
			{
				return;
			}

			decl.textures[index] = access;
			decl.textures_known |= bit;
		}
		else
		{
			ensure(index < gpu_capabilities::max_buffers_per_stage);
			const u32 bit = u32{1} << index;
			if ((decl.buffers_known & bit) && decl.buffers[index] == access)
			{
				return;
			}

			decl.buffers[index] = access;
			decl.buffers_known |= bit;
		}

		draw_access(access, table == table_vertex ? MTL::StageVertex : MTL::StageFragment);
	}

	MTL4::ComputeCommandEncoder* command_list::compute_command(std::span<const gpu_access> accesses, MTL::Stages stage, bool new_command)
	{
		ensure(m_is_open);

		auto& hs = *m_hazards;
		bool opened = false;

		if (!m_compute_encoder)
		{
			opened = true;
			end_render_pass(pass_end_reason::transfer);

			// Pending clears are recorded before the command (they were requested before it)
			flush_deferred_clears();

			// Autoreleased (+0): retained here, released in end_encoder()
			m_compute_encoder = m_commands->computeCommandEncoder();
			ensure(m_compute_encoder, "Metal: failed to begin compute encoder");
			m_compute_encoder->retain();
			m_compute_bindings = {}; // A new encoder starts without pipeline state or argument table

			hs.encoder_serial++;
			hs.command = 0;
			hs.encoder_cover = {};
		}

		if (new_command || !hs.command)
		{
			new_command = true;
			hs.command++;
			hs.ordering_points++;
		}

		const class_mask consumers = to_classes(stage);
		barrier_needs needs{};
		bool declared = false;
		bool undeclared = false;

		hs.ids.clear();
		for (const auto& access : accesses)
		{
			u32 id = umax;

			if (access.kind == gpu_access::unknown)
			{
				undeclared = true;
			}
			else if (access.kind != gpu_access::none)
			{
				id = hs.find(access.resource, access.kind == gpu_access::texture_subresources);
				hs.fold(hs.entries[id]);
				hs.check(hs.entries[id], access, false, consumers, needs);
				declared = true;
			}

			hs.ids.push_back(id);
		}

		if (declared || undeclared)
		{
			hs.check_unknown(consumers, needs);
		}

		if (undeclared)
		{
			// Full barrier: every earlier access, in every stage (earlier encoders) and every earlier command of this one
			needs.queue = all_classes;
			if (hs.command > 1)
			{
				needs.encoder = compute_classes;
			}

			hs.full_barriers++;
		}

		// Conservative ordering against earlier encoders (DESIGN.md §3): a compute encoder begins with a consumer barrier
		// on every stage of all earlier work of the queue (earlier command buffers included). Render passes do not declare
		// everything they touch (attachment loads/stores, tile memory), so a compute command overlapping a render pass is
		// only safe behind this barrier; a copy racing a pass shows as tile-shaped stale content (GoldenEye's 32x32-pixel
		// stale tiles). The declarations are still recorded: later render passes (vertex-stage reads) and submissions
		// (draws, external work) are checked against them.
		if (opened)
		{
			// Before every stage of the encoder (a later command may dispatch after a first blit, or vice versa). It covers
			// whatever the declarations asked of earlier encoders.
			hs.emit(m_compute_encoder, { .queue = all_classes }, compute_classes);
			needs.queue = 0;
		}

		// Within the encoder, every command after the first waits for the earlier commands of the encoder, whatever the
		// hazard check found (only the parts of one command, blit_concurrent(), stay concurrent). Ordering compute
		// commands by their declarations alone was tried (2026-10-05) and brought GoldenEye 007: Reloaded's stale
		// 32x32-pixel tiles back in colour and depth (the scene target in tiles of an earlier frame, the first-person
		// weapon depth-tested against garbage): some declaration of the depth-stencil scatter/gather or readback
		// chains does not cover every byte those commands touch. Until every compute command is proven to declare
		// exactly what it accesses, the encoder-order barrier stays. The declarations are still recorded for what is
		// checked against them (render passes' vertex-stage reads, submissions).
		if (new_command && hs.command > 1)
		{
			needs.encoder = compute_classes;
		}

		hs.emit(m_compute_encoder, needs, consumers);
		if (new_command && !opened && !needs.queue && !needs.encoder)
		{
			hs.ordering_points_free++;
		}

		// Nothing to check against earlier submissions: the barrier the encoder began with orders every command of the
		// encoder after all earlier work of the queue, earlier submissions included (a consumer barrier applies to the
		// work of all previous passes of the queue: "Synchronizing passes with consumer barriers")
		const u64 external = 0;
		for (usz i = 0; i < accesses.size(); ++i)
		{
			if (const u32 id = hs.ids[i]; id != umax)
			{
				hs.record(hs.entries[id], accesses[i], consumers, external);
			}
			else if (accesses[i].kind == gpu_access::unknown)
			{
				hs.record(hs.unknown, accesses[i], consumers, external);
			}
		}

		return m_compute_encoder;
	}

	MTL4::ComputeCommandEncoder* command_list::blit(std::initializer_list<gpu_access> accesses)
	{
		return compute_command({ accesses.begin(), accesses.size() }, MTL::StageBlit, true);
	}

	MTL4::ComputeCommandEncoder* command_list::blit(std::span<const gpu_access> accesses)
	{
		return compute_command(accesses, MTL::StageBlit, true);
	}

	MTL4::ComputeCommandEncoder* command_list::dispatch(std::span<const gpu_access> accesses)
	{
		return compute_command(accesses, MTL::StageDispatch, true);
	}

	MTL4::ComputeCommandEncoder* command_list::blit_concurrent(std::initializer_list<gpu_access> accesses)
	{
		return compute_command({ accesses.begin(), accesses.size() }, MTL::StageBlit, false);
	}

	MTL4::ComputeCommandEncoder* command_list::compute_encoder() const
	{
		return ensure(m_compute_encoder, "No compute command was declared");
	}

	void command_list::external_work(std::span<const gpu_access> accesses)
	{
		ensure(m_is_open);
		end_encoder();

		// The framework's passes come between our encoders: one serial for all of them, every stage (checked in full
		// against earlier submissions)
		auto& hs = *m_hazards;
		hs.encoder_serial++;
		hs.command = 0;
		const u64 external = pair_bits(all_classes, all_classes);

		for (const auto& access : accesses)
		{
			if (access.kind == gpu_access::unknown)
			{
				hs.fold(hs.unknown);
				hs.record(hs.unknown, access, all_classes, external);
			}
			else if (access.kind != gpu_access::none)
			{
				const u32 id = hs.find(access.resource, access.kind == gpu_access::texture_subresources);
				hs.fold(hs.entries[id]);
				hs.record(hs.entries[id], access, all_classes, external);
			}
		}
	}

	void command_list::end_encoder(pass_end_reason reason)
	{
		end_render_pass(reason);

		if (m_compute_encoder)
		{
			end_encoding(m_compute_encoder, stages_compute);
			m_compute_encoder->release();
			m_compute_encoder = nullptr;
		}
	}

	command_list::encoder_type command_list::active_encoder() const
	{
		if (m_render_encoder) return encoder_type::render;
		if (m_compute_encoder) return encoder_type::compute;
		return encoder_type::none;
	}

	void command_list::defer_clear(MTL::Texture* texture, u32 planes, u32 width, u32 height, const attachment_clear_value& value)
	{
		ensure(m_is_open && texture && planes);

		// Ordered after everything recorded so far
		end_encoder(pass_end_reason::other_pass);

		for (auto& entry : m_deferred_clears)
		{
			if (entry.texture.get() != texture)
			{
				continue;
			}

			if (entry.width == width && entry.height == height)
			{
				// Same area: the later values win for the planes both clear
				if (planes & aspect_color) entry.value.color = value.color;
				if (planes & aspect_depth) entry.value.depth = value.depth;
				if (planes & aspect_stencil) entry.value.stencil = value.stencil;
				entry.planes |= planes;
				return;
			}

			// Another area of the same texture: keep the order of the two clears (rare)
			flush_deferred_clears();
			break;
		}

		m_deferred_clears.push_back({ ref<MTL::Texture>::retain(texture), planes, width, height, value });
	}

	bool command_list::has_deferred_clear(const MTL::Texture* texture, u32 planes) const
	{
		return std::any_of(m_deferred_clears.begin(), m_deferred_clears.end(), [texture, planes](const deferred_clear& entry)
		{
			return entry.texture.get() == texture && (entry.planes & planes);
		});
	}

	void command_list::flush_deferred_clears()
	{
		if (m_deferred_clears.empty())
		{
			return;
		}

		ensure(!m_render_encoder && !m_compute_encoder);

		autorelease_scope pool;
		auto pending = std::move(m_deferred_clears);
		m_deferred_clears.clear();

		while (!pending.empty())
		{
			// One pass per render area and sample count: every colour texture of that kind (up to 8) and the first
			// depth-stencil texture
			const auto& first = pending.front();
			const u32 width = first.width, height = first.height;
			const NS::UInteger samples = first.texture->sampleCount();

			auto desc = ref(MTL4::RenderPassDescriptor::alloc()->init());
			u32 color_count = 0;
			bool has_depth_stencil = false;

			const auto setup = [](MTL::RenderPassAttachmentDescriptor* attachment, const MTL::Texture* texture)
			{
				attachment->setTexture(texture);
				attachment->setLoadAction(MTL::LoadActionClear);
				attachment->setStoreAction(MTL::StoreActionStore);
			};

			for (auto it = pending.begin(); it != pending.end();)
			{
				if (it->width != width || it->height != height || it->texture->sampleCount() != samples)
				{
					++it;
					continue;
				}

				if (it->planes & aspect_color)
				{
					if (color_count == 8)
					{
						++it;
						continue;
					}

					auto attachment = desc->colorAttachments()->object(color_count++);
					setup(attachment, it->texture.get());
					attachment->setClearColor(it->value.color);
				}
				else
				{
					if (has_depth_stencil)
					{
						++it;
						continue;
					}

					has_depth_stencil = true;

					// A combined depth-stencil texture is attached as both, like in every other pass; a plane that is
					// not cleared is loaded
					const auto format = it->texture->pixelFormat();
					if (is_depth_format(format))
					{
						auto attachment = desc->depthAttachment();
						setup(attachment, it->texture.get());
						attachment->setClearDepth(it->value.depth);

						if (!(it->planes & aspect_depth))
						{
							attachment->setLoadAction(MTL::LoadActionLoad);
						}
					}

					if (is_stencil_format(format))
					{
						auto attachment = desc->stencilAttachment();
						setup(attachment, it->texture.get());
						attachment->setClearStencil(it->value.stencil);

						if (!(it->planes & aspect_stencil))
						{
							attachment->setLoadAction(MTL::LoadActionLoad);
						}
					}
				}

				// The descriptor holds its own reference to the texture
				it = pending.erase(it);
			}

			desc->setRenderTargetWidth(width);
			desc->setRenderTargetHeight(height);
			desc->setDefaultRasterSampleCount(samples);

			open_render_encoder(desc.get(), false);
			end_render_pass(pass_end_reason::other_pass);
			count_pass_event(pass_event::clear_pass);
		}
	}

	MTL4::CommandBuffer* command_list::handle()
	{
		// Encoders of other components (MetalFX) run in recording order: pending clears go first
		if (m_is_open)
		{
			flush_deferred_clears();
		}

		return m_commands;
	}

	void command_list::push_debug_group(std::string_view name)
	{
		autorelease_scope pool;
		if (m_render_encoder) m_render_encoder->pushDebugGroup(ns_str(name));
		else if (m_compute_encoder) m_compute_encoder->pushDebugGroup(ns_str(name));
		else if (m_is_open) m_commands->pushDebugGroup(ns_str(name));
	}

	void command_list::pop_debug_group()
	{
		if (m_render_encoder) m_render_encoder->popDebugGroup();
		else if (m_compute_encoder) m_compute_encoder->popDebugGroup();
		else if (m_is_open) m_commands->popDebugGroup();
	}

	bool command_list::poke()
	{
		if (!m_is_pending)
		{
			return true;
		}

		if (m_submit_fence.signaled())
		{
			m_is_pending = false;
			m_submit_fence.reset();
			return true;
		}

		return false;
	}

	bool command_list::wait(u64 timeout_us)
	{
		if (!m_is_pending)
		{
			return true;
		}

		if (!m_submit_fence.wait(timeout_us))
		{
			return false;
		}

		m_is_pending = false;
		m_submit_fence.reset();
		return true;
	}

	void forget_queue_submissions(const MTL4::CommandQueue* queue)
	{
		std::lock_guard lock(submit_mutex());
		auto& states = queue_states();
		std::erase_if(states, [queue](const auto& state) { return state->queue == queue; });
	}
}
