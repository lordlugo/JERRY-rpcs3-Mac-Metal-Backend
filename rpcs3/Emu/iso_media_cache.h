#pragma once

#include "Loader/ISO.h"
#include "Utilities/File.h"
#include "Utilities/mutex.h"
#include "util/types.hpp"

#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Process-wide, bounded cache of parsed disc images (ISO files, and disc volumes read through their raw device) and of
// whole files read from them (icons, hover videos and sounds of the game list).
//
// Parsing an ISO walks its whole file system, and on a disc volume every read is a read of the raw device, so neither
// may happen on the UI thread: the game enumeration adds the archives it builds anyway (on its worker threads), and the
// game list loads what it needs through read_file() on worker threads, which reuse those archives and keep the bytes.
//
// An entry is used only while its source is unchanged: the identity of the path (the raw device backing a disc volume,
// size and times) is compared on every lookup, which costs a few metadata syscalls and no disc read. The game list
// empties the cache when it rescans the drives.
namespace iso_media_cache
{
	// Most recently used archives kept (a parsed file system takes ~200 bytes per file of the disc)
	constexpr usz max_archives = 16;

	// Bytes of file content kept over all the archives (least recently used files are dropped first)
	constexpr usz max_file_bytes = 64 * 1024 * 1024;

	// Larger files are returned but not kept
	constexpr usz max_kept_file_size = max_file_bytes / 4;

	using file_data = std::shared_ptr<const std::vector<u8>>;

	struct source_id
	{
		std::string disc_source; // what a disc volume is read from: its image file or its raw device (empty for an ISO file)
		bool is_raw_device = false; // "disc_source" is a raw device (see fs::get_optical_disc_source)
		u64 size = 0;
		s64 mtime = 0;
		s64 ctime = 0;
		bool is_directory = false;

		bool operator==(const source_id&) const = default;
	};

	struct entry
	{
		std::string path;
		source_id id;
		std::shared_ptr<iso_archive> archive;
		std::map<std::string, file_data> files;
	};

	struct cache_state
	{
		shared_mutex mutex;
		std::list<entry> entries; // most recently used first
		usz file_bytes = 0;
	};

	inline cache_state& get_state()
	{
		static cache_state s_state;
		return s_state;
	}

	// Identity of the ISO file or disc volume at "path" (false if it does not exist). Metadata only: no disc read
	inline bool get_source_id(const std::string& path, source_id& id)
	{
		fs::stat_t stat{};

		if (path.empty() || !fs::get_stat(path, stat))
		{
			return false;
		}

		id = {};

		// A disc volume is identified by what it is read from (its image file, or its raw device): a different disc mounted
		// at the same mount point is another source
		if (std::string source; fs::get_optical_disc_source(path, &source, &id.is_raw_device))
		{
			id.disc_source = std::move(source);
		}

		// The access time is left out: reading the source changes it
		id.size = stat.size;
		id.mtime = stat.mtime;
		id.ctime = stat.ctime;
		id.is_directory = stat.is_directory;
		return true;
	}

	namespace detail
	{
		inline void drop_entry(cache_state& state, std::list<entry>::iterator it)
		{
			for (const auto& [name, data] : it->files)
			{
				state.file_bytes -= data->size();
			}

			state.entries.erase(it);
		}

		// Entry of "path" whose source is still "id", moved to the front (a stale entry is dropped). Needs the lock
		inline entry* find(cache_state& state, const std::string& path, const source_id& id)
		{
			for (auto it = state.entries.begin(); it != state.entries.end(); it++)
			{
				if (it->path != path)
				{
					continue;
				}

				if (it->id != id)
				{
					drop_entry(state, it);
					return nullptr;
				}

				state.entries.splice(state.entries.begin(), state.entries, it);
				return &state.entries.front();
			}

			return nullptr;
		}

		// Needs the lock
		inline void insert(cache_state& state, const std::string& path, const source_id& id, std::shared_ptr<iso_archive> archive)
		{
			if (entry* e = find(state, path, id))
			{
				if (e->archive != archive)
				{
					// Parsed again by another thread: the files read through the old archive are still valid (same source)
					e->archive = std::move(archive);
				}

				return;
			}

			state.entries.push_front(entry{ .path = path, .id = id, .archive = std::move(archive), .files = {} });

			while (state.entries.size() > max_archives)
			{
				drop_entry(state, std::prev(state.entries.end()));
			}
		}

		// Parses the ISO at "path" (disc I/O). Nothing is logged if it is not an ISO
		inline std::shared_ptr<iso_archive> parse(const std::string& path, const source_id& id)
		{
			if (!is_iso_file(path))
			{
				return nullptr;
			}

			auto archive = std::make_shared<iso_archive>(path);

			if (!archive->is_valid())
			{
				return nullptr;
			}

			cache_state& state = get_state();
			std::lock_guard lock(state.mutex);
			insert(state, path, id, archive);
			return archive;
		}
	}

	// Adds an archive the caller has already parsed from "path" (the game enumeration does)
	inline void add_archive(const std::string& path, std::shared_ptr<iso_archive> archive)
	{
		source_id id{};

		if (!archive || !archive->is_valid() || !get_source_id(path, id))
		{
			return;
		}

		cache_state& state = get_state();
		std::lock_guard lock(state.mutex);
		detail::insert(state, path, id, std::move(archive));
	}

	// Parsed archive of the ISO at "path", or nullptr if it is not an ISO. Parses it on a miss (disc I/O): not for the UI
	// thread
	inline std::shared_ptr<iso_archive> get_archive(const std::string& path)
	{
		source_id id{};

		if (!get_source_id(path, id))
		{
			return nullptr;
		}

		{
			cache_state& state = get_state();
			std::lock_guard lock(state.mutex);

			if (entry* e = detail::find(state, path, id))
			{
				return e->archive;
			}
		}

		return detail::parse(path, id);
	}

	// Same result as "is_iso_file(path, nullptr, is_raw_device)", without reading the disc when the archive is cached
	inline bool is_iso(const std::string& path, bool* is_raw_device = nullptr)
	{
		if (source_id id{}; get_source_id(path, id))
		{
			cache_state& state = get_state();
			std::lock_guard lock(state.mutex);

			if (detail::find(state, path, id))
			{
				if (is_raw_device)
				{
					*is_raw_device = id.is_raw_device;
				}

				return true;
			}
		}

		return is_iso_file(path, nullptr, is_raw_device);
	}

	// Whole content of the file "file_path" of the ISO at "iso_path", or nullptr if it is missing or empty. Reads it (and
	// parses the ISO) on a miss (disc I/O): not for the UI thread
	inline file_data read_file(const std::string& iso_path, const std::string& file_path)
	{
		source_id id{};

		if (file_path.empty() || !get_source_id(iso_path, id))
		{
			return nullptr;
		}

		cache_state& state = get_state();
		std::shared_ptr<iso_archive> archive;

		{
			std::lock_guard lock(state.mutex);

			if (entry* e = detail::find(state, iso_path, id))
			{
				if (const auto found = e->files.find(file_path); found != e->files.end())
				{
					return found->second;
				}

				archive = e->archive;
			}
		}

		if (!archive && !(archive = detail::parse(iso_path, id)))
		{
			return nullptr;
		}

		const auto file = archive->open(file_path);

		if (!file)
		{
			return nullptr;
		}

		const u64 size = file->size();

		if (size == 0)
		{
			return nullptr;
		}

		auto data = std::make_shared<std::vector<u8>>(size);

		if (file->read_at(0, data->data(), size) != size)
		{
			return nullptr;
		}

		file_data result = std::move(data);

		if (size > max_kept_file_size)
		{
			return result;
		}

		std::lock_guard lock(state.mutex);

		// Keep the bytes only if the entry is still the one they were read from
		entry* e = detail::find(state, iso_path, id);

		if (!e || e->archive != archive)
		{
			return result;
		}

		if (const auto [it, inserted] = e->files.emplace(file_path, result); !inserted)
		{
			// Read concurrently by another thread
			return it->second;
		}

		state.file_bytes += size;

		// Drop the files of the least recently used archives first (never the one just read)
		for (auto it = state.entries.rbegin(); it != state.entries.rend() && state.file_bytes > max_file_bytes; it++)
		{
			for (auto f = it->files.begin(); f != it->files.end() && state.file_bytes > max_file_bytes;)
			{
				if (f->second == result)
				{
					f++;
					continue;
				}

				state.file_bytes -= f->second->size();
				f = it->files.erase(f);
			}
		}

		return result;
	}

	// Drops everything (the game list rescans the drives). Data already handed out stays valid
	inline void clear()
	{
		cache_state& state = get_state();
		std::lock_guard lock(state.mutex);
		state.entries.clear();
		state.file_bytes = 0;
	}
}
