#pragma once

#include "util/video_source.h"
#include "util/atomic.hpp"
#include "Utilities/mutex.h"

#include <QMovie>
#include <QBuffer>
#include <QFutureWatcher>
#include <QMediaPlayer>
#include <QVideoSink>
#include <QVideoFrame>
#include <QPixmap>
#include <QTimer>

class qt_video_source : public video_source
{
public:
	qt_video_source(bool is_emulation = false);
	virtual ~qt_video_source();

	void set_iso_path(const std::string& iso_path) override;
	void set_video_path(const std::string& video_path, bool video_in_archive) override;
	void set_audio_path(const std::string& audio_path, bool audio_in_archive) override;
	const QString& video_path() const { return m_video_path; }
	const QString& audio_path() const { return m_audio_path; }

	void get_image(std::vector<u8>& data, int& w, int& h, int& ch, int& bpp) override;
	bool has_new() const override { return m_has_new; }

	void set_active(bool active) override;
	bool get_active() const override { return m_active; }

	void start_movie_timer();
	void start_movie();
	void stop_movie();

	void start_audio();
	void stop_audio();

	QPixmap get_movie_image(const QVideoFrame& frame) const;

	void image_change_callback(const QVideoFrame& frame = {}) const;
	void set_image_change_callback(const std::function<void(const QVideoFrame&)>& func);

protected:
	void init_movie();

	// Reads the video and sound stored in the ISO on a worker thread. Returns true while they are being read: start_movie()
	// is called again once they are available
	bool load_archive_media();
	void drop_archive_media();

	shared_mutex m_image_mutex;

	atomic_t<bool> m_active = false;
	atomic_t<bool> m_has_new = false;

	bool m_video_in_archive = false;
	bool m_audio_in_archive = false;

	QString m_video_path;
	QString m_audio_path;
	u32 m_audio_instance_index = 0;
	u32 m_video_timer_timeout_ms = 0;
	std::string m_iso_path; // path of the source archive
	QByteArray m_video_data{};
	QImage m_image{};
	std::vector<u8> m_image_path;

	std::unique_ptr<QTimer> m_video_timer;
	std::unique_ptr<QBuffer> m_video_buffer;
	std::unique_ptr<QMediaPlayer> m_media_player;
	std::unique_ptr<QVideoSink> m_video_sink;
	std::unique_ptr<QMovie> m_movie;

	// Content of the video and sound files stored in the ISO (m_iso_path), read through the ISO media cache
	using archive_media = std::pair<std::shared_ptr<const std::vector<u8>>, std::shared_ptr<const std::vector<u8>>>;
	enum class archive_media_state
	{
		none,
		loading,
		loaded
	};
	archive_media_state m_archive_media_state = archive_media_state::none;
	archive_media m_archive_media{};
	std::unique_ptr<QFutureWatcher<archive_media>> m_archive_media_watcher;

	std::function<void(const QVideoFrame&)> m_image_change_callback = nullptr;

	friend class qt_video_source_wrapper;
};

// Wrapper for emulator usage
class qt_video_source_wrapper : public video_source
{
public:
	qt_video_source_wrapper() : video_source() {}
	virtual ~qt_video_source_wrapper();

	void set_iso_path(const std::string& iso_path) override;
	void set_video_path(const std::string& video_path, bool video_in_archive) override;
	void set_audio_path(const std::string& audio_path, bool audio_in_archive) override;
	void set_active(bool active) override;
	bool get_active() const override;
	bool has_new() const override { return m_qt_video_source && m_qt_video_source->has_new(); }
	void get_image(std::vector<u8>& data, int& w, int& h, int& ch, int& bpp) override;

private:
	void init_video_source();

	std::unique_ptr<qt_video_source> m_qt_video_source;
};
