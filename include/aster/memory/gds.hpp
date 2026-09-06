#pragma once
#include <string>

#include "aster/common/status.hpp"
#include "aster/hal/device.hpp"

namespace aster::memory {

// File to memory transfers. With cuFile the read lands directly in VRAM; otherwise pread into a host
// buffer then a HAL copy. The interface is identical so the memory manager does not care.
class FileReader {
 public:
  virtual ~FileReader() = default;
  virtual Status ReadToHost(const std::string& path, uint64_t offset, uint32_t bytes, void* dst) = 0;
  virtual Status ReadToDevice(const std::string& path, uint64_t offset, uint32_t bytes, void* dst_device,
                              hal::Stream s = {}) = 0;
  virtual bool direct_to_device() const = 0;
  virtual const char* name() const = 0;
};

class PosixFileReader final : public FileReader {
 public:
  explicit PosixFileReader(hal::DevicePtr dev) : dev_(std::move(dev)) {}
  Status ReadToHost(const std::string& path, uint64_t offset, uint32_t bytes, void* dst) override;
  Status ReadToDevice(const std::string& path, uint64_t offset, uint32_t bytes, void* dst_device, hal::Stream s) override;
  bool direct_to_device() const override { return false; }
  const char* name() const override { return "posix"; }

 private:
  hal::DevicePtr dev_;
};

#if ASTER_HAVE_CUFILE
class CuFileReader final : public FileReader {
 public:
  static Result<std::unique_ptr<CuFileReader>> Open(hal::DevicePtr dev);
  ~CuFileReader() override;
  Status ReadToHost(const std::string& path, uint64_t offset, uint32_t bytes, void* dst) override;
  Status ReadToDevice(const std::string& path, uint64_t offset, uint32_t bytes, void* dst_device, hal::Stream s) override;
  bool direct_to_device() const override { return true; }
  const char* name() const override { return "cufile"; }

 private:
  explicit CuFileReader(hal::DevicePtr dev) : dev_(std::move(dev)), fallback_(dev_) {}
  hal::DevicePtr dev_;
  PosixFileReader fallback_;
};
#endif

std::unique_ptr<FileReader> MakeFileReader(hal::DevicePtr dev, bool prefer_gds);
Status WriteFile(const std::string& path, const void* data, size_t bytes, bool fsync);
Status AppendFile(const std::string& path, const void* data, size_t bytes, bool fsync);
Result<std::vector<uint8_t>> ReadWholeFile(const std::string& path);
Result<uint64_t> FileSize(const std::string& path);
Status EnsureDir(const std::string& path);
bool FileExists(const std::string& path);

}  // namespace aster::memory
