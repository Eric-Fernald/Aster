#include "aster/memory/gds.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "aster/common/log.hpp"

#if defined(_WIN32)
#include <io.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if ASTER_HAVE_CUFILE
#include <cufile.h>
#endif

namespace aster::memory {

namespace fs = std::filesystem;

Status PosixFileReader::ReadToHost(const std::string& path, uint64_t offset, uint32_t bytes, void* dst) {
#if defined(_WIN32)
  std::ifstream f(path, std::ios::binary);
  if (!f) return Status::IoError("open " + path);
  f.seekg(static_cast<std::streamoff>(offset));
  f.read(static_cast<char*>(dst), bytes);
  if (static_cast<uint32_t>(f.gcount()) != bytes) return Status::IoError("short read " + path);
  return Status::OK();
#else
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return Status::IoError("open " + path);
  uint32_t done = 0;
  while (done < bytes) {
    ssize_t n = ::pread(fd, static_cast<char*>(dst) + done, bytes - done, static_cast<off_t>(offset + done));
    if (n <= 0) { ::close(fd); return Status::IoError("pread " + path); }
    done += static_cast<uint32_t>(n);
  }
  ::close(fd);
  return Status::OK();
#endif
}

Status PosixFileReader::ReadToDevice(const std::string& path, uint64_t offset, uint32_t bytes, void* dst_device,
                                     hal::Stream s) {
  if (!dev_) return Status::Invalid("no device for staged read");
  auto staging = dev_->MakePinnedBuffer(bytes);
  if (!staging) return Status::OutOfMemory("staging buffer");
  ASTER_RETURN_NOT_OK(ReadToHost(path, offset, bytes, staging->data()));
  ASTER_RETURN_NOT_OK(dev_->CopyHostToDevice(dst_device, staging->data(), bytes, s));
  return dev_->Synchronize(s);
}

#if ASTER_HAVE_CUFILE
Result<std::unique_ptr<CuFileReader>> CuFileReader::Open(hal::DevicePtr dev) {
  CUfileError_t e = cuFileDriverOpen();
  if (e.err != CU_FILE_SUCCESS) return Status::NotSupported("cuFileDriverOpen failed");
  return std::unique_ptr<CuFileReader>(new CuFileReader(std::move(dev)));
}

CuFileReader::~CuFileReader() { cuFileDriverClose(); }

Status CuFileReader::ReadToHost(const std::string& path, uint64_t offset, uint32_t bytes, void* dst) {
  return fallback_.ReadToHost(path, offset, bytes, dst);
}

Status CuFileReader::ReadToDevice(const std::string& path, uint64_t offset, uint32_t bytes, void* dst_device,
                                  hal::Stream s) {
  int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) return fallback_.ReadToDevice(path, offset, bytes, dst_device, s);
  CUfileDescr_t descr{};
  descr.handle.fd = fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  CUfileHandle_t h;
  if (cuFileHandleRegister(&h, &descr).err != CU_FILE_SUCCESS) {
    ::close(fd);
    return fallback_.ReadToDevice(path, offset, bytes, dst_device, s);
  }
  ssize_t n = cuFileRead(h, dst_device, bytes, static_cast<off_t>(offset), 0);
  cuFileHandleDeregister(h);
  ::close(fd);
  if (n != static_cast<ssize_t>(bytes)) return Status::IoError("cuFileRead short read " + path);
  return Status::OK();
}
#endif

std::unique_ptr<FileReader> MakeFileReader(hal::DevicePtr dev, bool prefer_gds) {
#if ASTER_HAVE_CUFILE
  if (prefer_gds && dev && dev->backend() == hal::Backend::Cuda) {
    auto r = CuFileReader::Open(dev);
    if (r.ok()) return std::move(r.value());
    ASTER_LOG(Warn, "GDS unavailable, using posix staged reads");
  }
#else
  (void)prefer_gds;
#endif
  return std::make_unique<PosixFileReader>(std::move(dev));
}

Status EnsureDir(const std::string& path) {
  std::error_code ec;
  fs::create_directories(path, ec);
  if (ec && !fs::is_directory(path)) return Status::IoError("mkdir " + path + ": " + ec.message());
  return Status::OK();
}

bool FileExists(const std::string& path) { return fs::exists(path); }

Result<uint64_t> FileSize(const std::string& path) {
  std::error_code ec;
  auto n = fs::file_size(path, ec);
  if (ec) return Status::NotFound(path);
  return static_cast<uint64_t>(n);
}

namespace {
Status WriteImpl(const std::string& path, const void* data, size_t bytes, bool append, bool fsync) {
#if defined(_WIN32)
  std::ofstream f(path, std::ios::binary | (append ? std::ios::app : std::ios::trunc));
  if (!f) return Status::IoError("open " + path);
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  f.flush();
  (void)fsync;
  return f ? Status::OK() : Status::IoError("write " + path);
#else
  int flags = O_WRONLY | O_CREAT | (append ? O_APPEND : O_TRUNC);
  int fd = ::open(path.c_str(), flags, 0644);
  if (fd < 0) return Status::IoError("open " + path);
  size_t done = 0;
  while (done < bytes) {
    ssize_t n = ::write(fd, static_cast<const char*>(data) + done, bytes - done);
    if (n <= 0) { ::close(fd); return Status::IoError("write " + path); }
    done += static_cast<size_t>(n);
  }
  if (fsync && ::fdatasync(fd) != 0) { ::close(fd); return Status::IoError("fsync " + path); }
  ::close(fd);
  return Status::OK();
#endif
}
}  // namespace

Status WriteFile(const std::string& path, const void* data, size_t bytes, bool fsync) {
  return WriteImpl(path, data, bytes, false, fsync);
}

Status AppendFile(const std::string& path, const void* data, size_t bytes, bool fsync) {
  return WriteImpl(path, data, bytes, true, fsync);
}

Result<std::vector<uint8_t>> ReadWholeFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return Status::NotFound(path);
  auto n = static_cast<size_t>(f.tellg());
  std::vector<uint8_t> out(n);
  f.seekg(0);
  if (n && !f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n))) return Status::IoError("read " + path);
  return out;
}

}  // namespace aster::memory
