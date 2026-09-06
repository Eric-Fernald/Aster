#pragma once
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"
#include "aster/exchange/partitioner.hpp"
#include "aster/hal/device.hpp"

namespace aster::exchange {

// Transport for shuffling batches between ranks. In process transport runs on one node with N
// workers; NCCL runs across GPUs on one node (NVLink/NVSwitch aware); UCX crosses nodes with
// GPUDirect RDMA. All three share this interface so the exchange operator does not care.
class Transport {
 public:
  virtual ~Transport() = default;
  virtual const char* name() const = 0;
  virtual int rank() const = 0;
  virtual int world_size() const = 0;
  virtual Status Send(int to, const RecordBatch& batch) = 0;
  virtual Result<RecordBatchPtr> Recv(int from) = 0;
  virtual Status Barrier() = 0;
  virtual double link_bytes_per_sec(int peer) const = 0;
};

class InProcessTransport final : public Transport {
 public:
  struct Group;
  static std::vector<std::shared_ptr<InProcessTransport>> CreateGroup(int world_size);
  const char* name() const override { return "inproc"; }
  int rank() const override { return rank_; }
  int world_size() const override { return world_; }
  Status Send(int to, const RecordBatch& batch) override;
  Result<RecordBatchPtr> Recv(int from) override;
  Status Barrier() override;
  double link_bytes_per_sec(int) const override { return 50e9; }

 private:
  InProcessTransport(std::shared_ptr<Group> g, int rank, int world) : group_(std::move(g)), rank_(rank), world_(world) {}
  std::shared_ptr<Group> group_;
  int rank_, world_;
};

#if ASTER_HAVE_NCCL
class NcclTransport final : public Transport {
 public:
  static Result<std::shared_ptr<NcclTransport>> Create(hal::DevicePtr dev, int rank, int world, const void* unique_id);
  ~NcclTransport() override;
  const char* name() const override { return "nccl"; }
  int rank() const override { return rank_; }
  int world_size() const override { return world_; }
  Status Send(int to, const RecordBatch& batch) override;
  Result<RecordBatchPtr> Recv(int from) override;
  Status Barrier() override;
  double link_bytes_per_sec(int peer) const override;

 private:
  NcclTransport(hal::DevicePtr dev, int rank, int world) : dev_(std::move(dev)), rank_(rank), world_(world) {}
  hal::DevicePtr dev_;
  void* comm_ = nullptr;
  hal::Stream stream_;
  int rank_, world_;
};
#endif

#if ASTER_HAVE_UCX
class UcxTransport final : public Transport {
 public:
  static Result<std::shared_ptr<UcxTransport>> Create(hal::DevicePtr dev, int rank, const std::vector<std::string>& peer_addrs);
  ~UcxTransport() override;
  const char* name() const override { return "ucx"; }
  int rank() const override { return rank_; }
  int world_size() const override { return world_; }
  Status Send(int to, const RecordBatch& batch) override;
  Result<RecordBatchPtr> Recv(int from) override;
  Status Barrier() override;
  double link_bytes_per_sec(int) const override { return 25e9; }

 private:
  UcxTransport(hal::DevicePtr dev, int rank, int world) : dev_(std::move(dev)), rank_(rank), world_(world) {}
  struct Impl;
  std::shared_ptr<Impl> impl_;
  hal::DevicePtr dev_;
  int rank_, world_;
};
#endif

enum class ExchangeStrategy { HashShuffle, BroadcastSmaller, Local };

// Shuffle or broadcast one input across ranks; chooses broadcast for the smaller side on PCIe only nodes.
class ExchangeOperator {
 public:
  ExchangeOperator(std::shared_ptr<Transport> t, std::vector<int> keys);
  static ExchangeStrategy Choose(uint64_t left_bytes, uint64_t right_bytes, double link_bps, bool nvswitch);
  Result<std::vector<RecordBatchPtr>> Shuffle(const std::vector<RecordBatchPtr>& local);
  Result<std::vector<RecordBatchPtr>> Broadcast(const std::vector<RecordBatchPtr>& local);
  uint64_t bytes_sent() const { return bytes_sent_; }

 private:
  std::shared_ptr<Transport> transport_;
  HashPartitioner partitioner_;
  uint64_t bytes_sent_ = 0;
};

std::vector<uint8_t> SerializeBatch(const RecordBatch& b);
Result<RecordBatchPtr> DeserializeBatch(const uint8_t* data, size_t len);

}  // namespace aster::exchange
