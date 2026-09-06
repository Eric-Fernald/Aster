#include "aster/exchange/exchange.hpp"

#include <cstring>
#include <map>

#include "aster/ingest/wal.hpp"

#if ASTER_HAVE_NCCL
#include <nccl.h>
#endif
#if ASTER_HAVE_UCX
#include <ucp/api/ucp.h>
#endif

namespace aster::exchange {

std::vector<uint8_t> SerializeBatch(const RecordBatch& b) { return ingest::WriteAheadLog::SerializeBatch("", b); }

Result<RecordBatchPtr> DeserializeBatch(const uint8_t* data, size_t len) {
  ASTER_ASSIGN_OR_RETURN(auto rec, ingest::WriteAheadLog::DeserializeBatch(data, len));
  return rec.batch;
}

struct InProcessTransport::Group {
  int world;
  std::mutex mu;
  std::condition_variable cv;
  std::map<std::pair<int, int>, std::deque<RecordBatchPtr>> queues;  // (from, to)
  int barrier_count = 0;
  int barrier_gen = 0;
};

std::vector<std::shared_ptr<InProcessTransport>> InProcessTransport::CreateGroup(int world_size) {
  auto g = std::make_shared<Group>();
  g->world = world_size;
  std::vector<std::shared_ptr<InProcessTransport>> out;
  for (int r = 0; r < world_size; ++r) out.push_back(std::shared_ptr<InProcessTransport>(new InProcessTransport(g, r, world_size)));
  return out;
}

Status InProcessTransport::Send(int to, const RecordBatch& batch) {
  {
    std::lock_guard<std::mutex> lk(group_->mu);
    group_->queues[{rank_, to}].push_back(std::make_shared<RecordBatch>(batch));
  }
  group_->cv.notify_all();
  return Status::OK();
}

Result<RecordBatchPtr> InProcessTransport::Recv(int from) {
  std::unique_lock<std::mutex> lk(group_->mu);
  auto& q = group_->queues[{from, rank_}];
  group_->cv.wait(lk, [&] { return !q.empty(); });
  RecordBatchPtr b = q.front();
  q.pop_front();
  return b;
}

Status InProcessTransport::Barrier() {
  std::unique_lock<std::mutex> lk(group_->mu);
  int gen = group_->barrier_gen;
  if (++group_->barrier_count == group_->world) {
    group_->barrier_count = 0;
    ++group_->barrier_gen;
    group_->cv.notify_all();
  } else {
    group_->cv.wait(lk, [&] { return group_->barrier_gen != gen; });
  }
  return Status::OK();
}

#if ASTER_HAVE_NCCL
Result<std::shared_ptr<NcclTransport>> NcclTransport::Create(hal::DevicePtr dev, int rank, int world, const void* unique_id) {
  std::shared_ptr<NcclTransport> t(new NcclTransport(std::move(dev), rank, world));
  ncclUniqueId id;
  std::memcpy(&id, unique_id, sizeof id);
  ncclComm_t comm;
  if (ncclCommInitRank(&comm, world, id, rank) != ncclSuccess) return Status::Internal("ncclCommInitRank failed");
  t->comm_ = comm;
  ASTER_ASSIGN_OR_RETURN(t->stream_, t->dev_->CreateStream());
  return t;
}

NcclTransport::~NcclTransport() {
  if (comm_) ncclCommDestroy(static_cast<ncclComm_t>(comm_));
  dev_->DestroyStream(stream_);
}

Status NcclTransport::Send(int to, const RecordBatch& batch) {
  std::vector<uint8_t> bytes = SerializeBatch(batch);
  uint64_t len = bytes.size();
  auto d_len = dev_->MakeDeviceBuffer(sizeof len, stream_);
  auto d_buf = dev_->MakeDeviceBuffer(len, stream_);
  ASTER_RETURN_NOT_OK(dev_->CopyHostToDevice(d_len->data(), &len, sizeof len, stream_));
  ASTER_RETURN_NOT_OK(dev_->CopyHostToDevice(d_buf->data(), bytes.data(), len, stream_));
  auto s = static_cast<cudaStream_t>(stream_.handle);
  if (ncclSend(d_len->data(), sizeof len, ncclUint8, to, static_cast<ncclComm_t>(comm_), s) != ncclSuccess) return Status::Internal("ncclSend len");
  if (ncclSend(d_buf->data(), len, ncclUint8, to, static_cast<ncclComm_t>(comm_), s) != ncclSuccess) return Status::Internal("ncclSend payload");
  return dev_->Synchronize(stream_);
}

Result<RecordBatchPtr> NcclTransport::Recv(int from) {
  uint64_t len = 0;
  auto d_len = dev_->MakeDeviceBuffer(sizeof len, stream_);
  auto s = static_cast<cudaStream_t>(stream_.handle);
  if (ncclRecv(d_len->data(), sizeof len, ncclUint8, from, static_cast<ncclComm_t>(comm_), s) != ncclSuccess) return Status::Internal("ncclRecv len");
  ASTER_RETURN_NOT_OK(dev_->CopyDeviceToHost(&len, d_len->data(), sizeof len, stream_));
  ASTER_RETURN_NOT_OK(dev_->Synchronize(stream_));
  auto d_buf = dev_->MakeDeviceBuffer(len, stream_);
  if (ncclRecv(d_buf->data(), len, ncclUint8, from, static_cast<ncclComm_t>(comm_), s) != ncclSuccess) return Status::Internal("ncclRecv payload");
  std::vector<uint8_t> bytes(len);
  ASTER_RETURN_NOT_OK(dev_->CopyDeviceToHost(bytes.data(), d_buf->data(), len, stream_));
  ASTER_RETURN_NOT_OK(dev_->Synchronize(stream_));
  return DeserializeBatch(bytes.data(), bytes.size());
}

Status NcclTransport::Barrier() {
  auto d = dev_->MakeDeviceBuffer(4, stream_);
  if (ncclAllReduce(d->data(), d->data(), 1, ncclInt, ncclSum, static_cast<ncclComm_t>(comm_), static_cast<cudaStream_t>(stream_.handle)) != ncclSuccess)
    return Status::Internal("nccl barrier");
  return dev_->Synchronize(stream_);
}

double NcclTransport::link_bytes_per_sec(int peer) const {
  int access = 0;
  cudaDeviceCanAccessPeer(&access, rank_, peer);
  return access ? 900e9 : 25e9;  // NVLink/NVSwitch vs PCIe
}
#endif

#if ASTER_HAVE_UCX
struct UcxTransport::Impl {
  ucp_context_h context = nullptr;
  ucp_worker_h worker = nullptr;
  std::vector<ucp_ep_h> endpoints;
};

Result<std::shared_ptr<UcxTransport>> UcxTransport::Create(hal::DevicePtr dev, int rank, const std::vector<std::string>& peer_addrs) {
  std::shared_ptr<UcxTransport> t(new UcxTransport(std::move(dev), rank, static_cast<int>(peer_addrs.size())));
  t->impl_ = std::make_shared<Impl>();
  ucp_params_t params{};
  params.field_mask = UCP_PARAM_FIELD_FEATURES;
  params.features = UCP_FEATURE_TAG | UCP_FEATURE_RMA;
  ucp_config_t* config;
  if (ucp_config_read(nullptr, nullptr, &config) != UCS_OK) return Status::Internal("ucp_config_read");
  ucs_status_t st = ucp_init(&params, config, &t->impl_->context);
  ucp_config_release(config);
  if (st != UCS_OK) return Status::Internal("ucp_init");
  ucp_worker_params_t wp{};
  wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  wp.thread_mode = UCS_THREAD_MODE_SINGLE;
  if (ucp_worker_create(t->impl_->context, &wp, &t->impl_->worker) != UCS_OK) return Status::Internal("ucp_worker_create");
  for (const auto& addr : peer_addrs) {
    ucp_ep_params_t ep{};
    ep.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep.address = reinterpret_cast<const ucp_address_t*>(addr.data());
    ucp_ep_h h = nullptr;
    if (!addr.empty() && ucp_ep_create(t->impl_->worker, &ep, &h) != UCS_OK) return Status::Internal("ucp_ep_create");
    t->impl_->endpoints.push_back(h);
  }
  return t;
}

UcxTransport::~UcxTransport() {
  if (!impl_) return;
  for (auto ep : impl_->endpoints) if (ep) ucp_ep_destroy(ep);
  if (impl_->worker) ucp_worker_destroy(impl_->worker);
  if (impl_->context) ucp_cleanup(impl_->context);
}

namespace {
void UcxWait(ucp_worker_h w, void* req) {
  if (req == nullptr || UCS_PTR_IS_ERR(req)) return;
  while (ucp_request_check_status(req) == UCS_INPROGRESS) ucp_worker_progress(w);
  ucp_request_free(req);
}
}  // namespace

Status UcxTransport::Send(int to, const RecordBatch& batch) {
  std::vector<uint8_t> bytes = SerializeBatch(batch);
  uint64_t len = bytes.size();
  ucp_request_param_t p{};
  // GPUDirect RDMA: device resident buffers are registered by UCX when the memory type is CUDA.
  UcxWait(impl_->worker, ucp_tag_send_nbx(impl_->endpoints[to], &len, sizeof len, 1, &p));
  UcxWait(impl_->worker, ucp_tag_send_nbx(impl_->endpoints[to], bytes.data(), len, 2, &p));
  return Status::OK();
}

Result<RecordBatchPtr> UcxTransport::Recv(int) {
  uint64_t len = 0;
  ucp_request_param_t p{};
  UcxWait(impl_->worker, ucp_tag_recv_nbx(impl_->worker, &len, sizeof len, 1, ~0ull, &p));
  std::vector<uint8_t> bytes(len);
  UcxWait(impl_->worker, ucp_tag_recv_nbx(impl_->worker, bytes.data(), len, 2, ~0ull, &p));
  return DeserializeBatch(bytes.data(), bytes.size());
}

Status UcxTransport::Barrier() {
  ucp_worker_flush(impl_->worker);
  return Status::OK();
}
#endif

ExchangeOperator::ExchangeOperator(std::shared_ptr<Transport> t, std::vector<int> keys)
    : transport_(std::move(t)), partitioner_(std::move(keys), static_cast<uint32_t>(transport_->world_size())) {}

ExchangeStrategy ExchangeOperator::Choose(uint64_t left_bytes, uint64_t right_bytes, double link_bps, bool nvswitch) {
  // On NVSwitch the shuffle runs at NVLink speed and moves less data; on PCIe broadcast the small side.
  if (nvswitch) return ExchangeStrategy::HashShuffle;
  uint64_t smaller = std::min(left_bytes, right_bytes);
  uint64_t shuffle_bytes = left_bytes + right_bytes;
  double t_shuffle = double(shuffle_bytes) / link_bps;
  double t_broadcast = double(smaller) * 7 / link_bps;
  return t_broadcast < t_shuffle ? ExchangeStrategy::BroadcastSmaller : ExchangeStrategy::HashShuffle;
}

Result<std::vector<RecordBatchPtr>> ExchangeOperator::Shuffle(const std::vector<RecordBatchPtr>& local) {
  int world = transport_->world_size(), me = transport_->rank();
  std::vector<std::vector<RecordBatchPtr>> outgoing(world);
  for (const auto& b : local) {
    ASTER_ASSIGN_OR_RETURN(auto parts, partitioner_.Split(*b));
    for (int r = 0; r < world; ++r) if (parts[r]->num_rows()) outgoing[r].push_back(parts[r]);
  }
  std::vector<RecordBatchPtr> received;
  for (int r = 0; r < world; ++r) {
    RecordBatchPtr merged = ConcatBatches(outgoing[r]);
    if (r == me) { if (merged->num_rows()) received.push_back(merged); continue; }
    if (merged->columns.empty() && !local.empty()) merged->schema = local[0]->schema;
    bytes_sent_ += merged->nbytes();
    ASTER_RETURN_NOT_OK(transport_->Send(r, *merged));
  }
  for (int r = 0; r < world; ++r) {
    if (r == me) continue;
    ASTER_ASSIGN_OR_RETURN(auto b, transport_->Recv(r));
    if (b->num_rows()) received.push_back(b);
  }
  ASTER_RETURN_NOT_OK(transport_->Barrier());
  return received;
}

Result<std::vector<RecordBatchPtr>> ExchangeOperator::Broadcast(const std::vector<RecordBatchPtr>& local) {
  int world = transport_->world_size(), me = transport_->rank();
  RecordBatchPtr mine = ConcatBatches(local);
  std::vector<RecordBatchPtr> all;
  if (mine->num_rows()) all.push_back(mine);
  for (int r = 0; r < world; ++r) {
    if (r == me) continue;
    bytes_sent_ += mine->nbytes();
    ASTER_RETURN_NOT_OK(transport_->Send(r, *mine));
  }
  for (int r = 0; r < world; ++r) {
    if (r == me) continue;
    ASTER_ASSIGN_OR_RETURN(auto b, transport_->Recv(r));
    if (b->num_rows()) all.push_back(b);
  }
  ASTER_RETURN_NOT_OK(transport_->Barrier());
  return all;
}

}  // namespace aster::exchange
