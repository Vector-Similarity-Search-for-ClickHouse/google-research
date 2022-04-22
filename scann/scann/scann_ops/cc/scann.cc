// Copyright 2022 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "scann/scann_ops/cc/scann.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/internal/sysinfo.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "google/protobuf/arena.h"
#include "partitioner.pb.h"
#include "brute_force.pb.h"
#include "centers.pb.h"
#include "scann.pb.h"
#include "scann/tree_x_hybrid/tree_x_params.h"
#include "scann/utils/common.h"
#include "scann/utils/io_npy.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/scann_config_utils.h"
#include "scann/utils/threads.h"
#include "tensorflow/core/platform/status.h"

namespace research_scann {
namespace {

int GetNumCPUs() { return std::max(absl::base_internal::NumCPUs(), 1); }

template <typename T>
Status ParseTextProto(T* proto, const string& proto_str) {
  ::google::protobuf::TextFormat::ParseFromString(proto_str, proto);
  return OkStatus();
}

unique_ptr<DenseDataset<float>> InitDataset(ConstSpan<float> dataset,
                                            DatapointIndex n_points) {
  if (dataset.empty()) return nullptr;

  vector<float> dataset_vec(dataset.data(), dataset.data() + dataset.size());
  return absl::make_unique<DenseDataset<float>>(dataset_vec, n_points);
}

}  // namespace

Status ScannInterface::Initialize(
    ConstSpan<float> dataset, ConstSpan<int32_t> datapoint_to_token,
    ConstSpan<uint8_t> hashed_dataset, ConstSpan<int8_t> int8_dataset,
    ConstSpan<float> int8_multipliers, ConstSpan<float> dp_norms,
    DatapointIndex n_points, const std::string& artifacts_dir) {
  ScannConfig config;
  SCANN_RETURN_IF_ERROR(
      ReadProtobufFromFile(artifacts_dir + "/scann_config.pb", &config));
  SingleMachineFactoryOptions opts;
  if (!hashed_dataset.empty()) {
    opts.ah_codebook = std::make_shared<CentersForAllSubspaces>();
    SCANN_RETURN_IF_ERROR(ReadProtobufFromFile(
        artifacts_dir + "/ah_codebook.pb", opts.ah_codebook.get()));
  }
  if (!datapoint_to_token.empty()) {
    opts.serialized_partitioner = std::make_shared<SerializedPartitioner>();
    SCANN_RETURN_IF_ERROR(
        ReadProtobufFromFile(artifacts_dir + "/serialized_partitioner.pb",
                             opts.serialized_partitioner.get()));
  }
  return Initialize(config, opts, dataset, datapoint_to_token, hashed_dataset,
                    int8_dataset, int8_multipliers, dp_norms, n_points);
}

Status ScannInterface::Initialize(
    ScannConfig config, SingleMachineFactoryOptions opts,
    ConstSpan<float> dataset, ConstSpan<int32_t> datapoint_to_token,
    ConstSpan<uint8_t> hashed_dataset, ConstSpan<int8_t> int8_dataset,
    ConstSpan<float> int8_multipliers, ConstSpan<float> dp_norms,
    DatapointIndex n_points) {
  config_ = config;
  if (opts.ah_codebook != nullptr) {
    vector<uint8_t> hashed_db(hashed_dataset.data(),
                              hashed_dataset.data() + hashed_dataset.size());
    opts.hashed_dataset =
        std::make_shared<DenseDataset<uint8_t>>(hashed_db, n_points);
  }
  if (opts.serialized_partitioner != nullptr) {
    if (datapoint_to_token.size() != n_points)
      return InvalidArgumentError(
          absl::StrFormat("datapoint_to_token length=%d but expected %d",
                          datapoint_to_token.size(), n_points));
    opts.datapoints_by_token =
        std::make_shared<vector<std::vector<DatapointIndex>>>(
            opts.serialized_partitioner->n_tokens());
    for (auto [dp_idx, token] : Enumerate(datapoint_to_token))
      opts.datapoints_by_token->at(token).push_back(dp_idx);
  }
  if (!int8_dataset.empty()) {
    auto int8_data = std::make_shared<PreQuantizedFixedPoint>();
    vector<int8_t> int8_vec(int8_dataset.data(),
                            int8_dataset.data() + int8_dataset.size());
    int8_data->fixed_point_dataset =
        std::make_shared<DenseDataset<int8_t>>(int8_vec, n_points);

    int8_data->multiplier_by_dimension = make_shared<vector<float>>(
        int8_multipliers.begin(), int8_multipliers.end());

    int8_data->squared_l2_norm_by_datapoint =
        make_shared<vector<float>>(dp_norms.begin(), dp_norms.end());
    opts.pre_quantized_fixed_point = int8_data;
  }
  return Initialize(InitDataset(dataset, n_points), opts);
}

Status ScannInterface::Initialize(ConstSpan<float> dataset,
                                  DatapointIndex n_points,
                                  const std::string& config,
                                  int training_threads) {
  SCANN_RETURN_IF_ERROR(ParseTextProto(&config_, config));
  if (training_threads < 0)
    return InvalidArgumentError("training_threads must be non-negative");
  if (training_threads == 0) training_threads = GetNumCPUs();
  SingleMachineFactoryOptions opts;

  opts.parallelization_pool =
      StartThreadPool("scann_threadpool", training_threads - 1);
  return Initialize(InitDataset(dataset, n_points), opts);
}

Status ScannInterface::Initialize(shared_ptr<DenseDataset<float>> dataset,
                                  SingleMachineFactoryOptions opts) {
  TF_ASSIGN_OR_RETURN(dimensionality_, opts.ComputeConsistentDimensionality(
                                           config_.hash(), dataset.get()));
  TF_ASSIGN_OR_RETURN(n_points_, opts.ComputeConsistentSize(dataset.get()));

  if (dataset && config_.has_partitioning() &&
      config_.partitioning().partitioning_type() ==
          PartitioningConfig::SPHERICAL)
    dataset->set_normalization_tag(research_scann::UNITL2NORM);
  TF_ASSIGN_OR_RETURN(scann_, SingleMachineFactoryScann<float>(
                                  config_, dataset, std::move(opts)));

  const std::string& distance = config_.distance_measure().distance_measure();
  const absl::flat_hash_set<std::string> negated_distances{
      "DotProductDistance", "BinaryDotProductDistance", "AbsDotProductDistance",
      "LimitedInnerProductDistance"};
  result_multiplier_ =
      negated_distances.find(distance) == negated_distances.end() ? 1 : -1;

  if (config_.has_partitioning()) {
    min_batch_size_ = 1;
  } else {
    if (config_.has_hash())
      min_batch_size_ = 16;
    else
      min_batch_size_ = 256;
  }
  return OkStatus();
}

SearchParameters ScannInterface::GetSearchParameters(int final_nn,
                                                     int pre_reorder_nn,
                                                     int leaves) const {
  SearchParameters params;
  bool has_reordering = config_.has_exact_reordering();
  int post_reorder_nn = -1;
  if (has_reordering) {
    post_reorder_nn = final_nn;
  } else {
    pre_reorder_nn = final_nn;
  }
  params.set_pre_reordering_num_neighbors(pre_reorder_nn);
  params.set_post_reordering_num_neighbors(post_reorder_nn);
  if (leaves > 0) {
    auto tree_params = std::make_shared<TreeXOptionalParameters>();
    tree_params->set_num_partitions_to_search_override(leaves);
    params.set_searcher_specific_optional_parameters(tree_params);
  }
  return params;
}

vector<SearchParameters> ScannInterface::GetSearchParametersBatched(
    int batch_size, int final_nn, int pre_reorder_nn, int leaves,
    bool set_unspecified) const {
  vector<SearchParameters> params(batch_size);
  bool has_reordering = config_.has_exact_reordering();
  int post_reorder_nn = -1;
  if (has_reordering) {
    post_reorder_nn = final_nn;
  } else {
    pre_reorder_nn = final_nn;
  }
  std::shared_ptr<research_scann::TreeXOptionalParameters> tree_params;
  if (leaves > 0) {
    tree_params = std::make_shared<TreeXOptionalParameters>();
    tree_params->set_num_partitions_to_search_override(leaves);
  }

  for (auto& p : params) {
    p.set_pre_reordering_num_neighbors(pre_reorder_nn);
    p.set_post_reordering_num_neighbors(post_reorder_nn);
    if (tree_params) p.set_searcher_specific_optional_parameters(tree_params);
    if (set_unspecified) scann_->SetUnspecifiedParametersToDefaults(&p);
  }
  return params;
}

Status ScannInterface::Search(const DatapointPtr<float> query,
                              NNResultsVector* res, int final_nn,
                              int pre_reorder_nn, int leaves) const {
  if (query.dimensionality() != dimensionality_)
    return InvalidArgumentError("Query doesn't match dataset dimsensionality");
  SearchParameters params =
      GetSearchParameters(final_nn, pre_reorder_nn, leaves);
  scann_->SetUnspecifiedParametersToDefaults(&params);
  return scann_->FindNeighbors(query, params, res);
}

Status ScannInterface::SearchBatched(const DenseDataset<float>& queries,
                                     MutableSpan<NNResultsVector> res,
                                     int final_nn, int pre_reorder_nn,
                                     int leaves) const {
  if (queries.dimensionality() != dimensionality_)
    return InvalidArgumentError("Query doesn't match dataset dimsensionality");
  if (!std::isinf(scann_->default_pre_reordering_epsilon()) ||
      !std::isinf(scann_->default_post_reordering_epsilon()))
    return InvalidArgumentError("Batch querying isn't supported with epsilon");
  auto params = GetSearchParametersBatched(queries.size(), final_nn,
                                           pre_reorder_nn, leaves, true);
  return scann_->FindNeighborsBatched(queries, params, MakeMutableSpan(res));
}

Status ScannInterface::SearchBatchedParallel(const DenseDataset<float>& queries,
                                             MutableSpan<NNResultsVector> res,
                                             int final_nn, int pre_reorder_nn,
                                             int leaves) const {
  const size_t numQueries = queries.size();
  const size_t numCPUs = GetNumCPUs();

  const size_t kBatchSize = std::min(
      std::max(min_batch_size_, DivRoundUp(numQueries, numCPUs)), 256ul);
  auto pool = StartThreadPool("pool", numCPUs - 1);
  return ParallelForWithStatus<1>(
      Seq(DivRoundUp(numQueries, kBatchSize)), pool.get(), [&](size_t i) {
        size_t begin = kBatchSize * i;
        size_t curSize = std::min(numQueries - begin, kBatchSize);
        vector<float> queryCopy(
            queries.data().begin() + begin * dimensionality_,
            queries.data().begin() + (begin + curSize) * dimensionality_);
        DenseDataset<float> curQueryDataset(queryCopy, curSize);
        return SearchBatched(curQueryDataset, res.subspan(begin, curSize),
                             final_nn, pre_reorder_nn, leaves);
      });
}

Status ScannInterface::Serialize(std::string path) {
  TF_ASSIGN_OR_RETURN(auto opts, scann_->ExtractSingleMachineFactoryOptions());

  SCANN_RETURN_IF_ERROR(
      WriteProtobufToFile(path + "/scann_config.pb", &config_));
  if (opts.ah_codebook != nullptr)
    SCANN_RETURN_IF_ERROR(
        WriteProtobufToFile(path + "/ah_codebook.pb", opts.ah_codebook.get()));
  if (opts.serialized_partitioner != nullptr)
    SCANN_RETURN_IF_ERROR(
        WriteProtobufToFile(path + "/serialized_partitioner.pb",
                            opts.serialized_partitioner.get()));
  if (opts.datapoints_by_token != nullptr) {
    vector<int32_t> datapoint_to_token(n_points_);
    for (const auto& [token_idx, dps] : Enumerate(*opts.datapoints_by_token))
      for (auto dp_idx : dps) datapoint_to_token[dp_idx] = token_idx;
    SCANN_RETURN_IF_ERROR(
        VectorToNumpy(path + "/datapoint_to_token.npy", datapoint_to_token));
  }
  if (opts.hashed_dataset != nullptr) {
    SCANN_RETURN_IF_ERROR(
        DatasetToNumpy(path + "/hashed_dataset.npy", *(opts.hashed_dataset)));
  }
  if (opts.pre_quantized_fixed_point != nullptr) {
    auto fixed_point = opts.pre_quantized_fixed_point;
    auto dataset = fixed_point->fixed_point_dataset;
    if (dataset != nullptr) {
      SCANN_RETURN_IF_ERROR(
          DatasetToNumpy(path + "/int8_dataset.npy", *dataset));
    }
    auto multipliers = fixed_point->multiplier_by_dimension;
    if (multipliers != nullptr) {
      SCANN_RETURN_IF_ERROR(
          VectorToNumpy(path + "/int8_multipliers.npy", *multipliers));
    }
    auto norms = fixed_point->squared_l2_norm_by_datapoint;
    if (norms != nullptr) {
      SCANN_RETURN_IF_ERROR(VectorToNumpy(path + "/dp_norms.npy", *norms));
    }
  }
  TF_ASSIGN_OR_RETURN(auto dataset, Float32DatasetIfNeeded());
  if (dataset != nullptr)
    SCANN_RETURN_IF_ERROR(DatasetToNumpy(path + "/dataset.npy", *dataset));
  return OkStatus();
}

namespace writer_detail {
  template <typename Writer> 
  void WriteProtobufToWriter(Writer& writer, google::protobuf::Message *message) {
    std::string serialized_message;
    if(!message->SerializeToString(&serialized_message)) {
      // return InternalError("Failed to write proto to str");
    }
    size_t size = serialized_message.size();
    writer.write(reinterpret_cast<char*>(&size), sizeof(size_t));
    writer.write(serialized_message.data(), size);
  }

  template <typename Writer, typename T>
  void SpanToWriter(Writer& writer, ConstSpan<T> span) {
    size_t size = span.size();
    writer.write(reinterpret_cast<char*>(&size), sizeof(size_t));
    writer.write(reinterpret_cast<char*>(span.data()), sizeof(T) * size);
  }

  template <typename Writer, typename T> 
  void VectorToWriter(Writer& writer, const vector<T>& data, 
                        const vector<size_t>& dim_size = {}) {
    SpanToWriter(writer, ConstSpan<T>(data.data(), data.size()));
    SpanToWriter(writer, ConstSpan<T>(dim_size.data(), dim_size.size()));  
  } 

  template <typename Writer, typename T> 
  void DatasetToWriter(Writer& writer, const vector<T>& data) {
    SpanToWriter(writer, ConstSpan<T>(data.data(), data.size()));
  }

  template <typename Writer>
  void WriteZeroSize(Writer & writer) {
    size_t size = 0;
    writer.write(reinterpret_cast<char*>(&size), sizeof(size_t));
  }
}  // namespace writer_detail

template <typename Writer> 
Status ScannInterface::Serialize(Writer& writer) {
  TF_ASSIGN_OR_RETURN(auto opts, scann_->ExtractSingleMachineFactoryOptions());

  writer_detail::WriteProtobufToWriter(writer, &config_);
  if (opts.ah_codebook != nullptr) 
    writer_detail::WriteProtobufToWriter(writer, opts.ah_codebook.get());
  else 
    writer_detail::WriteZeroSize(writer);

  if (opts.serialized_partitioner != nullptr)
    writer_detail::WriteProtobufToWriter(writer, opts.serialized_partitioner.get());
  else 
    writer_detail::WriteZeroSize(writer);


  if (opts.datapoints_by_token != nullptr) {
    vector<int32_t> datapoint_to_token(n_points_);
    for (const auto& [token_idx, dps] : Enumerate(*opts.datapoints_by_token))
      for (auto dp_idx : dps) datapoint_to_token[dp_idx] = token_idx;
    writer_detail::VectorToWriter(writer, datapoint_to_token);
  } else 
    writer_detail::WriteZeroSize(writer);

  if (opts.hashed_dataset != nullptr)
    writer_detail::DatasetToWriter(writer, *(opts.hashed_dataset));
  else 
    writer_detail::WriteZeroSize(writer);

  if (opts.pre_quantized_fixed_point != nullptr) {
    auto fixed_point = opts.pre_quantized_fixed_point;
    auto dataset = fixed_point->fixed_point_dataset;
    if (dataset != nullptr) {
      writer_detail::DatasetToWriter(writer, *dataset);
    } else {
      writer_detail::WriteZeroSize(writer);
    }
    auto multipliers = fixed_point->multiplier_by_dimension;
    if (multipliers != nullptr) {
      writer_detail::VectorToWriter(writer, *multipliers);
    } else {
      writer_detail::WriteZeroSize(writer);
    }
    auto norms = fixed_point->squared_l2_norm_by_datapoint;
    if (norms != nullptr) {
      writer_detail::VectorToWriter(writer, *norms);
    } else {
      writer_detail::WriteZeroSize(writer);
    }
  } else {
    writer_detail::WriteZeroSize(writer);
    writer_detail::WriteZeroSize(writer);
    writer_detail::WriteZeroSize(writer);
  }

  TF_ASSIGN_OR_RETURN(auto dataset, Float32DatasetIfNeeded());
  if (dataset != nullptr)
    writer_detail::DatasetToWriter(writer, *dataset);
  else 
    writer_detail::WriteZeroSize(writer);

  return OkStatus();  
}

namespace reader_detail {
  template <typename Reader> 
  bool ReadProtobufFromReader(Reader& reader, google::protobuf::Message *message) {
    size_t size;
    reader.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    if (size != 0) {
      std::string data(size, '\0');
      message->ParseFromString(data);
    }
    
    return size != 0;
  }

  template <typename Reader, typename T>
  bool SpanFromReader(Reader& reader, std::vector<T>& data) {
    size_t size;
    reader.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    // data.clear();
    data.reserve(size);
    reader.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
    return size != 0;
  }

  template <typename Reader, typename T>
  bool VecFromReader(Reader& reader, std::vector<T>& data, std::vector<size_t>& shape) {
    return SpanFromReader(reader, data) && SpanFromReader(reader, shape);    
  }

  template <typename Reader, typename T>
  bool DatasetFromReader(Reader& reader, std::vector<T>& data) {
    return SpanFromReader(reader, data);
  }

}  // namespace reader_detail

namespace detail {
  template <typename T>
  ConstSpan<T> VecToSpan(std::vector<T>& vec) {
    return {vec.data(), vec.size()};
  }
} // namespace detail

template <typename Reader>
Status ScannInterface::Deserialize(Reader& reader) {
  ScannConfig config;
  reader_detail::ReadProtobufFromReader(reader, &config);

  SingleMachineFactoryOptions opts;
  auto ah_codebook = std::make_shared<CentersForAllSubspaces>();
  if (reader_detail::ReadProtobufFromReader(reader, ah_codebook.get())) {
    opts.ah_codebook = std::move(ah_codebook);
  }
  auto serialized_partitioner = std::make_shared<SerializedPartitioner>();
  if (reader_detail::ReadProtobufFromReader(reader, serialized_partitioner.get())) {
    opts.serialized_partitioner = std::move(serialized_partitioner);
  }

  DatapointIndex n_points = kInvalidDatapointIndex;

  std::vector<int32_t> datapoint_to_token;
  std::vector<size_t> datapoint_to_token_shape;

  if (reader_detail::VecFromReader(reader, datapoint_to_token, datapoint_to_token_shape)) {
    n_points = datapoint_to_token_shape[0];
  }

  std::vector<uint8_t> hashed_dataset;
  if (reader_detail::DatasetFromReader(reader, hashed_dataset)) {
    n_points = hashed_dataset.size();
  }

  std::vector<int8_t> int8_dataset;
  if (reader_detail::DatasetFromReader(reader, int8_dataset)) {
    n_points = int8_dataset.size();
  }

  std::vector<float> multiplier;
  std::vector<size_t> multiplier_shape;
  if (reader_detail::VecFromReader(reader, multiplier, multiplier_shape)) {
    n_points = multiplier_shape[0];
  }

  std::vector<float> dp_norm;
  std::vector<size_t> dp_norm_shape;
  if (reader_detail::VecFromReader(reader, dp_norm, dp_norm_shape)) {
    n_points = dp_norm_shape[0];
  }

  std::vector<float> dataset;
  if (reader_detail::DatasetFromReader(reader, dataset)) {
    n_points = dataset.size();
  }

  return Initialize(config,
             opts,
             detail::VecToSpan(dataset),
             detail::VecToSpan(datapoint_to_token),
             detail::VecToSpan(hashed_dataset),
             detail::VecToSpan(int8_dataset),
             detail::VecToSpan(multiplier),
             detail::VecToSpan(dp_norm),
             n_points);
}

StatusOr<SingleMachineFactoryOptions> ScannInterface::ExtractOptions() {
  return scann_->ExtractSingleMachineFactoryOptions();
}

}  // namespace research_scann
