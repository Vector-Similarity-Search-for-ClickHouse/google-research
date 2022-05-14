#include "scann_searcher.hpp"

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace scann {

using research_scann::ConstSpan;
using research_scann::DatapointPtr;
using research_scann::DenseDataset;
using research_scann::MakeConstSpan;
using research_scann::MakeMutableSpan;
using research_scann::NNResultsVector;
using research_scann::Status;

// void RuntimeErrorIfNotOk(const char* prefix, const Status& status) {
//   if (!status.ok()) {
//     std::string msg = prefix + std::string(status.error_message());
//     throw std::runtime_error(msg);
//   }
// }

ScannSearcher::ScannSearcher(ConstDataSetWrapper<float, 2> dataset,
                             const std::string& config, int training_threads) {
  ConstSpan<float> span_dataset(dataset.Data(), dataset.Size());

  "ScannInterface initialization failed",
      scann_.Initialize(span_dataset, dataset.Shape()[0], config,
                        training_threads);
}

std::pair<std::vector<DatapointIndex>, std::vector<float>>&
ScannSearcher::SearchBatched(ConstDataSetWrapper<float, 2> queries,
                             int final_nn, int pre_reorder_nn,
                             int leaves_to_search, bool parallel) {
  std::vector<float> queries_vec(queries.Data(),
                                 queries.Data() + queries.Size());
  auto query_dataset = DenseDataset<float>(queries_vec, queries.Shape()[0]);

  std::vector<NNResultsVector> res(query_dataset.size());
  // Status status;
  if (parallel)
    scann_.SearchBatchedParallel(query_dataset, MakeMutableSpan(res), final_nn,
                                 pre_reorder_nn, leaves_to_search);
  else
    scann_.SearchBatched(query_dataset, MakeMutableSpan(res), final_nn,
                         pre_reorder_nn, leaves_to_search);

  for (const auto& nn_res : res) {
    final_nn = std::max<int>(final_nn, nn_res.size());
  }

  std::vector<DatapointIndex> indices(query_dataset.size() * final_nn);
  std::vector<float> distances(query_dataset.size() * final_nn);

  scann_.ReshapeBatchedNNResult(MakeConstSpan(res), indices.data(),
                                distances.data(), final_nn);
  search_result_ = {indices, distances};
  return search_result_;
}

void ScannSearcher::Serialize(IWriter& writer) {
  "ScaNNSearcher serialization Failed", scann_.Serialize(writer);
}

void ScannSearcher::Deserialize(IReader& reader) {
  "ScaNNSearcher deserialization Failed", scann_.Deserialize(reader);
}

}  // namespace scann

extern "C" {

scann::ScannSearcher* ScannSearcherCreate(scann::ConstDataSetWrapper<float, 2>* dataset,
                                          std::string* config,
                                          int training_threads) {
  return new scann::ScannSearcher(*dataset, *config, training_threads);
}

void ScannSearcherDestroy(scann::ScannSearcher* self) { delete self; }

std::pair<std::vector<research_scann::DatapointIndex>, std::vector<float>>*
ScannSearcherSearchBatched(scann::ScannSearcher* self,
                           scann::ConstDataSetWrapper<float, 2> queries,
                           int final_nn, int pre_reorder_nn,
                           int leaves_to_search, bool parallel) {
  return &self->SearchBatched(queries, final_nn, pre_reorder_nn,
                              leaves_to_search, parallel);
}

void ScannSearcherSerialize(scann::ScannSearcher* self,
                            scann::IWriter& writer) {
  self->Serialize(writer);
}
void ScannSearcherDeserialize(scann::ScannSearcher* self,
                              scann::IReader& reader) {
  self->Deserialize(reader);
}
};
