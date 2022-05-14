#pragma once

#include <string>
#include <vector>

#include "dataset.hpp"
#include "io.hpp"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace scann {

using research_scann::DatapointIndex;

class ScannSearcher {
 public:
  ScannSearcher(ConstDataSetWrapper<float, 2> dataset,
                const std::string& config, int training_threads);

  std::pair<std::vector<DatapointIndex>, std::vector<float>>& SearchBatched(
      ConstDataSetWrapper<float, 2> queries, int final_nn = -1,
      int pre_reorder_nn = -1, int leaves_to_search = -1,
      bool parallel = false);

  void Serialize(IWriter& writer);
  void Deserialize(IReader& reader);

 private:
  research_scann::ScannInterface scann_{};
  std::pair<std::vector<DatapointIndex>, std::vector<float>> search_result_{};
};

}  // namespace scann

extern "C" {

scann::ScannSearcher* ScannSearcherCreate(scann::ConstDataSetWrapper<float, 2>* dataset,
                                          std::string* config,
                                          int training_threads);
void ScannSearcherDestroy(scann::ScannSearcher* self);

std::pair<std::vector<research_scann::DatapointIndex>, std::vector<float>>*
ScannSearcherSearchBatched(scann::ScannSearcher* self,
                           scann::ConstDataSetWrapper<float, 2> queries,
                           int final_nn = -1, int pre_reorder_nn = -1,
                           int leaves_to_search = -1, bool parallel = false);

void ScannSearcherSerialize(scann::ScannSearcher* self, scann::IWriter& writer);
void ScannSearcherDeserialize(scann::ScannSearcher* self,
                              scann::IReader& reader);
};
