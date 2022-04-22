#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dataset.hpp"


// #include "scann_wrapper.hpp"
// #include "scann/scann_ops/cc/scann.h"
// #include "scann/utils/types.h"
// #include "scann/utils/common.h"

namespace research_scann {


using DatapointIndex = uint32_t;

class ScannInterface;


}


namespace scann {

using research_scann::DatapointIndex;

class ScannSearcher {
 public:
  ScannSearcher();
  ScannSearcher(ConstDataSetWrapper<float, 2> dataset, const std::string& config, int training_threads);
  ~ScannSearcher();
//   std::pair<std::vector<DatapointIndex>, std::vector<float>> Search(ConstDataSetWrapper<float, 1> query, int final_num_neighbors = -1,
//          int pre_reorder_num_neighbors = -1,
//          int leaves_to_search = -1);

  std::pair<std::vector<DatapointIndex>, std::vector<float>> SearchBatched(ConstDataSetWrapper<float, 2> queries, int final_nn = -1,
         int pre_reorder_nn = -1,
         int leaves_to_search = -1, bool parallel = false);

//   ... SearchBatchedParallel(NpArray queries, int final_num_neighbors = -1,
//          int pre_reorder_num_neighbors = -1,
//          int leaves_to_search = -1);

  // Writer must iplement .write(const char * from, size_t n);
  template<typename Writer>
  void Serialize(Writer & writer) const;

  // Writer must iplement .write(const char * from, size_t n);
  template<typename Reader>
  void Deserialize(Reader & reader);

  bool IsInitialized() const;
 private:
  // std::unique_ptr<research_scann::ScannInterface> scann_;
  research_scann::ScannInterface* scann_;
};

// ScannSearcher LoadSearcher() {
//   // Not implemented
// }

}  // namespace research_scann
