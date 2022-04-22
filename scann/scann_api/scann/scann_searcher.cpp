#include "scann_searcher.hpp"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace scann
{

using research_scann::ConstSpan;
using research_scann::DatapointPtr;
using research_scann::DenseDataset;
using research_scann::MakeConstSpan;
using research_scann::MakeMutableSpan;
using research_scann::NNResultsVector;
using research_scann::Status;

ScannSearcher::ScannSearcher() : scann_(new research_scann::ScannInterface())
{
}

ScannSearcher::ScannSearcher(ConstDataSetWrapper<float, 2> dataset, const std::string & config, int training_threads) : ScannSearcher()
{
    ConstSpan<float> span_dataset(dataset.Data(), dataset.Size());

    scann_->Initialize(span_dataset, dataset.Shape()[0], config, training_threads);
}

ScannSearcher::~ScannSearcher()
{
    delete scann_;
}

// std::pair<std::vector<DatapointIndex>, std::vector<float>>
// ScannSearcher::Search(ConstDataSetWrapper<float, 1> query,
//                       int final_num_neighbors,
//                       int pre_reorder_num_neighbors,
//                       int leaves_to_search) {
//   DatapointPtr<float> ptr(nullptr, query.Data(), query.Size(), query.Size());
//   NNResultsVector res;
//   auto status = scann_.Search(ptr, &res, final_num_neighbors,
//   pre_reorder_num_neighbors, leaves_to_search);
//   // Check status

//   // ???? return res;
//   std::vector<DatapointIndex> indices(res.size());
//   std::vector<float> distances(res.size());

// }

std::pair<std::vector<DatapointIndex>, std::vector<float>>
ScannSearcher::SearchBatched(ConstDataSetWrapper<float, 2> queries, int final_nn, int pre_reorder_nn, int leaves_to_search, bool parallel)
{
    std::vector<float> queries_vec(queries.Data(), queries.Data() + queries.Size());
    auto query_dataset = DenseDataset<float>(queries_vec, queries.Shape()[0]);

    std::vector<NNResultsVector> res(query_dataset.size());
    //Status status;
    if (parallel)
        scann_->SearchBatchedParallel(query_dataset, MakeMutableSpan(res), final_nn, pre_reorder_nn, leaves_to_search);
    else
        scann_->SearchBatched(query_dataset, MakeMutableSpan(res), final_nn, pre_reorder_nn, leaves_to_search);

    for (const auto & nn_res : res)
    {
        final_nn = std::max<int>(final_nn, nn_res.size());
    }

    std::vector<DatapointIndex> indices(query_dataset.size() * final_nn);
    std::vector<float> distances(query_dataset.size() * final_nn);


    scann_->ReshapeBatchedNNResult(MakeConstSpan(res), indices.data(), distances.data(), final_nn);
    return {indices, distances};
}

bool ScannSearcher::IsInitialized() const
{
    return scann_ != nullptr;
}

template <typename Writer>
void ScannSearcher::Serialize(Writer& writer) const
{
    scann_->Serialize(writer);
}

template <typename Reader>
void ScannSearcher::Deserialize(Reader& reader)
{
    scann_->Deserialize(reader);
}

} // namespace scann