#include "scann/scann_builder.hpp"

int main() {

  std::vector<float> data = {1, 2, 4, 5};

  const int dimension = 3;
  const int sample_size = data.size() / dimension;

  scann::ConstDataSetWrapper<float, 2> data_set(data,
                                                  {sample_size, dimension});

  const size_t num_neighbors = 10;
  const std::string distance_measure = "dot_product";

  const size_t num_leaves = 2000;
  const size_t num_leaves_to_search = 100;
  const size_t training_sample_size = sample_size;

  const size_t dimension_per_block = 2;
  const float anisotropic_quantization_threshold = 0.2;

  const size_t reordering_num_neighbors = 100;

  // searcher =
  auto searcher = std::move(
      scann::ScannBuilder(data_set, num_neighbors, distance_measure)
          .Tree(num_leaves, num_leaves_to_search, training_sample_size)
          .ScoreAh(dimension_per_block, anisotropic_quantization_threshold)
          .Reorder(reordering_num_neighbors)
          .Build());

  
}