set(SCANN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SCANN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

cmake_minimum_required(VERSION 3.10)
project(ProjectName)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-mssse3 -msse4)

# abslei 
add_subdirectory(abseil-cpp)

# gtest 
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# protobuf
find_package(Protobuf)

set(Protobuf_IMPORT_DIRS ${SCANN_SOURCE_DIR})

# eigen3
set(Eigen3_DIR eigen-3.4.0/cmake)
find_package(Eigen3 REQUIRED)

# TensorFlow
# list(APPEND CMAKE_MODULE_PATH "${SCANN_SOURCE_DIR}/modules")
# find_package(TensorFlow REQUIRED)
set(TensorFlow_INCLUDE_DIRS ${SCANN_SOURCE_DIR}/tensorflow/tensorflow)
set(TensorFlow_LIBRARIES ${SCANN_SOURCE_DIR}/tensorflow/old/libtensorflow_cc.so)


set(INCLUDE_DIRS 
${SCANN_SOURCE_DIR}
${SCANN_BINARY_DIR}
${Protobuf_INCLUDE_DIRS}
${gtest_SOURCE_DIR}/include
${gtest_SOURCE_DIR}
)

include_directories(${INCLUDE_DIRS})

set(PROTO_SOURCES
${SCANN_SOURCE_DIR}/scann/data_format/features.proto
${SCANN_SOURCE_DIR}/scann/partitioning/linear_projection_tree.proto
${SCANN_SOURCE_DIR}/scann/partitioning/partitioner.proto
${SCANN_SOURCE_DIR}/scann/partitioning/kmeans_tree_partitioner.proto

${SCANN_SOURCE_DIR}/scann/proto/scann.proto
${SCANN_SOURCE_DIR}/scann/proto/exact_reordering.proto
${SCANN_SOURCE_DIR}/scann/proto/brute_force.proto
${SCANN_SOURCE_DIR}/scann/proto/disjoint_restrict_token.proto
${SCANN_SOURCE_DIR}/scann/proto/results.proto
${SCANN_SOURCE_DIR}/scann/proto/crowding.proto
${SCANN_SOURCE_DIR}/scann/proto/distance_measure.proto
${SCANN_SOURCE_DIR}/scann/proto/projection.proto
${SCANN_SOURCE_DIR}/scann/proto/metadata.proto
${SCANN_SOURCE_DIR}/scann/proto/hash.proto
${SCANN_SOURCE_DIR}/scann/proto/input_output.proto
${SCANN_SOURCE_DIR}/scann/proto/restricts.proto
${SCANN_SOURCE_DIR}/scann/proto/incremental_updates.proto
${SCANN_SOURCE_DIR}/scann/proto/hashed.proto
${SCANN_SOURCE_DIR}/scann/proto/partitioning.proto
${SCANN_SOURCE_DIR}/scann/proto/centers.proto

${SCANN_SOURCE_DIR}/scann/coscann/v2_restricts.proto
${SCANN_SOURCE_DIR}/scann/trees/kmeans_tree/kmeans_tree.proto
)


protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_SOURCES})
add_library(cc_proto ${PROTO_SRCS} ${PROTO_HDRS})

target_include_directories(cc_proto PUBLIC
${SCANN_BINARY_DIR}
${SCANN_BINARY_DIR}/scann/proto
${SCANN_BINARY_DIR}/scann/coscann
${SCANN_BINARY_DIR}/scann/trees/kmeans_tree
${SCANN_BINARY_DIR}/scann/partitioning
${SCANN_BINARY_DIR}/scann/data_format
${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(cc_proto PUBLIC ${Protobuf_LIBRARIES})

add_subdirectory(scann_api/scann)
add_subdirectory(scann)

set(SCANN_INCLUDE_DIRS ${SCANN_SOURCE_DIR}/scann_api)

add_executable(Main main.cpp)
target_link_libraries(Main scann_interface)
target_include_directories(Main PUBLIC ${SCANN_INCLUDE_DIRS})