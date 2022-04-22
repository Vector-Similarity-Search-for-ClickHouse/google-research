cmake_minimum_required(VERSION 3.10)
project(ProjectName)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-mssse3 -msse4)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules")

# abslei 
# taken from contrib

# gtest 
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# protobuf
# taken from contrib

set(Protobuf_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/contrib/protobuf/src)
set(Protobuf_IMPORT_DIRS 
${CMAKE_SOURCE_DIR}/contrib/protobuf/src
${CMAKE_CURRENT_SOURCE_DIR}
)

# eigen3
set(Eigen3_DIR eigen-3.4.0/cmake)
find_package(Eigen3 REQUIRED)

# TensorFlow
# search in root (/usr/local) 
find_package(TensorFlow REQUIRED)

set(SCANN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/scann_api CACHE PATH "")

set(INCLUDE_DIRS 
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_BINARY_DIR}
${Protobuf_INCLUDE_DIRS}
${gtest_SOURCE_DIR}/include
${gtest_SOURCE_DIR}
)

include_directories(${INCLUDE_DIRS})


set(PROTO_SOURCES
scann/data_format/features.proto
scann/partitioning/linear_projection_tree.proto
scann/partitioning/partitioner.proto
scann/partitioning/kmeans_tree_partitioner.proto

scann/proto/scann.proto
scann/proto/exact_reordering.proto
scann/proto/brute_force.proto
scann/proto/disjoint_restrict_token.proto
scann/proto/results.proto
scann/proto/crowding.proto
scann/proto/distance_measure.proto
scann/proto/projection.proto
scann/proto/metadata.proto
scann/proto/hash.proto
scann/proto/input_output.proto
scann/proto/restricts.proto
scann/proto/incremental_updates.proto
scann/proto/hashed.proto
scann/proto/partitioning.proto
scann/proto/centers.proto

scann/coscann/v2_restricts.proto
scann/trees/kmeans_tree/kmeans_tree.proto
)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_SOURCES})

add_library(cc_proto ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(cc_proto PUBLIC
${CMAKE_CURRENT_BINARY_DIR}
${CMAKE_CURRENT_BINARY_DIR}/scann/proto
${CMAKE_CURRENT_BINARY_DIR}/scann/coscann
${CMAKE_CURRENT_BINARY_DIR}/scann/trees/kmeans_tree
${CMAKE_CURRENT_BINARY_DIR}/scann/partitioning
${CMAKE_CURRENT_BINARY_DIR}/scann/data_format
${Protobuf_INCLUDE_DIRS}
)


add_subdirectory(scann_api/scann)
add_subdirectory(scann)

