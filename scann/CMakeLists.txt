cmake_minimum_required(VERSION 3.10)

project(scann)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-mssse3 -msse4)

#gtest 
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# protobuf
INCLUDE(FindProtobuf)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules")

# abslei 
# set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(abseil-cpp)



# find packages
set(Eigen3_DIR /home/nikita/Downloads/eigen-3.4.0/cmake)
find_package(TensorFlow REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Protobuf REQUIRED)

message("${EIGEN3_INCLUDE_DIR}")
message("${PROTOBUF_INCLUDE_DIRS}")

set(Protobuf_IMPORT_DIRS ${PROJECT_SOURCE_DIR})


set(PROJECT_INCLUDE_DIRS 
${PROJECT_SOURCE_DIR} 
${PROJECT_BINARY_DIR} 
${EIGEN3_INCLUDE_DIR} 
${TensorFlow_INCLUDE_DIRS}  
${Protobuf_INCLUDE_DIRS}
"${gtest_SOURCE_DIR}/include"
"${gtest_SOURCE_DIR}"
)

include_directories(${PROJECT_INCLUDE_DIRS})

message(${PROJECT_BINARY_DIR})
message(${TensorFlow_LIBRARIES})


#-------------------------------------------------------------------------------

file(GLOB_RECURSE PROTO_SOURCES "*.proto")

#message("${PROTO_SOURCES}")

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_SOURCES})

add_library(cc_proto ${PROTO_SRCS} ${PROTO_HDRS})

add_library(proto_input_output_cc_proto INTERFACE)
target_link_libraries(proto_input_output_cc_proto INTERFACE cc_proto)

add_library(data_format_features_cc_proto INTERFACE)
target_link_libraries(data_format_features_cc_proto INTERFACE cc_proto)

add_library(proto_hashed_cc_proto INTERFACE)
target_link_libraries(proto_hashed_cc_proto INTERFACE cc_proto)

add_library(partitioning_partitioner_cc_proto INTERFACE)
target_link_libraries(partitioning_partitioner_cc_proto INTERFACE cc_proto)

add_library(proto_exact_reordering_cc_proto INTERFACE)
target_link_libraries(proto_exact_reordering_cc_proto INTERFACE cc_proto)

add_library(proto_results_cc_proto INTERFACE)
target_link_libraries(proto_results_cc_proto INTERFACE cc_proto)

add_library(proto_distance_measure_cc_proto INTERFACE)
target_link_libraries(proto_distance_measure_cc_proto INTERFACE cc_proto)

add_library(distance_measures INTERFACE)
target_link_libraries(distance_measures INTERFACE cc_proto)

add_library(proto_hash_cc_proto INTERFACE)
target_link_libraries(proto_hash_cc_proto INTERFACE cc_proto)

add_library(proto_projection_cc_proto INTERFACE)
target_link_libraries(proto_projection_cc_proto INTERFACE cc_proto)

add_library(proto_scann_cc_proto INTERFACE)
target_link_libraries(proto_scann_cc_proto INTERFACE cc_proto)

add_library(proto_brute_force_cc_proto INTERFACE)
target_link_libraries(proto_brute_force_cc_proto INTERFACE cc_proto)

add_library(proto_incremental_updates_cc_proto INTERFACE)
target_link_libraries(proto_incremental_updates_cc_proto INTERFACE cc_proto)

add_library(proto_metadata_cc_proto INTERFACE)
target_link_libraries(proto_metadata_cc_proto INTERFACE cc_proto)

add_library(proto_partitioning_cc_proto INTERFACE)
target_link_libraries(proto_partitioning_cc_proto INTERFACE cc_proto)

add_library(proto_restricts_cc_proto INTERFACE)
target_link_libraries(proto_restricts_cc_proto INTERFACE cc_proto)

add_library(proto_centers_cc_proto INTERFACE)
target_link_libraries(proto_centers_cc_proto INTERFACE cc_proto)

add_library(tree_kmeans_tree_kmeans_tree_cc_proto INTERFACE)
target_link_libraries(tree_kmeans_tree_kmeans_tree_cc_proto INTERFACE cc_proto)

add_library(partitioning_kmeans_tree_partitioner_cc_proto INTERFACE)
target_link_libraries(partitioning_kmeans_tree_partitioner_cc_proto INTERFACE cc_proto)

add_library(proto_crowding_cc_proto INTERFACE)
target_link_libraries(proto_crowding_cc_proto INTERFACE cc_proto)

#-------------------------------------------------------------------------------

add_subdirectory(scann)

add_executable(mian main.cpp)
target_link_libraries(mian PUBLIC scann_ops_cc_scann ${TensorFlow_LIBRARIES} ${PROTOBUF_LIBRARIES}) 