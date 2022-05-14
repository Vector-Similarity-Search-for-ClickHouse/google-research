add_library(scann_interface SHARED
scann_builder.hpp
scann_builder.cpp
scann_searcher.hpp
scann_searcher.cpp
dataset.hpp
io.hpp
)

target_link_libraries(scann_interface PUBLIC scann_ops_cc_scann oss_wrappers_tf_dependency)
target_include_directories(scann_interface PUBLIC ${INCLUDE_DIRS})