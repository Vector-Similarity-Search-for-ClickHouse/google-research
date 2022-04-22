add_library(scann_interface STATIC
scann_builder.hpp
scann_builder.cpp
scann_searcher.hpp
scann_searcher.cpp
dataset.hpp
scann_wrapper.hpp
)

target_link_libraries(scann_interface PRIVATE scann_ops_cc_scann cxx)
target_include_directories(scann_interface PRIVATE ${INCLUDE_DIRS})