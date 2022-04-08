add_library(scann_interface SHARED
scann_builder.hpp
scann_builder.cpp
scann_ops_pybind.hpp
scann_ops_pybind.cpp
dataset.hpp
)

target_link_libraries(scann_interface PRIVATE scann_ops_cc_scann)
target_include_directories(scann_interface PRIVATE ${SCANN_COMMON_INCLUDE_DIRS})