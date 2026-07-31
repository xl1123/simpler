# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.15)
project(simpler_mpi_l3_transport LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)

add_library(simpler_mpi_l3_transport SHARED mpi_l3_transport.cpp)
target_compile_features(simpler_mpi_l3_transport PRIVATE cxx_std_17)
target_link_libraries(simpler_mpi_l3_transport PRIVATE MPI::MPI_CXX)
target_compile_options(simpler_mpi_l3_transport PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(simpler_mpi_l3_transport PROPERTIES OUTPUT_NAME simpler_mpi_l3_transport)
