cmake_minimum_required(VERSION 3.18)
set(CMAKE_BUILD_TYPE Release)

project(HairMSNN 
  VERSION 1.0 
  LANGUAGES C CXX CUDA
)

set(OptiX_ROOT_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0/")

if (APPLE)
	set(CMAKE_MACOSX_RPATH ON)
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#################################
# CUDA COMPILER SETUP
#################################

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

get_directory_property(TCNN_HAS_PARENT PARENT_DIRECTORY)

#################################
# adapted from https://stackoverflow.com/a/69353718
#################################

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")

if (DEFINED ENV{TCNN_CUDA_ARCHITECTURES})
	message(STATUS "Obtained target architecture from environment variable TCNN_CUDA_ARCHITECTURES=$ENV{TCNN_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES $ENV{TCNN_CUDA_ARCHITECTURES})
elseif (TCNN_CUDA_ARCHITECTURES)
	message(STATUS "Obtained target architecture from CMake variable TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})
else()
	set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
endif()

# Remove unsupported architectures
list(FILTER CMAKE_CUDA_ARCHITECTURES EXCLUDE REGEX "PTX")
list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)

# If the CUDA version does not permit targeting Ampere, don't do so.
if ((80 IN_LIST CMAKE_CUDA_ARCHITECTURES OR 86 IN_LIST CMAKE_CUDA_ARCHITECTURES) AND CUDA_VERSION VERSION_LESS 11.0)
	message(WARNING "CUDA version ${CUDA_VERSION} is too low for targeting Ampere GPUs. Reverting to compute capability 75.")
	list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES 80 86)
	if (NOT CMAKE_CUDA_ARCHITECTURES)
		list(APPEND CMAKE_CUDA_ARCHITECTURES 75)
	endif()
endif()

# Sort the list to obtain lowest architecture that must be compiled for.
list(SORT CMAKE_CUDA_ARCHITECTURES COMPARE NATURAL ORDER ASCENDING)
list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)

string(REPLACE "-virtual" "" MIN_GPU_ARCH "${MIN_GPU_ARCH}")

message(STATUS "Targeting GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if (TCNN_HAS_PARENT)
	set(TCNN_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
	set(TCNN_CUDA_VERSION ${CUDA_VERSION} PARENT_SCOPE)
endif()

if (MIN_GPU_ARCH LESS_EQUAL 70)
	message(WARNING
		"Fully fused MLPs do not support GPU architectures of 70 or less. "
		"Falling back to CUTLASS MLPs. Remove GPU architectures 70 and lower "
		"to allow maximum performance"
	)
endif()

if (CUDA_VERSION VERSION_LESS 10.2)
	message(FATAL_ERROR "CUDA version too low. tiny-cuda-nn require CUDA 10.2 or higher.")
endif()

list(APPEND TCNN_DEFINITIONS -DTCNN_MIN_GPU_ARCH=${MIN_GPU_ARCH})
if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
	# Only compile the shampoo optimizer if
	# a new enough cuBLAS version is available.
	list(APPEND TCNN_DEFINITIONS -DTCNN_SHAMPOO)
endif()

if (TCNN_HAS_PARENT)
	set(TCNN_DEFINITIONS ${TCNN_DEFINITIONS} PARENT_SCOPE)
endif()

if (MSVC)
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcudafe=--diag_suppress=unrecognized_gcc_pragma")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

#################################
# TCNN DEPS
#################################
if (MSVC)
else()
	set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda/targets/x86_64-linux)
endif()
find_library(
	CUDA_CUBLASLT_LIBRARY cublasLt
	${CUDA_TOOLKIT_ROOT_DIR}/lib64
	${CUDA_TOOLKIT_ROOT_DIR}/lib
)

#################################
# TCNN SOURCE
#################################
set(BUILD_SHARED_LIBS OFF)
add_subdirectory("extern/tiny-cuda-nn/dependencies/fmt")

set(TCNN_SOURCES
  extern/tiny-cuda-nn/src/common.cu
	extern/tiny-cuda-nn/src/common_device.cu
	extern/tiny-cuda-nn/src/cpp_api.cu
	extern/tiny-cuda-nn/src/cutlass_mlp.cu
	extern/tiny-cuda-nn/src/encoding.cu
	extern/tiny-cuda-nn/src/loss.cu
	extern/tiny-cuda-nn/src/network.cu
	extern/tiny-cuda-nn/src/object.cu
	extern/tiny-cuda-nn/src/optimizer.cu
	extern/tiny-cuda-nn/src/reduce_sum.cu
)

if (MIN_GPU_ARCH GREATER 70)
	list(APPEND TCNN_SOURCES extern/tiny-cuda-nn/src/fully_fused_mlp.cu)
endif()

add_library(tiny-cuda-nn STATIC ${TCNN_SOURCES})
target_compile_definitions(tiny-cuda-nn PUBLIC ${TCNN_DEFINITIONS})
target_compile_options(tiny-cuda-nn PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_include_directories(tiny-cuda-nn PUBLIC
	"extern/tiny-cuda-nn/include/"
	"extern/tiny-cuda-nn/dependencies"
	"extern/tiny-cuda-nn/dependencies/cutlass/include"
	"extern/tiny-cuda-nn/dependencies/cutlass/tools/util/include"
	"extern/tiny-cuda-nn/dependencies/fmt/include"
)
target_link_libraries(tiny-cuda-nn
  PUBLIC ${CUDA_LIBRARIES} cuda cublas fmt
)

###############################################################################
# Everything else
###############################################################################

add_subdirectory(extern/owl/ EXCLUDE_FROM_ALL)
add_subdirectory(extern/tinyobjloader)
add_subdirectory(extern/tinyexr)

include_directories(${OptiX_INCLUDE})
include_directories(headers)
include_directories(cuda_headers)
include_directories(extern/cyCodeBase)
include_directories(extern/tinyobjloader)
include_directories(extern/tinyexr)
include_directories(extern/tinyexr/deps/miniz/)
include_directories(extern/imgui)
include_directories(extern/imgui/backends/)
include_directories(
  "extern/tiny-cuda-nn/include"
	"extern/tiny-cuda-nn/dependencies"
	"extern/tiny-cuda-nn/dependencies/cutlass/include"
	"extern/tiny-cuda-nn/dependencies/cutlass/tools/util/include"
	"extern/tiny-cuda-nn/dependencies/fmt/include"
)

#################################
# Path tracing
#################################

embed_ptx(
  OUTPUT_TARGET
    path_tracing_ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    cuda/path_tracing.cu
    cuda_headers/path_tracing.cuh

    # Header Files
    cuda_headers/common.cuh
    cuda_headers/optix_common.cuh
    cuda_headers/frostbite_anisotropic.cuh
    cuda_headers/random.cuh
    cuda_headers/utils.cuh
    cuda_headers/curve_utils.cuh
    cuda_headers/disney_hair.cuh
)

add_executable(render_path_tracing
  # Host code
  "render_path_tracing.cu" "scene.cpp" "model.cpp"

  # IMGUI files
  "extern/imgui/imgui.cpp" "extern/imgui/imgui_demo.cpp" "extern/imgui/imgui_draw.cpp" 
  "extern/imgui/imgui_tables.cpp"
  "extern/imgui/imgui_widgets.cpp"
  "extern/imgui/backends/imgui_impl_glfw.cpp" "extern/imgui/backends/imgui_impl_opengl2.cpp"

  # TINYEXR & DEPS
  "extern/tinyexr/deps/miniz/miniz.c"
)

target_compile_options(render_path_tracing PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_link_libraries(render_path_tracing
  # Compiled PTX
  PRIVATE path_tracing_ptx
  # OWL libs
  PUBLIC owl::owl owl_viewer
)

#################################
# nrc
#################################

embed_ptx(
  OUTPUT_TARGET
    nrc_ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    cuda/nrc.cu
    cuda_headers/nrc.cuh

    # Header Files
    cuda_headers/common.cuh
    cuda_headers/optix_common.cuh
    cuda_headers/frostbite_anisotropic.cuh
    cuda_headers/random.cuh
    cuda_headers/utils.cuh
    cuda_headers/curve_utils.cuh
    cuda_headers/disney_hair.cuh
)

add_executable(render_nrc
  # Host code
  "render_nrc.cu" "scene.cpp" "model.cpp" 
  "cuda/neural_network.cu"

  # IMGUI files
  "extern/imgui/imgui.cpp" "extern/imgui/imgui_demo.cpp" "extern/imgui/imgui_draw.cpp" 
  "extern/imgui/imgui_tables.cpp"
  "extern/imgui/imgui_widgets.cpp"
  "extern/imgui/backends/imgui_impl_glfw.cpp" "extern/imgui/backends/imgui_impl_opengl2.cpp"

  # TINYEXR & DEPS
  "extern/tinyexr/deps/miniz/miniz.c"
)

target_compile_definitions(render_nrc PUBLIC ${TCNN_DEFINITIONS})
target_compile_options(render_nrc PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_link_libraries(render_nrc
  # Compiled PTX
  PRIVATE nrc_ptx
  # OWL libs
  PUBLIC owl::owl owl_viewer
  # TCNN
  PUBLIC ${CUDA_LIBRARIES} tiny-cuda-nn
)

#################################
# hair_msnn
#################################

embed_ptx(
  OUTPUT_TARGET
    hair_msnn_ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    cuda/hair_msnn.cu
    cuda_headers/hair_msnn.cuh

    # Header Files
    cuda_headers/common.cuh
    cuda_headers/optix_common.cuh
    cuda_headers/frostbite_anisotropic.cuh
    cuda_headers/random.cuh
    cuda_headers/utils.cuh
    cuda_headers/curve_utils.cuh
    cuda_headers/disney_hair.cuh
)

add_executable(render_hair_msnn
  # Host code
  "render_hair_msnn.cu" "scene.cpp" "model.cpp" 
  "cuda/neural_network.cu"

  # IMGUI files
  "extern/imgui/imgui.cpp" "extern/imgui/imgui_demo.cpp" "extern/imgui/imgui_draw.cpp" 
  "extern/imgui/imgui_tables.cpp"
  "extern/imgui/imgui_widgets.cpp"
  "extern/imgui/backends/imgui_impl_glfw.cpp" "extern/imgui/backends/imgui_impl_opengl2.cpp"

  # TINYEXR & DEPS
  "extern/tinyexr/deps/miniz/miniz.c"
)

target_compile_definitions(render_hair_msnn PUBLIC ${TCNN_DEFINITIONS})
target_compile_options(render_hair_msnn PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_link_libraries(render_hair_msnn
  # Compiled PTX
  PRIVATE hair_msnn_ptx
  # OWL libs
  PUBLIC owl::owl owl_viewer
  # TCNN
  PUBLIC ${CUDA_LIBRARIES} tiny-cuda-nn
)