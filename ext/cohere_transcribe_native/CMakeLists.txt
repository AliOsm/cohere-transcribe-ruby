cmake_minimum_required(VERSION 3.15...3.31)

project(cohere_transcribe_native VERSION 0.8.9 LANGUAGES C CXX)

option(COHERE_TRANSCRIBE_CUDA "Build the optional ggml CUDA backend" OFF)
option(COHERE_TRANSCRIBE_METAL "Build the optional ggml Metal backend" OFF)
option(COHERE_TRANSCRIBE_NATIVE "Tune the CPU backend for the build host" OFF)
option(COHERE_TRANSCRIBE_OPENMP "Use OpenMP in ggml and Cohere feature extraction" OFF)

if(COHERE_TRANSCRIBE_METAL AND NOT APPLE)
    message(FATAL_ERROR "COHERE_TRANSCRIBE_METAL is only supported on Apple platforms")
endif()

if(COHERE_TRANSCRIBE_CUDA AND APPLE)
    message(FATAL_ERROR "The packaged CUDA build is supported on Linux only")
endif()

if(COHERE_TRANSCRIBE_CUDA AND COHERE_TRANSCRIBE_METAL)
    message(FATAL_ERROR "CUDA and Metal cannot be enabled in the same packaged runtime")
endif()

get_filename_component(COHERE_TRANSCRIBE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(CRISPASR_VENDOR "${COHERE_TRANSCRIBE_ROOT}/vendor/crispasr")
set(GGML_VENDOR "${CRISPASR_VENDOR}/ggml")

if(NOT EXISTS "${CRISPASR_VENDOR}/src/cohere.cpp" OR NOT EXISTS "${GGML_VENDOR}/CMakeLists.txt")
    message(FATAL_ERROR "The vendored CrispASR/ggml source snapshot is incomplete")
endif()

# ggml diagnostics retain __FILE__ strings in release builds.  Map the
# installation and build roots to stable names so identical source gems do not
# produce path-dependent shared libraries.  Keep the more specific build map
# last because RubyGems normally builds beneath the extracted source root.
set(COHERE_TRANSCRIBE_SOURCE_PREFIX_MAP
    "-ffile-prefix-map=${COHERE_TRANSCRIBE_ROOT}=/cohere-transcribe-source")
set(COHERE_TRANSCRIBE_BUILD_PREFIX_MAP
    "-ffile-prefix-map=${CMAKE_BINARY_DIR}=/cohere-transcribe-build")
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C>:${COHERE_TRANSCRIBE_SOURCE_PREFIX_MAP}>"
        "$<$<COMPILE_LANGUAGE:C>:${COHERE_TRANSCRIBE_BUILD_PREFIX_MAP}>"
    )
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:${COHERE_TRANSCRIBE_SOURCE_PREFIX_MAP}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${COHERE_TRANSCRIBE_BUILD_PREFIX_MAP}>"
    )
endif()

# ggml is linked statically into one private libcrispasr artifact. This avoids
# relying on loader search paths or shipping a family of versioned sibling
# libraries while retaining the upstream target definitions unchanged.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static dependency libraries" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${COHERE_TRANSCRIBE_CUDA} CACHE BOOL "" FORCE)
set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
set(GGML_METAL ${COHERE_TRANSCRIBE_METAL} CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ${COHERE_TRANSCRIBE_METAL} CACHE BOOL "" FORCE)
set(GGML_NATIVE ${COHERE_TRANSCRIBE_NATIVE} CACHE BOOL "" FORCE)
set(GGML_OPENMP ${COHERE_TRANSCRIBE_OPENMP} CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
set(GGML_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_RPC OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_OPENVINO OFF CACHE BOOL "" FORCE)

# With GGML_NATIVE=OFF, upstream ggml otherwise selects a common AVX2/FMA
# profile on x86. Disable those opt-ins for a baseline source gem; builders
# can explicitly request host tuning with COHERE_TRANSCRIBE_NATIVE=ON.
if(NOT COHERE_TRANSCRIBE_NATIVE)
    foreach(feature SSE42 AVX AVX_VNNI AVX2 BMI2 FMA F16C AVX512
                    AVX512_VBMI AVX512_VNNI AVX512_BF16
                    AMX_TILE AMX_INT8 AMX_BF16)
        set(GGML_${feature} OFF CACHE BOOL "" FORCE)
    endforeach()
endif()

# Cohere model tensors contain hierarchical names longer than ggml's stock
# default. This matches the defining setting in the pinned CrispASR build.
add_compile_definitions(GGML_MAX_NAME=128)

add_subdirectory("${GGML_VENDOR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)

if(COHERE_TRANSCRIBE_CUDA)
    # NVCC forwards prefix maps through the host compiler. A response file
    # preserves paths containing spaces, while object-directory temporaries
    # avoid PID-derived tmpxft names that otherwise survive the final link.
    # Its device frontend also derives internal module names from the absolute
    # source path. Keep NVCC's per-file basename and give that path component a
    # stable identity shared by equivalent source trees.
    set(COHERE_TRANSCRIBE_CUDA_PREFIX_MAP_RESPONSE "cohere_cuda_prefix_map.rsp")
    set(COHERE_TRANSCRIBE_CUDA_PREFIX_MAP_CONTENT
        "\"${COHERE_TRANSCRIBE_SOURCE_PREFIX_MAP}\"\n\"${COHERE_TRANSCRIBE_BUILD_PREFIX_MAP}\"\n")
    get_target_property(COHERE_TRANSCRIBE_CUDA_BINARY_DIR ggml-cuda BINARY_DIR)
    set(COHERE_TRANSCRIBE_CUDA_SOURCE_IDENTITY
        "/cohere-transcribe-source/vendor/crispasr/ggml/src/ggml-cuda")
    file(GENERATE
        OUTPUT "${CMAKE_BINARY_DIR}/${COHERE_TRANSCRIBE_CUDA_PREFIX_MAP_RESPONSE}"
        CONTENT "${COHERE_TRANSCRIBE_CUDA_PREFIX_MAP_CONTENT}"
    )
    if(NOT COHERE_TRANSCRIBE_CUDA_BINARY_DIR STREQUAL CMAKE_BINARY_DIR)
        file(GENERATE
            OUTPUT "${COHERE_TRANSCRIBE_CUDA_BINARY_DIR}/${COHERE_TRANSCRIBE_CUDA_PREFIX_MAP_RESPONSE}"
            CONTENT "${COHERE_TRANSCRIBE_CUDA_PREFIX_MAP_CONTENT}"
        )
    endif()
    target_compile_options(ggml-cuda PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=@${COHERE_TRANSCRIBE_CUDA_PREFIX_MAP_RESPONSE}>"
        "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--objdir-as-tempdir>"
        "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcudafe=--orig_src_path_name=${COHERE_TRANSCRIBE_CUDA_SOURCE_IDENTITY}>"
    )

    # CUDA 12.9 added NVCC's supported random-seed option for stable PTX and
    # object output. Give each translation unit a distinct seed derived from
    # its canonical relative path, as required by NVCC. CUDA's ggml subproject
    # already requires CMake 3.18, which supplies TARGET_DIRECTORY here.
    get_target_property(COHERE_TRANSCRIBE_CUDA_SOURCES ggml-cuda SOURCES)
    foreach(COHERE_TRANSCRIBE_CUDA_SOURCE IN LISTS COHERE_TRANSCRIBE_CUDA_SOURCES)
        if(COHERE_TRANSCRIBE_CUDA_SOURCE MATCHES "\\.cu$")
            if(IS_ABSOLUTE "${COHERE_TRANSCRIBE_CUDA_SOURCE}")
                set(COHERE_TRANSCRIBE_CUDA_SOURCE_ABSOLUTE
                    "${COHERE_TRANSCRIBE_CUDA_SOURCE}")
            else()
                get_filename_component(COHERE_TRANSCRIBE_CUDA_SOURCE_ABSOLUTE
                    "${COHERE_TRANSCRIBE_CUDA_SOURCE}" ABSOLUTE
                    BASE_DIR "${GGML_VENDOR}/src/ggml-cuda")
            endif()
            file(RELATIVE_PATH COHERE_TRANSCRIBE_CUDA_SOURCE_RELATIVE
                "${GGML_VENDOR}/src/ggml-cuda"
                "${COHERE_TRANSCRIBE_CUDA_SOURCE_ABSOLUTE}")
            string(MAKE_C_IDENTIFIER "${COHERE_TRANSCRIBE_CUDA_SOURCE_RELATIVE}"
                COHERE_TRANSCRIBE_CUDA_SOURCE_SEED)
            set_property(SOURCE "${COHERE_TRANSCRIBE_CUDA_SOURCE}"
                TARGET_DIRECTORY ggml-cuda APPEND PROPERTY COMPILE_OPTIONS
                "$<$<AND:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>,$<VERSION_GREATER_EQUAL:$<CUDA_COMPILER_VERSION>,12.9>>:--frandom-seed=cohere-transcribe-${COHERE_TRANSCRIBE_CUDA_SOURCE_SEED}>")
        endif()
    endforeach()

    find_package(CUDAToolkit REQUIRED)
    if(CUDAToolkit_VERSION VERSION_LESS "12.9")
        message(STATUS
            "NVCC ${CUDAToolkit_VERSION} predates --frandom-seed; "
            "its generated CUDA objects may differ byte-for-byte across rebuilds")
    endif()
endif()

find_package(Threads REQUIRED)

# ggml's upstream fatal-error path can fork and attach gdb/lldb to itself.
# A library embedded in a Ruby process must never launch a debugger behind the
# caller's back, so retain its in-process backtrace while compiling the
# subprocess branch out of this packaged build. The vendored source keeps its
# upstream default when it is built outside this CMake project.
target_compile_definitions(ggml-base PRIVATE GGML_BACKTRACE_NO_DEBUGGER)

add_library(crispasr SHARED
    cohere_abi.cpp
    "${CRISPASR_VENDOR}/src/cohere.cpp"
    "${CRISPASR_VENDOR}/src/cohere_frontend.cpp"
    "${CRISPASR_VENDOR}/src/core/gguf_loader.cpp"
    "${CRISPASR_VENDOR}/src/core/mel.cpp"
    "${CRISPASR_VENDOR}/src/crispasr_imatrix.cpp"
)

target_compile_features(crispasr PRIVATE cxx_std_20)
target_include_directories(crispasr PRIVATE
    "${CRISPASR_VENDOR}/src"
    "${GGML_VENDOR}/include"
    "${GGML_VENDOR}/src"
)
target_link_libraries(crispasr PRIVATE ggml ggml-base ggml-cpu Threads::Threads)

if(COHERE_TRANSCRIBE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(crispasr PRIVATE OpenMP::OpenMP_CXX)
endif()

if(COHERE_TRANSCRIBE_METAL)
    target_compile_definitions(crispasr PRIVATE GGML_USE_METAL)
    target_link_libraries(crispasr PRIVATE ggml-metal)
endif()

if(COHERE_TRANSCRIBE_CUDA)
    target_compile_definitions(crispasr PRIVATE GGML_USE_CUDA)
    target_link_libraries(crispasr PRIVATE ggml-cuda)
    # Source gems are compiled on the destination machine. Preserve the exact
    # toolkit library directory found by CMake so non-system CUDA installs
    # (including tarball and environment-managed toolkits) remain loadable after
    # RubyGems moves libcrispasr out of the build tree.
    set_property(TARGET crispasr APPEND PROPERTY
        BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}")
    set_property(TARGET crispasr APPEND PROPERTY
        INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")
endif()

if(MSVC)
    target_compile_definitions(crispasr PRIVATE _USE_MATH_DEFINES NOMINMAX)
else()
    target_compile_options(crispasr PRIVATE -O3)
endif()

set_target_properties(crispasr PROPERTIES
    OUTPUT_NAME crispasr
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

if(APPLE)
    target_link_options(crispasr PRIVATE
        "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/exports.macos")
elseif(UNIX)
    # GNU ld records the version script in its generated dependency file.
    # Keep that entry relative and space-free so Ninja can parse it when the
    # source or build root contains spaces.
    set(COHERE_TRANSCRIBE_EXPORTS_MAP "cohere_exports.map")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
        "${CMAKE_BINARY_DIR}/${COHERE_TRANSCRIBE_EXPORTS_MAP}"
        COPYONLY
    )
    target_link_options(crispasr PRIVATE
        "LINKER:--version-script=${COHERE_TRANSCRIBE_EXPORTS_MAP}")
endif()

# Audio decoding remains independent from the model runtime.  This tiny
# adapter dynamically loads a compatible FFmpeg runtime when present, so the
# source gem neither links to a particular libav major nor needs FFmpeg -dev
# headers.  Keeping it in a separate library lets libsndfile-only installs use
# the gem even when FFmpeg is absent.
add_library(cohere_audio SHARED audio_abi.cpp)
target_compile_features(cohere_audio PRIVATE cxx_std_20)
target_link_libraries(cohere_audio PRIVATE ${CMAKE_DL_LIBS})
if(MSVC)
    target_compile_definitions(cohere_audio PRIVATE NOMINMAX)
else()
    target_compile_options(cohere_audio PRIVATE -O3)
endif()
set_target_properties(cohere_audio PROPERTIES
    OUTPUT_NAME cohere_audio
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)
if(APPLE)
    target_link_options(cohere_audio PRIVATE
        "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/audio_exports.macos")
elseif(UNIX)
    set(COHERE_TRANSCRIBE_AUDIO_EXPORTS_MAP "cohere_audio_exports.map")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/audio_exports.map"
        "${CMAKE_BINARY_DIR}/${COHERE_TRANSCRIBE_AUDIO_EXPORTS_MAP}"
        COPYONLY
    )
    target_link_options(cohere_audio PRIVATE
        "LINKER:--version-script=${COHERE_TRANSCRIBE_AUDIO_EXPORTS_MAP}")
endif()

# A dedicated component lets extconf install only the private runtime and not
# ggml's headers, CMake package metadata, or static archives.
install(TARGETS crispasr cohere_audio
    RUNTIME DESTINATION . COMPONENT cohere_transcribe_runtime
    LIBRARY DESTINATION . COMPONENT cohere_transcribe_runtime NAMELINK_COMPONENT cohere_transcribe_runtime
)
