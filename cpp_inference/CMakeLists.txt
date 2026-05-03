cmake_minimum_required(VERSION 3.15)

project(edge_onnx_cpp_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ONNXRUNTIME_DIR)
    message(FATAL_ERROR "Please set ONNXRUNTIME_DIR to your ONNX Runtime installation path")
endif()

include_directories(${ONNXRUNTIME_DIR}/include)

add_executable(edge_onnx_cpp main.cpp)

if(WIN32)
    target_link_libraries(edge_onnx_cpp ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)
else()
    target_link_libraries(edge_onnx_cpp ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
endif()