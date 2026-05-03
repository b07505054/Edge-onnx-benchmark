#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <random>
#include <fstream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: cpp_inference <model_path>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edge_onnx_cpp");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
    std::wstring w_model_path(model_path.begin(), model_path.end());
    Ort::Session session(env, w_model_path.c_str(), session_options);
#else
    Ort::Session session(env, model_path.c_str(), session_options);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);

    const char* input_name = input_name_alloc.get();
    const char* output_name = output_name_alloc.get();

    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    size_t input_tensor_size = 1 * 3 * 224 * 224;

    std::vector<float> input_tensor_values(input_tensor_size);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& v : input_tensor_values) {
        v = dist(gen);
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_size,
        input_shape.data(),
        input_shape.size()
    );

    std::vector<const char*> input_names = {input_name};
    std::vector<const char*> output_names = {output_name};

    int warmup = 20;
    int iterations = 100;

    for (int i = 0; i < warmup; ++i) {
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            1
        );
    }

    std::vector<double> latencies;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            1
        );

        auto end = std::chrono::high_resolution_clock::now();

        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(latency_ms);
    }

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg_latency = sum / latencies.size();

    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Backend: ONNX Runtime C++" << std::endl;
    std::cout << "Input shape: [1, 3, 224, 224]" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Average latency: " << avg_latency << " ms" << std::endl;
    std::ofstream out("results/cpp_benchmark.csv", std::ios::app);

    out << "backend,model,avg_latency_ms\n";
    out << "ONNX Runtime C++," << model_path << "," << avg_latency << "\n";

    out.close();
    return 0;
}