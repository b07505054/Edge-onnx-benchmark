import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="models/mobilenet_v2_optimized.onnx")
    parser.add_argument("--output", type=str, default="models/mobilenet_v2_int8.onnx")
    args = parser.parse_args()

    quantize_dynamic(
        model_input=args.input,
        model_output=args.output,
        weight_type=QuantType.QInt8,
    )

    print(f"Saved INT8 quantized model to {args.output}")


if __name__ == "__main__":
    main()