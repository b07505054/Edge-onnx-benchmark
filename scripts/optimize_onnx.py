import argparse
import onnx
from onnxsim import simplify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="models/mobilenet_v2_fp32.onnx")
    parser.add_argument("--output", type=str, default="models/mobilenet_v2_optimized.onnx")
    args = parser.parse_args()

    model = onnx.load(args.input)

    simplified_model, check = simplify(model)

    if not check:
        raise RuntimeError("ONNX simplification failed")

    onnx.save(simplified_model, args.output)

    print(f"Saved optimized ONNX model to {args.output}")


if __name__ == "__main__":
    main()