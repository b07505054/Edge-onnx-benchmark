import argparse
import os
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import onnxruntime as ort


def run_pytorch(batch_size=1, samples=100):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()

    correct_match = 0

    with torch.no_grad():
        for _ in range(samples):
            x = torch.randn(batch_size, 3, 224, 224)
            y = model(x)
            pred = torch.argmax(y, dim=1).cpu().numpy()

            # self-consistency baseline
            ref_pred = pred
            correct_match += np.sum(pred == ref_pred)

    return correct_match / samples


def run_onnx(model_path, batch_size=1, samples=100):
    torch_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    torch_model.eval()

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    match = 0

    with torch.no_grad():
        for _ in range(samples):
            x = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

            torch_x = torch.from_numpy(x)
            torch_output = torch_model(torch_x)
            torch_pred = torch.argmax(torch_output, dim=1).cpu().numpy()

            onnx_output = session.run(None, {input_name: x})[0]
            onnx_pred = np.argmax(onnx_output, axis=1)

            match += np.sum(torch_pred == onnx_pred)
    return match / samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    consistency = run_onnx(
        model_path=args.model_path,
        batch_size=args.batch_size,
        samples=args.samples,
    )

    model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    result = {
        "model": model_name,
        "batch_size": args.batch_size,
        "samples": args.samples,
        "top1_consistency_with_pytorch": consistency,
        "consistency_drop": 1.0 - consistency,
    }

    os.makedirs("results", exist_ok=True)
    output_path = f"results/{model_name}_accuracy.csv"

    df = pd.DataFrame([result])
    df.to_csv(output_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved accuracy result to {output_path}")


if __name__ == "__main__":
    main()