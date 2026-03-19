#!/usr/bin/env python3

import sys


def main() -> int:
    errors = []

    try:
        import torch
    except Exception as exc:
        torch = None
        errors.append(f"Failed to import torch: {exc}")

    try:
        import faiss
    except Exception as exc:
        faiss = None
        errors.append(f"Failed to import faiss: {exc}")

    print("=== GPU Readiness Check ===")

    if torch is not None:
        cuda_available = torch.cuda.is_available()
        print(f"torch version        : {torch.__version__}")
        print(f"torch cuda available : {cuda_available}")
        print(f"torch cuda devices   : {torch.cuda.device_count()}")
        if cuda_available:
            print(f"torch device[0]      : {torch.cuda.get_device_name(0)}")
        else:
            errors.append("PyTorch cannot see a CUDA GPU")

    if faiss is not None:
        faiss_gpu_count = faiss.get_num_gpus()
        version = getattr(faiss, "__version__", "unknown")
        print(f"faiss version        : {version}")
        print(f"faiss gpu count      : {faiss_gpu_count}")
        if faiss_gpu_count < 1:
            errors.append("FAISS cannot see any GPU")

    if errors:
        print("\nStatus: FAILED")
        for issue in errors:
            print(f"- {issue}")
        return 1

    print("\nStatus: OK (GPU is ready for torch + faiss)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())