import argparse
import os
import cv2 
from utils import IMG_SIZE, bilinear_unwarping, load_model

try:
    from make_onnx import run_onnx_inference
    from utils import IMG_SIZE, bilinear_unwarping 
except ImportError as e:
    print(f"Lỗi import: {e}")
    print("Hãy đảm bảo rằng file make_onnx.py và utils.py nằm trong cùng thư mục hoặc trong PYTHONPATH.")
    exit(1)

def main(args):
   
    onnx_fp32_path = args.onnx_fp32_path
    onnx_fp16_path = args.onnx_fp16_path
    img_path = args.img_path
    model_img_size = (args.img_width, args.img_height)

    if not os.path.exists(img_path):
        print(f"Lỗi: Không tìm thấy ảnh test tại: {img_path}")
        return

    if not os.path.exists(onnx_fp32_path):
        print(f"Lỗi: Không tìm thấy model ONNX FP32 tại: {onnx_fp32_path}")
        print(f"Hãy chạy 'python make_onnx.py --ckpt-path /path/to/your/best_model.pkl --img-path {img_path}' trước.")
        return

    if not os.path.exists(onnx_fp16_path):
        print(f"Lỗi: Không tìm thấy model ONNX FP16 tại: {onnx_fp16_path}")
        print(f"Hãy chạy 'python make_onnx.py --ckpt-path /path/to/your/best_model.pkl --img-path {img_path}' trước.")
        return

    print(f"--- Chạy Inference trên ảnh: {img_path} ---")
    print(f"--- Kích thước input model: {model_img_size} ---")

    # Chạy inference với model ONNX FP32
    print("\n--- ONNX FP32 Inference ---")
    try:
        run_onnx_inference(
            onnx_model_path=onnx_fp32_path,
            img_path=img_path,
            model_img_size=model_img_size,
            precision='fp32'
        )
        print(f"Đã hoàn thành inference với ONNX FP32.")
    except Exception as e:
        print(f"Lỗi khi chạy ONNX FP32 inference: {e}")

    # Chạy inference với model ONNX FP16
    print("\n--- ONNX FP16 Inference ---")
    try:
        run_onnx_inference(
            onnx_model_path=onnx_fp16_path,
            img_path=img_path,
            model_img_size=model_img_size,
            precision='fp16'
        )
        print(f"Đã hoàn thành inference với ONNX FP16.")
    except Exception as e:
        print(f"Lỗi khi chạy ONNX FP16 inference: {e}")

    print("\n--- Tất cả các tác vụ inference ONNX đã hoàn tất. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy inference với các model ONNX đã export.")
    parser.add_argument(
        "--onnx-fp32-path",
        type=str,
        default="./exported_models/uvdocnet_fp32.onnx",
        help="Đường dẫn đến model ONNX FP32 (.onnx)"
    )
    parser.add_argument(
        "--onnx-fp16-path",
        type=str,
        default="./exported_models/uvdocnet_fp16.onnx",
        help="Đường dẫn đến model ONNX FP16 (.onnx)"
    )
    parser.add_argument(
        "--img-path",
        type=str,
        required=True,
        help="Đường dẫn đến ảnh để test inference"
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=IMG_SIZE[1] if 'IMG_SIZE' in globals() else 712, # Sử dụng IMG_SIZE từ utils nếu import thành công
        help="Chiều cao ảnh cho model input"
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=IMG_SIZE[0] if 'IMG_SIZE' in globals() else 488, # Sử dụng IMG_SIZE từ utils nếu import thành công
        help="Chiều rộng ảnh cho model input"
    )

    args = parser.parse_args()
    main(args)