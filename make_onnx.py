import os
import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np
import cv2
import argparse
from utils import IMG_SIZE, bilinear_unwarping, load_model
import copy 

def export_to_onnx(original_model, img_size, output_path, precision='fp32'): 
    model = copy.deepcopy(original_model)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) 

    dummy_input = torch.randn(1, 3, img_size[1], img_size[0]).to(device)

    if precision == 'fp16':
        model = model.half()
        dummy_input = dummy_input.half()
    else: 
        model = model.float()

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['point_positions2D', 'output2'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'point_positions2D': {0: 'batch_size'},
            'output2': {0: 'batch_size'}
        }
    )

    print(f"Đã xuất model sang ONNX ({precision}) với kích thước cố định: {output_path}")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Model ONNX hợp lệ: {output_path}")

def export_to_torchscript(original_model, img_size, output_path, precision='fp32'): 
    model = copy.deepcopy(original_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval() 

    example_input = torch.randn(1, 3, img_size[1], img_size[0]).to(device)

    if precision == 'fp16':
        model = model.half()
        example_input = example_input.half()
    else: 
        model = model.float()

    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(output_path)

    print(f"Đã xuất model sang TorchScript ({precision}): {output_path}")

def run_onnx_inference(onnx_model_path, img_path, model_img_size, precision='fp32'):

    # Tạo ONNX Runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_model_path, providers=providers)

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Không thể load ảnh: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_processed = img_rgb.astype(np.float32) / 255.0


    inp = cv2.resize(img_processed, model_img_size).transpose(2, 0, 1)[np.newaxis, ...]

    if precision == 'fp16':
        inp = inp.astype(np.float16)
    else:
        inp = inp.astype(np.float32)

    # Chạy inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inp})
    point_positions2D = outputs[0]  # [1, 2, Gh, Gw]

    size = img_rgb.shape[:2][::-1] 

    warped_img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0 
    point_positions_tensor = torch.from_numpy(point_positions2D).float()

    unwarped = bilinear_unwarping(
        warped_img=warped_img_tensor,
        point_positions=point_positions_tensor,
        img_size=tuple(size),
    )

    unwarped_np = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    unwarped_BGR = cv2.cvtColor(unwarped_np, cv2.COLOR_RGB2BGR)    

    ten_file, _ = os.path.splitext(os.path.basename(img_path))
    output_path = "./img_output/" + ten_file + f"_unwarp_onnx_{precision}.png"    
    cv2.imwrite(output_path, unwarped_BGR)


    return unwarped_BGR


def run_torchscript_inference(pt_model_path, img_path, model_img_size, precision='fp32'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running TorchScript Inference on device: {device} ---")

    model = torch.jit.load(pt_model_path, map_location=device)
    model.eval()

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Không thể load ảnh: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_processed = img_rgb.astype(np.float32) / 255.0

    inp = torch.from_numpy(cv2.resize(img_processed, model_img_size).transpose(2, 0, 1)).unsqueeze(0)
    inp = inp.to(device)

    if 'fp16' in pt_model_path.lower():
        if inp.dtype != torch.float16:
            print("Casting input to FP16 for FP16 TorchScript model.")
            inp = inp.half()

    with torch.no_grad():
        try:
            outputs = model(inp)
            if isinstance(outputs, tuple) and len(outputs) > 0:
                point_positions2D = outputs[0]
            else:
                point_positions2D = outputs 
        except Exception as e:
            print(f"Initial model call failed: {e}. Retrying assuming single output.")
            point_positions2D = model(inp)


    size = img_rgb.shape[:2][::-1]
    warped_img_tensor_orig = torch.from_numpy(img_rgb.transpose(2,0,1)).unsqueeze(0).float() / 255.0
    warped_img_tensor_orig = warped_img_tensor_orig.to(device)
    point_positions2D_on_device = point_positions2D.to(device)

   
    if warped_img_tensor_orig.dtype != point_positions2D_on_device.dtype:
        print(f"Casting warped_img_tensor_orig from {warped_img_tensor_orig.dtype} to {point_positions2D_on_device.dtype} to match point_positions dtype.")
        warped_img_tensor_orig = warped_img_tensor_orig.to(point_positions2D_on_device.dtype)

    if point_positions2D_on_device.dim() == 4 and point_positions2D_on_device.size(0) > 0:
        point_positions_for_unwarp = point_positions2D_on_device[0].unsqueeze(0)
    else:
        point_positions_for_unwarp = point_positions2D_on_device if point_positions2D_on_device.dim() == 4 else point_positions2D_on_device.unsqueeze(0)

    print(f"Device of warped_img_tensor_orig (before bilinear_unwarping): {warped_img_tensor_orig.device}, Dtype: {warped_img_tensor_orig.dtype}")
    print(f"Device of point_positions_for_unwarp (before bilinear_unwarping): {point_positions_for_unwarp.device}, Dtype: {point_positions_for_unwarp.dtype}")

    unwarped = bilinear_unwarping(
        warped_img=warped_img_tensor_orig,
        point_positions=point_positions_for_unwarp,
        img_size=tuple(size),
    )

    unwarped_np = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    unwarped_BGR = cv2.cvtColor(unwarped_np, cv2.COLOR_RGB2BGR)
    output_filename_precision = "fp16" if 'fp16' in pt_model_path.lower() else "fp32"
    if precision != output_filename_precision: 
        print(f"Warning: function precision '{precision}' differs from model path precision '{output_filename_precision}'. Using model path for filename.")

    output_path = os.path.splitext(img_path)[0] + f"_unwarp_torchscript_{output_filename_precision}.png"
    cv2.imwrite(output_path, unwarped_BGR)

    print(f"Đã lưu ảnh unwarped TorchScript ({output_filename_precision}): {output_path}")
    return unwarped_BGR

def run_pytorch_inference(ckpt_path, img_path, model_img_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(ckpt_path, device) 
    model.to(device)
    model.eval()

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Không thể load ảnh: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_processed = img_rgb.astype(np.float32) / 255.0

    inp = torch.from_numpy(cv2.resize(img_processed, model_img_size).transpose(2, 0, 1)).unsqueeze(0)

    # Make prediction
    inp = inp.to(device)
    with torch.no_grad():
        try:           
            outputs = model(inp)
            if isinstance(outputs, tuple) and len(outputs) > 0:
                point_positions2D = outputs[0]
            else:
                point_positions2D = outputs 
        except Exception as e:             
            print(f"Initial model call failed: {e}. Retrying assuming single output.")
            point_positions2D = model(inp) 

    # Unwarp
    size = img_rgb.shape[:2][::-1]
    warped_img_tensor_orig = torch.from_numpy(img_rgb.transpose(2,0,1)).unsqueeze(0).float().to(device) / 255.0

    if point_positions2D.dim() == 4 and point_positions2D.size(0) > 0:
         point_positions_for_unwarp = point_positions2D[0].unsqueeze(0)
    else:
         point_positions_for_unwarp = point_positions2D if point_positions2D.dim() == 4 else point_positions2D.unsqueeze(0)


    unwarped = bilinear_unwarping(
        warped_img=warped_img_tensor_orig,
        point_positions=point_positions_for_unwarp, 
        img_size=tuple(size),
    )
    unwarped_np = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # Save result
    unwarped_BGR = cv2.cvtColor(unwarped_np, cv2.COLOR_RGB2BGR)
    output_path = os.path.splitext(img_path)[0] + "_unwarp_pytorch.png"
    cv2.imwrite(output_path, unwarped_BGR)

    print(f"Đã lưu ảnh unwarped PyTorch: {output_path}")
    return unwarped_BGR

def convert_and_test(ckpt_path, img_path, model_img_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_model = load_model(ckpt_path, device) 


    output_dir = "exported_models"
    os.makedirs(output_dir, exist_ok=True)

    print("=== XUẤT MODEL ===")

    # Export ONNX với fp32
    onnx_fp32_path = os.path.join(output_dir, "uvdocnet_fp32.onnx")
    export_to_onnx(original_model, model_img_size, onnx_fp32_path, precision='fp32')

    # Export ONNX với fp16
    onnx_fp16_path = os.path.join(output_dir, "uvdocnet_fp16.onnx")
    export_to_onnx(original_model, model_img_size, onnx_fp16_path, precision='fp16')

    # Export TorchScript với fp32
    pt_fp32_path = os.path.join(output_dir, "uvdocnet_fp32.pt")
    export_to_torchscript(original_model, model_img_size, pt_fp32_path, precision='fp32')

    # Export TorchScript với fp16
    pt_fp16_path = os.path.join(output_dir, "uvdocnet_fp16.pt")
    export_to_torchscript(original_model, model_img_size, pt_fp16_path, precision='fp16')

    if img_path and os.path.exists(img_path):
        print("\n=== CHẠY INFERENCE ===")

        print("\n--- PyTorch Inference ---")
        run_pytorch_inference(ckpt_path, img_path, model_img_size)

        # ONNX inference
        print("\n--- ONNX Inference ---")
        run_onnx_inference(onnx_fp32_path, img_path, model_img_size, precision='fp32')
        run_onnx_inference(onnx_fp16_path, img_path, model_img_size, precision='fp16')

        # TorchScript inference
        print("\n--- TorchScript Inference ---")
        run_torchscript_inference(pt_fp32_path, img_path, model_img_size, precision='fp32')
        run_torchscript_inference(pt_fp16_path, img_path, model_img_size, precision='fp16')

    else:
        print("Không có ảnh test hoặc ảnh không tồn tại. Chỉ export model.")

def benchmark_models(ckpt_path, onnx_fp32_path, onnx_fp16_path, pt_fp32_path, pt_fp16_path, img_path, model_img_size, num_runs=10):

    import time

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp_resized_np = cv2.resize(img_rgb, model_img_size) # (H, W, C)

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Benchmark PyTorch
    pytorch_model = load_model(ckpt_path).to(device).eval()
    inp_tensor_pytorch = torch.from_numpy(inp_resized_np.transpose(2, 0, 1)).unsqueeze(0).to(device) # (1, C, H, W)

    # Warm-up PyTorch
    for _ in range(5):
        with torch.no_grad():
            _ = pytorch_model(inp_tensor_pytorch)
    if torch.cuda.is_available(): torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = pytorch_model(inp_tensor_pytorch)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    results['PyTorch FP32'] = (time.time() - start_time) / num_runs
    del pytorch_model 

    # Benchmark ONNX
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    session_fp32 = ort.InferenceSession(onnx_fp32_path, providers=providers)
    inp_np_onnx_fp32 = inp_resized_np.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    input_name_onnx = session_fp32.get_inputs()[0].name

    # Warm-up ONNX FP32
    for _ in range(5):
        _ = session_fp32.run(None, {input_name_onnx: inp_np_onnx_fp32})
    if torch.cuda.is_available() and 'CUDAExecutionProvider' in providers: torch.cuda.synchronize()


    start_time = time.time()
    for _ in range(num_runs):
        _ = session_fp32.run(None, {input_name_onnx: inp_np_onnx_fp32})
    if torch.cuda.is_available() and 'CUDAExecutionProvider' in providers: torch.cuda.synchronize()
    results['ONNX FP32'] = (time.time() - start_time) / num_runs
    del session_fp32

    session_fp16 = ort.InferenceSession(onnx_fp16_path, providers=providers)
    inp_np_onnx_fp16 = inp_resized_np.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float16)

    # Warm-up ONNX FP16
    for _ in range(5):
        _ = session_fp16.run(None, {input_name_onnx: inp_np_onnx_fp16})
    if torch.cuda.is_available() and 'CUDAExecutionProvider' in providers: torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_runs):
        _ = session_fp16.run(None, {input_name_onnx: inp_np_onnx_fp16})
    if torch.cuda.is_available() and 'CUDAExecutionProvider' in providers: torch.cuda.synchronize()
    results['ONNX FP16'] = (time.time() - start_time) / num_runs
    del session_fp16


    ts_model_fp32 = torch.jit.load(pt_fp32_path, map_location=device).eval()
    inp_tensor_ts_fp32 = torch.from_numpy(inp_resized_np.transpose(2,0,1)).unsqueeze(0).to(device)

    for _ in range(5):
        with torch.no_grad():
            _ = ts_model_fp32(inp_tensor_ts_fp32)
    if torch.cuda.is_available(): torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = ts_model_fp32(inp_tensor_ts_fp32)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    results['TorchScript FP32'] = (time.time() - start_time) / num_runs
    del ts_model_fp32

    ts_model_fp16 = torch.jit.load(pt_fp16_path, map_location=device).eval() 
    inp_tensor_ts_fp16 = torch.from_numpy(inp_resized_np.transpose(2,0,1)).unsqueeze(0).to(device).half() 

    for _ in range(5):
        with torch.no_grad():
            _ = ts_model_fp16(inp_tensor_ts_fp16)
    if torch.cuda.is_available(): torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = ts_model_fp16(inp_tensor_ts_fp16)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    results['TorchScript FP16'] = (time.time() - start_time) / num_runs
    del ts_model_fp16

    print(f"\n=== BENCHMARK RESULTS ({num_runs} runs, after warm-up) ===")
    for name, time_avg in results.items():
        print(f"{name}: {time_avg*1000:.2f} ms") 

    if 'PyTorch FP32' in results:
        baseline = results['PyTorch FP32']
        print(f"\nSpeedup so với PyTorch FP32:")
        for name, time_avg in results.items():
            if name != 'PyTorch FP32' and time_avg > 0: 
                speedup = baseline / time_avg
                print(f"{name}: {speedup:.2f}x")
            elif name == 'PyTorch FP32':
                 print(f"{name}: 1.00x (Baseline)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi UVDocNet sang ONNX/TorchScript và chạy inference")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="./model/best_model.pkl",
        help="Đường dẫn đến model weights (pkl file)"
    )
    parser.add_argument(
        "--img-path",
        type=str,
        help="Đường dẫn ảnh để test inference"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Chạy benchmark so sánh các format"
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=IMG_SIZE[1],
        help="Chiều cao ảnh cho model input"
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=IMG_SIZE[0], 
        help="Chiều rộng ảnh cho model input"
    )

    args = parser.parse_args()
    MODEL_IMG_SIZE_RUNTIME = (args.img_width, args.img_height)


    if not os.path.exists(args.ckpt_path):
        print(f"Lỗi: Không tìm thấy file checkpoint: {args.ckpt_path}")
    elif args.img_path and not os.path.exists(args.img_path):
        print(f"Lỗi: Không tìm thấy file ảnh test: {args.img_path}. Sẽ chỉ export model.")
        convert_and_test(args.ckpt_path, None, MODEL_IMG_SIZE_RUNTIME)
    else:
        convert_and_test(args.ckpt_path, args.img_path, MODEL_IMG_SIZE_RUNTIME)

        if args.benchmark:
            if not args.img_path:
                print("Cần --img-path để chạy benchmark.")
            else:
                output_dir = "exported_models"
                onnx_fp32 = os.path.join(output_dir, "uvdocnet_fp32.onnx")
                onnx_fp16 = os.path.join(output_dir, "uvdocnet_fp16.onnx")
                pt_fp32 = os.path.join(output_dir, "uvdocnet_fp32.pt")
                pt_fp16 = os.path.join(output_dir, "uvdocnet_fp16.pt")

                if all(map(os.path.exists, [onnx_fp32, onnx_fp16, pt_fp32, pt_fp16])):
                    benchmark_models(
                        args.ckpt_path,
                        onnx_fp32,
                        onnx_fp16,
                        pt_fp32,
                        pt_fp16,
                        args.img_path,
                        MODEL_IMG_SIZE_RUNTIME
                    )
                else:
                    print("Một vài model đã export không tìm thấy. Bỏ qua benchmark.")