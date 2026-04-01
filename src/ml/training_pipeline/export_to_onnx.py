"""Model Export and Optimization Pipeline for Local 8GB VRAM.

Converts PyTorch models (TFT, CNN, PPO) to ONNX format 
and applies INT8 Quantization for maximum performance.
"""

import torch
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

def export_cnn_to_onnx(model_path="models/cnn_microstructure.pth", onnx_path="models/cnn_microstructure.onnx"):
    """
    Export 1D-CNN to ONNX and apply dynamic quantization.
    """
    from src.ml.training_pipeline.train_cnn import MicrostructureCNN
    model = MicrostructureCNN(num_features=40, seq_len=100)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(1, 40, 100)
    
    # 1. Export to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ CNN Exported to: {onnx_path}")

    # 2. Dynamic Quantization (INT8)
    quant_path = onnx_path.replace(".onnx", "_int8.onnx")
    quantize_dynamic(
        onnx_path, 
        quant_path, 
        weight_type=QuantType.QUInt8
    )
    print(f"💎 CNN Quantized (INT8) to: {quant_path}")

def export_tft_to_onnx(tft_model, onnx_path="models/tft_forecast.onnx"):
    """
    Note: TFT (TemporalFusionTransformer) export to ONNX can be complex 
    due to Attention layers. We recommend using TorchScript for TFT.
    """
    scripted_model = torch.jit.script(tft_model)
    scripted_model.save("models/tft_forecast_scripted.pt")
    print(f"✅ TFT Scripted (TorchScript) to: models/tft_forecast_scripted.pt")

if __name__ == "__main__":
    print("🚀 Model Export Pipeline Initialized. Optimizing for 8GB VRAM...")
    # export_cnn_to_onnx() 
