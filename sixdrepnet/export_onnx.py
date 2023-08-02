import torch
import onnx
from regressor import SixDRepNet_Detector

if __name__ == '__main__':
    # EXPORT
    model = SixDRepNet_Detector(export=True).model
    model.eval()
    input_names = [ "input" ]
    output_names = [ "output" ]
    dummy_input = torch.randint(0, 255, (10, 224, 224, 3), device="cuda", dtype=torch.uint8)
    torch.onnx.export(model, dummy_input, "SixDRepNetWithPostProcess.onnx", verbose=True, input_names=input_names, 
                    output_names=output_names, dynamic_axes={'input': {0: 'bs'},'output': {0: 'bs', 1:'3'}})
    model = onnx.load('SixDRepNetWithPostProcess.onnx')
    model.doc_string = 'Model need RGB NHWC UINT8 image\nOutput for image: pitch, yaw, roll.'
    onnx.save(model, 'SixDRepNetWithPostProc.onnx')
