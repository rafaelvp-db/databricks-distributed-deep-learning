# Databricks notebook source
!pip install --upgrade torch torchvision

# COMMAND ----------

from torchvision import models, datasets, transforms as T
import torch
from PIL import Image
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Serialize to ONNX

# COMMAND ----------

resnet50 = models.resnet50(pretrained=True)

# Download ImageNet labels
!curl -o /tmp/imagenet_classes.txt https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

# Read the categories
with open("/tmp/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Export the model to ONNX
image_height = 224
image_width = 224
x = torch.randn(1, 3, image_height, image_width, requires_grad=True)
torch_out = resnet50(x)
torch.onnx.export(
  resnet50,                     # model being run
  x,                            # model input (or a tuple for multiple inputs)
  "/tmp/resnet50.onnx",         # where to save the model (can be a file or file-like object)
  export_params=True,           # store the trained parameter weights inside the model file
  opset_version=12,             # the ONNX version to export the model to
  do_constant_folding=True,     # whether to execute constant folding for optimization
  input_names = ['input'],      # the model's input names
  output_names = ['output']     # the model's output names
)    

# COMMAND ----------

# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "/tmp/dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# COMMAND ----------

# Pre-processing for ResNet-50 Inferencing, from https://pytorch.org/hub/pytorch_vision_resnet/
resnet50.eval()  
filename = '/tmp/dog.jpg' # change to your filename

input_image = Image.open(filename)
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
print("GPU Availability: ", torch.cuda.is_available())
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    resnet50.to('cuda')

# COMMAND ----------

# Inference with ONNX Runtime
import onnxruntime
from onnx import numpy_helper
import time

session_fp32 = onnxruntime.InferenceSession("/tmp/resnet50.onnx", providers=['CPUExecutionProvider'])
# session_fp32 = onnxruntime.InferenceSession("resnet50.onnx", providers=['CUDAExecutionProvider'])
# session_fp32 = onnxruntime.InferenceSession("resnet50.onnx", providers=['OpenVINOExecutionProvider'])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

latency = []
def run_sample(session, image_file, categories, inputs):
    start = time.time()
    input_arr = inputs.cpu().detach().numpy()
    ort_outputs = session.run([], {'input':input_arr})[0]
    latency.append(time.time() - start)
    output = ort_outputs.flatten()
    output = softmax(output) # this is optional
    top5_catid = np.argsort(-output)[:5]
    for catid in top5_catid:
        print(categories[catid], output[catid])
    return ort_outputs

ort_output = run_sample(session_fp32, '/tmp/dog.jpg', categories, input_batch)
print("ONNX Runtime CPU/GPU/OpenVINO Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## OpenVINO

# COMMAND ----------

!pip install openvino-dev["torch"]

# COMMAND ----------

# DBTITLE 1,Comparison with OpenVINO
# Inference with OpenVINO
from openvino.runtime import Core

ie = Core()
onnx_model_path = "/tmp/resnet50.onnx"
model_onnx = ie.read_model(model=onnx_model_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

# inference
output_layer = next(iter(compiled_model_onnx.outputs))

latency = []
input_arr = input_batch.detach().numpy()
inputs = {'input': input_arr}
start = time.time()
request = compiled_model_onnx.create_infer_request()
output = request.infer(inputs=inputs)

outputs = request.get_output_tensor(output_layer.index).data
latency.append(time.time() - start)

print("OpenVINO CPU Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))

print("***** Verifying correctness *****")
for i in range(2):
    print('OpenVINO and ONNX Runtime output {} are close:'.format(i), np.allclose(ort_output, outputs, rtol=1e-05, atol=1e-04))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## PyTorch

# COMMAND ----------

!pip install validators

# COMMAND ----------

# DBTITLE 1,Comparison with PyTorch
filename = '/tmp/dog.jpg'

input_image = Image.open(filename)
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
input_batch = input_batch.to("cpu")

with torch.no_grad():
  output = torch.nn.functional.softmax(resnet50(input_batch), dim=1)
    
#results = utils.pick_n_best(predictions=output, n=1)

result = torch.topk(output, k = 5)
print(result)

# COMMAND ----------

index = torch.flatten(result.indices)[0].item()
categories[index]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## File Sizes

# COMMAND ----------

!ls "/tmp/resnet50.onnx" -all

# COMMAND ----------

torch.save(resnet50, "/tmp/model.bin")

# COMMAND ----------

!ls /tmp/model.bin -all

# COMMAND ----------

# TorchScript

# Switch the model to eval model
resnet50.eval()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(resnet50, input_batch)

# Save the TorchScript model
traced_script_module.save("/tmp/traced_resnet_model.pt")

# COMMAND ----------

!ls /tmp/traced_resnet_model.pt -all

# COMMAND ----------

# Further reference

# https://github.com/mtszkw/fast-torch
