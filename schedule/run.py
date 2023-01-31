import sys

import cv2
import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
import torch
import numpy as np
import onnx 

# load onnx model
onnx_model = onnx.load_model("retinaface_sim_.onnx")

# import the graph using TVM frontend Relay
input_name = "input"
input_shape = [1, 3, 320, 320]
shape_dict = {input_name: input_shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# build the graph to llvm target with given optimizations
target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# performance benchmark
dtype = "float32"
module = graph_executor.GraphModule(lib["default"](dev))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("input", data_tvm)
timer = module.benchmark(dev, repeat=32, min_repeat_ms=500)
print("running time before auto schedule:", timer.mean)

# export the unoptimied as runtime library
lib.export_library("/home/vagrant/work/RetinaFace_TVM/schedule/retinaface_sim_navie.so")

# ------------------- auto scheduler part ------------------ #
# search tasks and their weights
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
print("total tasks:", len(tasks))
for i, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (i, task.workload_key))
    print(task.compute_dag)

# auto schedule and keep records in log_file, you can resume your tuning from it
log_file = "tune_result.json"
tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=20000,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
)
tuner.tune(tune_option)

# apply the best schedule record to module and build library
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)


# performance benchmark
dtype = "float32"
module = graph_executor.GraphModule(lib["default"](dev))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("input", data_tvm)
timer = module.benchmark(dev, repeat=32, min_repeat_ms=500)
print("running time after auto schedule:", timer.mean)

# export optimal as runtime library after auto scheduling
lib.export_library("/home/vagrant/work/RetinaFace_TVM/schedule/retinaface_sim_opt.so")

