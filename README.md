# RetinaFace_TVM
Optimizing RetinaFace inference using TVM.

Having been exploring TVM for a while, after using the ANSOR(auto-scheduler) to 
generate the cuda kernel [here](https://github.com/digital-nomad-cheng/matmul_cuda_kernel_tvm), I want to use it to deploy some face detection model
that previously deployed using other framework. After several days trials and errors, finally made it.

First, I compared the performance between the default schedule and the schedule generated by ANSOR.
You can just check `schedule/run.py` for exporting the two versions of runtime and `cpp` folder for 
all the C++ inference benchmark details. In a nutshell, on my machine(), the default schedule runtime
is 27.56ms and the optimized schedule runtime is 23.97ms - almost 13% speedup.

![result.jpg](https://github.com/digital-nomad-cheng/RetinaFace_TVM/blob/main/result.jpg)
## Main Contributions
- Demonstrate how to use tvm runtime library under C++.
- Benchmark tvm scheduler performance.
- Show how to use tvm for object detection task.
