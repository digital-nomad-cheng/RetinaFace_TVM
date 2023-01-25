#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>
#include <opencv2/opencv.hpp>

void DeployRetinaFaceExecutor() {

  LOG(INFO) << "Load image...";
  cv::Mat image = cv::imread("/home/vagrant/tvm/apps/howto_deploy/RetinaFace_TVM/test.jpg");
  cv::imshow("image", image);
  cv::waitKey(0);
  cv::Mat image_resized = cv::Mat::zeros(320, 480, CV_8UC3);
  cv::resize(image, image_resized, cv::Size(480, 320), 0, 0, cv::INTER_LINEAR);
  cv::imshow("image_resized", image_resized);
  cv::waitKey(0);
  
  LOG(INFO) << "Preprocess image...";
  cv::Scalar mean_pixel(104.0f, 117.0f, 123.0f);
  
  cv::Mat image_float;
  image_resized.convertTo(image_float, CV_32FC3);
  cv::Mat mean_mat = cv::Mat(image_float.rows, image_float.cols, CV_32FC3, mean_pixel); 
  cv::subtract(image_float, mean_mat, image_float);
  
  cv::imshow("subtract mean", image_float);
  cv::waitKey(0);
  
  LOG(INFO) << "copy image to input tensor...";
  
  int channel_offset = 320*480*sizeof(float);

  cv::Mat split_mat[3];
  cv::split(image_float, split_mat);
  LOG(INFO) << "split" << image_float.cols << image_float.rows;
  float* x = (float *)malloc(3*320*480*sizeof(float));
  memcpy(x, split_mat[2].ptr<float>(), channel_offset);
  LOG(INFO) << "channel 0";
  memcpy(x+channel_offset/sizeof(float), split_mat[1].ptr<float>(), channel_offset);
  LOG(INFO) << "channel 1";
  memcpy(x+channel_offset/sizeof(float)*2, split_mat[0].ptr<float>(), channel_offset);
  LOG(INFO) << "channel 2";
   
    


  
  LOG(INFO) << "Running retinaface graph executor...";
  // load in the library
  DLDevice dev{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("lib/retinaface_sim.so");
  // create the graph executor module
  //
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  // Use the C++ API
  tvm::runtime::NDArray input_tensor = tvm::runtime::NDArray::Empty({1, 3, 320, 480}, DLDataType{kDLFloat, 32, 1}, dev);
  tvm::runtime::NDArray  locs = tvm::runtime::NDArray::Empty({1, 6300, 4}, DLDataType{kDLFloat, 32, 1}, dev);
  tvm::runtime::NDArray  confs = tvm::runtime::NDArray::Empty({1, 6300, 2}, DLDataType{kDLFloat, 32, 1}, dev);
  tvm::runtime::NDArray  ldmks = tvm::runtime::NDArray::Empty({1, 6300, 10}, DLDataType{kDLFloat, 32, 1}, dev);
  

  TVMArrayCopyFromBytes(input_tensor, x, channel_offset*3);
  
  /*
  for (int i = 0; i < 624; ++i) {
    for (int j = 0; j < 1024; ++j) {
      static_cast<float*>(x->data)[i * 1024 + j] = i * 2 + j;
    }
  }
  */
  // set the right input
  set_input("input", input_tensor);
  // run the code
  run();
  std::cout << "ok" << std::endl;
  // get the output
  get_output(0, confs);
  get_output(1, locs);
  get_output(2, ldmks);
  
}


int main(void) {
  DeployRetinaFaceExecutor();
  return 0;
}
