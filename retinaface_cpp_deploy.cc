#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>
#include <opencv2/opencv.hpp>

struct Point {
    float x;
    float y;
};

struct Box
{
    /* 
     * cx: x of box center
     * cy: y of box center
     * sx: width of box
     * sy: height of box
     */

    float cx;
    float cy;
    float sx;
    float sy;
};

struct BBox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    Point landmarks[5];
};

std::vector<Box> create_anchors(int _w, int _h) {
  std::vector<Box> anchors;
  
  std::vector<std::vector<int> > feature_map(3), anchor_sizes(3);
  float strides[3] = {8, 16, 32};
  for (int i = 0; i < feature_map.size(); ++i) {
    feature_map[i].push_back(ceil(_h/strides[i]));
    feature_map[i].push_back(ceil(_w/strides[i]));
  }
  std::vector<int> stage1_size = {10, 20};
  anchor_sizes[0] = stage1_size;
  std::vector<int> stage2_size = {32, 64};
  anchor_sizes[1] = stage2_size;
    std::vector<int> stage3_size = {128, 256};
    anchor_sizes[2] = stage3_size;
    
    for (int k = 0; k < feature_map.size(); ++k) {
        std::vector<int> anchor_size = anchor_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i) {
            for (int j = 0; j < feature_map[k][1]; ++j) {
                for (int l = 0; l < anchor_size.size(); ++l) {
                    float kx = anchor_size[l]* 1.0 / _w;
                    float ky = anchor_size[l]* 1.0 / _h;
                    float cx = (j + 0.5) * strides[k] / _w;
                    float cy = (i + 0.5) * strides[k] / _h;
                    anchors.push_back({cx, cy, kx, ky});
                }
            }
        }
    }
  return anchors;
}

void nms(std::vector<BBox>& bboxes, float nms_threshold)
{
    std::vector<float> bbox_areas(bboxes.size());
    for (int i = 0; i < bboxes.size(); i++) {
        bbox_areas[i] = (bboxes.at(i).x2 - bboxes.at(i).x1 + 1) * (bboxes.at(i).y2 - bboxes.at(i).y1 + 1);
    }

    for (int i = 0; i < bboxes.size(); i++) {
        for (int j = i + 1; j < bboxes.size(); ) {
            float xx1 = std::max(bboxes[i].x1, bboxes[j].x1);
            float yy1 = std::max(bboxes[i].y1, bboxes[j].y1);
            float xx2 = std::min(bboxes[i].x2, bboxes[j].x2);
            float yy2 = std::min(bboxes[i].y2, bboxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float IoU = inter / (bbox_areas[i] + bbox_areas[j] - inter);
            if (IoU >= nms_threshold) {
                bboxes.erase(bboxes.begin() + j);
                bbox_areas.erase(bbox_areas.begin() + j);
            } else {
                j++;
            }
        }
    }
}

static inline void clip_bboxes(BBox& bbox, int w, int h)
{
    if(bbox.x1 < 0) bbox.x1 = 0;
    if(bbox.y1 < 0) bbox.y1 = 0;
    if(bbox.x2 > w) bbox.x2 = w;
    if(bbox.y2 > h) bbox.y2 = h;
}

void DeployRetinaFaceExecutor() {

  float _nms_threshold = 0.4;
  float _score_threshold = 0.6;

  LOG(INFO) << "Load image...";
  cv::Mat image = cv::imread("/home/vagrant/tvm/apps/howto_deploy/RetinaFace_TVM/test.jpg");
  cv::imshow("image", image);
  cv::waitKey(0);
  cv::Mat image_resized = cv::Mat::zeros(320, 320, CV_8UC3);
  cv::resize(image, image_resized, cv::Size(320, 320), 0, 0, cv::INTER_LINEAR);
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
  
  int channel_offset = 320*320*sizeof(float);

  cv::Mat split_mat[3];
  cv::split(image_float, split_mat);
  LOG(INFO) << "split" << image_float.cols << image_float.rows;
  float* x = (float *)malloc(3*320*320*sizeof(float));
  memcpy(x, split_mat[2].ptr<float>(), channel_offset);
  LOG(INFO) << "channel 0";
  memcpy(x+channel_offset/sizeof(float), split_mat[1].ptr<float>(), channel_offset);
  LOG(INFO) << "channel 1";
  memcpy(x+channel_offset/sizeof(float)*2, split_mat[0].ptr<float>(), channel_offset);
  LOG(INFO) << "channel 2";
   
    

  LOG(INFO) << "create anchors for w: " << 320 << " h: " << 320 << "...";
  auto anchors = create_anchors(320, 320);
  
  LOG(INFO) << "Running retinaface graph executor...";
  // load in the library
  DLDevice dev{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("lib/retinaface_sim_.so");
  // create the graph executor module
  //
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  // Use the C++ API
  // tvm::runtime::NDArray input_tensor = tvm::runtime::NDArray::Empty({1, 3, 320, 320}, DLDataType{kDLFloat, 32, 1}, dev);
  DLTensor* input_tensor;
  int64_t input_shape[4] = {1, 3, 320, 320};
  
  TVMArrayAlloc(input_shape, 4, kDLFloat, 32, 1, kDLCPU, 0, &input_tensor);
  
  tvm::runtime::NDArray  locs = tvm::runtime::NDArray::Empty({1, 4200, 4}, DLDataType{kDLFloat, 32, 1}, dev);
  tvm::runtime::NDArray  confs = tvm::runtime::NDArray::Empty({1, 4200, 2}, DLDataType{kDLFloat, 32, 1}, dev);
  tvm::runtime::NDArray  ldmks = tvm::runtime::NDArray::Empty({1, 4200, 10}, DLDataType{kDLFloat, 32, 1}, dev);
  
   
  LOG(INFO) << "Copy bytes into input tensor.";
  TVMArrayCopyFromBytes(input_tensor, x, channel_offset*3);
  
  set_input("input", input_tensor);

  LOG(INFO) << "Run inference...";
  run();
  // get the output
  get_output(1, confs);
  get_output(0, locs);
  get_output(2, ldmks);
  LOG(INFO) << "Get output tensor...";


  std::vector<BBox> final_bboxes;

  int confs_idx = 0;
  int locs_idx = 0;
  int ldmks_idx = 0;
  std::cout << "anchors size:" << anchors.size() << std::endl;
  for (const Box& anchor: anchors) {
    BBox bbox;
    Box refined_box;
    float score = static_cast<float *>(confs->data)[confs_idx+1];
    float offsets[4];
    LOG(INFO) << locs_idx;
    offsets[0] = static_cast<float *>(locs->data)[locs_idx+0];
    offsets[1] = static_cast<float *>(locs->data)[locs_idx+1];
    offsets[2] = static_cast<float *>(locs->data)[locs_idx+2];
    offsets[3] = static_cast<float *>(locs->data)[locs_idx+3];

    if ( score > _score_threshold) {
            // score
            bbox.score = score;

            // bbox
            refined_box.cx = anchor.cx + offsets[0] * 0.1 * anchor.sx;
            refined_box.cy = anchor.cy + offsets[1] * 0.1 * anchor.sy;
            refined_box.sx = anchor.sx * exp(offsets[2] * 0.2);
            refined_box.sy = anchor.sy * exp(offsets[3] * 0.2);

            bbox.x1 = (refined_box.cx - refined_box.sx/2) * image.cols;
            bbox.y1 = (refined_box.cy - refined_box.sy/2) * image.rows;
            bbox.x2 = (refined_box.cx + refined_box.sx/2) * image.cols;
            bbox.y2 = (refined_box.cy + refined_box.sy/2) * image.rows;

            clip_bboxes(bbox, image.cols, image.rows);

            // landmark
             for (int i = 0; i < 5; i++) {
                bbox.landmarks[i].x = (anchor.cx + static_cast<float *>(ldmks->data)[2*i+ldmks_idx] * 0.1 * anchor.sx) * image.cols;
                bbox.landmarks[i].y = (anchor.cy + static_cast<float *>(ldmks->data)[2*i+1+ldmks_idx] * 0.1 * anchor.sy) * image.rows;
            }

            final_bboxes.push_back(bbox);

    }
    confs_idx += 2;
    locs_idx += 4;
    ldmks_idx += 10;
  }

  std::sort(final_bboxes.begin(), final_bboxes.end(), [](BBox &lsh, BBox &rsh) {
        return lsh.score > rsh.score;
    });
  nms(final_bboxes, _nms_threshold);
  


    float scale = 1.0f;
    std::cout << "total faces:" << final_bboxes.size() << std::endl;
    for (BBox& bbox : final_bboxes) {
        cv::putText(image, std::to_string(bbox.score), cv::Size(bbox.x1/scale - 5, bbox.y1/scale - 5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
    	cv::rectangle(image, cv::Point(bbox.x1/scale, bbox.y1/scale), cv::Point(bbox.x2/scale, bbox.y2/scale), cv::Scalar(255, 0, 0), 2);
    	for(int i = 0; i < 5; i++) {
    		cv::Point p(bbox.landmarks[i].x/scale, bbox.landmarks[i].y/scale);
    		cv::circle(image, p, 1, cv::Scalar(255, 0, 0), 4);
    	}
    }
    cv::imwrite("../result.jpg", image);
    cv::imshow("image", image);
    cv::waitKey(0);

}


int main(void) {
  DeployRetinaFaceExecutor();
  return 0;
}
