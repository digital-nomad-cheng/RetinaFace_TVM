#ifndef RETINAFACE_H
#define RETINAFACE_H

#include <string>

#include <opencv2/opencv.hpp>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>


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

class RetinaFace {
public:
    RetinaFace(const std::string& runtime_lib_path);
    ~RetinaFace();

    void detect(const cv::Mat& image, std::vector<BBox>& final_bboxes) const;

private:
    static void create_anchors(std::vector<Box>& anchors, int w, int h);
    static void nms(std::vector<BBox>& input_bboxes, float nms_threshold=0.5);
    static void clip_bboxes(BBox& bbox, int w, int h);

    void preprocess(const cv::Mat& image) const;

    float _nms_threshold = 0.4;
    float _score_threshold = 0.6;
    
    const float _mean_vals[3] = {104.f, 117.f, 123.f};
    const int _in_w = 320;
    const int _in_h = 320;

    // module function
    DLDevice _dev{kDLCPU, 0};
    tvm::runtime::Module _mod_factory;
    tvm::runtime::Module _mod;
    tvm::runtime::PackedFunc set_input;
    tvm::runtime::PackedFunc get_output;
    tvm::runtime::PackedFunc run;

    // define input and output tensor
    DLTensor* _input_tensor;
    tvm::runtime::NDArray  _locs_output;
    tvm::runtime::NDArray  _confs_output; 
    tvm::runtime::NDArray  _ldmks_output;

    std::vector<Box> anchors;
};
#endif
