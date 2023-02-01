#include "retinaface.hpp"


RetinaFace::RetinaFace(const std::string& runtime_lib_path) {
    _mod_factory = tvm::runtime::Module::LoadFromFile(runtime_lib_path);
    _mod = _mod_factory.GetFunction("default")(_dev);
    set_input = _mod.GetFunction("set_input");
    get_output = _mod.GetFunction("get_output");
    run = _mod.GetFunction("run");
    if (run == nullptr) {
        std::cout << "Failed to load runtime library from path: " <<  runtime_lib_path << std::endl;
        exit(-1);
    }
    std::cout << "Successfully loaded runtime library from path: " <<  runtime_lib_path << std::endl;

    // allocate input tensor
    int64_t input_shape[4] = {1, 3, _in_h, _in_w};
    
    TVMArrayAlloc(input_shape, 4, kDLFloat, 32, 1, kDLCPU, 0, &_input_tensor); 

    // get output tensor
    // output tensor size should be: pow(320/8, 2)*2*(1+1/4+1/16) = 4200
    _confs_output = tvm::runtime::NDArray::Empty({1, 4200, 2}, DLDataType{kDLFloat, 32, 1}, _dev);
    _locs_output = tvm::runtime::NDArray::Empty({1, 4200, 4}, DLDataType{kDLFloat, 32, 1}, _dev);
    _ldmks_output = tvm::runtime::NDArray::Empty({1, 4200, 10}, DLDataType{kDLFloat, 32, 1}, _dev);
    
    this->create_anchors(this->anchors, _in_w, _in_h);
}

RetinaFace::~RetinaFace() {
    LOG(INFO) << "model released";
}

void RetinaFace::detect(const cv::Mat& image, std::vector<BBox>& final_bboxes) const {
    
    // preprocess image and set input tensor
    preprocess(image);

    set_input("input", _input_tensor);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    for (int i = 0; i < 100; i++) {
        run();
    }
    end = std::chrono::system_clock::now();
    double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOG(INFO) << "Elapsed time for running inference: " << elapsed_time; 
    
    
    // get the output
    get_output(0, _locs_output);
    get_output(1, _confs_output);
    get_output(2, _ldmks_output);
   
  
    int confs_idx = 0;
    int locs_idx = 0;
    int ldmks_idx = 0;
    for (const Box& anchor: anchors) {
        BBox bbox;
        Box refined_box;
        float score = static_cast<float *>(_confs_output->data)[confs_idx+1];
        float offsets[4];
        offsets[0] = static_cast<float *>(_locs_output->data)[locs_idx+0];
        offsets[1] = static_cast<float *>(_locs_output->data)[locs_idx+1];
        offsets[2] = static_cast<float *>(_locs_output->data)[locs_idx+2];
        offsets[3] = static_cast<float *>(_locs_output->data)[locs_idx+3];

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
                bbox.landmarks[i].x = (anchor.cx + static_cast<float *>(_ldmks_output->data)[2*i+ldmks_idx] * 0.1 * anchor.sx) * image.cols;
                bbox.landmarks[i].y = (anchor.cy + static_cast<float *>(_ldmks_output->data)[2*i+1+ldmks_idx] * 0.1 * anchor.sy) * image.rows;
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
}

void RetinaFace::preprocess(const cv::Mat& image) const {
    // TODO: considering reducing memory copy
    cv::Mat image_resized = cv::Mat::zeros(320, 320, CV_8UC3);
    cv::resize(image, image_resized, cv::Size(320, 320), 0, 0, cv::INTER_LINEAR);
    cv::Scalar mean_pixel(_mean_vals[0], _mean_vals[1], _mean_vals[2]);
    
    cv::Mat image_float;
    image_resized.convertTo(image_float, CV_32FC3);
    cv::Mat mean_mat = cv::Mat(image_float.rows, image_float.cols, CV_32FC3, mean_pixel); 
    cv::subtract(image_float, mean_mat, image_float);
    
    int channel_offset = _in_w*_in_h*sizeof(float);

    cv::Mat split_mat[3];
    cv::split(image_float, split_mat);
    float* x = (float *)malloc(3*_in_w*_in_h*sizeof(float));
    memcpy(x, split_mat[2].ptr<float>(), channel_offset);
    memcpy(x+channel_offset/sizeof(float), split_mat[1].ptr<float>(), channel_offset);
    memcpy(x+channel_offset/sizeof(float)*2, split_mat[0].ptr<float>(), channel_offset);
    TVMArrayCopyFromBytes(_input_tensor, x, channel_offset*3);
}

void RetinaFace::create_anchors(std::vector<Box>& anchors, int _w, int _h) {

    anchors.clear();
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
}

void RetinaFace::nms(std::vector<BBox>& bboxes, float nms_threshold) {
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
inline void RetinaFace::clip_bboxes(BBox& bbox, int w, int h) {
    if(bbox.x1 < 0) bbox.x1 = 0;
    if(bbox.y1 < 0) bbox.y1 = 0;
    if(bbox.x2 > w) bbox.x2 = w;
    if(bbox.y2 > h) bbox.y2 = h;
}
