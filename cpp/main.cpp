
#include <opencv2/opencv.hpp>

#include "retinaface.hpp"
int main(void) {
    cv::Mat image = cv::imread("/home/vagrant/work/RetinaFace_TVM/test.jpg");
    cv::imshow("image", image);
    cv::waitKey(0);
    RetinaFace detector("/home/vagrant/work/RetinaFace_TVM/lib/retinaface_sim_opt.so");
    std::vector<BBox> final_bboxes;
    detector.detect(image, final_bboxes);
    std::cout << "Loaded model" << std::endl;

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
    cv::imwrite("result.jpg", image);
    cv::imshow("result", image);
    cv::waitKey(0);
    return 0;
}
