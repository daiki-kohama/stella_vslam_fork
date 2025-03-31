#ifndef STELLA_VSLAM_FEATURE_LIGHTGLUE_H
#define STELLA_VSLAM_FEATURE_LIGHTGLUE_H

#include "stella_vslam/type.h"

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <iostream>

namespace stella_vslam {
namespace feature {

std::vector<float> preprocess_for_extract(const cv::Mat& bgr_img);
std::vector<cv::Point2f> normalize_keypoints(const std::vector<cv::Point2f>& keypts, float width, float height);

class dl_extractor {
public:
    explicit dl_extractor(const std::string& model_path, const std::vector<std::vector<float>>& mask_rects = {},
                          const bool use_cuda = false);

    void run(std::vector<cv::_InputArray>& in_images,
             const cv::_InputArray& in_image_mask,
             std::vector<std::vector<cv::Point2f>>& imgs_keypts,
             std::vector<std::vector<std::vector<float>>>& imgs_descriptors,
             std::vector<std::unordered_set<unsigned int>>& imgs_valid_indices);

    //! A vector of keypoint area represents mask area
    //! Each areas are denoted as form of [x_min / cols, x_max / cols, y_min / rows, y_max / rows]
    std::vector<std::vector<float>> mask_rects_;

private:
    //! Create a mask matrix that constructed by rectangles
    void create_rectangle_mask(const unsigned int cols, const unsigned int rows);

    //! ONNX session for deep learning based feature extractor
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime::Extractor"};
    Ort::Session session_{nullptr};

    //! rectangle mask has been already initialized or not
    bool mask_is_initialized_ = false;
    cv::Mat rect_mask_;
};

class lg_matcher {
public:
    explicit lg_matcher(const std::string& model_path, const bool use_cuda = false);

    unsigned int run(std::vector<std::vector<cv::Point2f>>& imgs_keypts,
                     std::vector<std::vector<std::vector<float>>>& imgs_descriptors,
                     std::vector<std::unordered_set<unsigned int>>& imgs_valid_indices,
                     unsigned int width,
                     unsigned int height,
                     std::vector<std::pair<unsigned int, unsigned int>>& matches,
                     std::vector<float>& scores);

private:
    //! ONNX session for lightglue matcher
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime::Matcher"};
    Ort::Session session_{nullptr};
};

} // namespace feature
} // namespace stella_vslam

#endif // STELLA_VSLAM_FEATURE_LIGHTGLUE_H
