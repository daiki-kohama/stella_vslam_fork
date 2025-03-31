#include "stella_vslam/feature/lightglue.h"
#include "stella_vslam/type.h"

#include <iostream>

#include <spdlog/spdlog.h>

namespace stella_vslam {
namespace feature {

// Convert BGR image to normalized grayscale tensor (1x1xHxW)
std::vector<float> preprocess_for_extract(const cv::Mat& bgr_img) {
    cv::Mat rgb_img;
    cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
    rgb_img.convertTo(rgb_img, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> rgb_channels(3);
    cv::split(rgb_img, rgb_channels);
    cv::Mat gray = 0.299 * rgb_channels[0] + 0.587 * rgb_channels[1] + 0.114 * rgb_channels[2];

    int height = gray.rows;
    int width = gray.cols;
    std::vector<float> input_tensor(1 * 1 * height * width);
    std::memcpy(input_tensor.data(), gray.data, height * width * sizeof(float));

    return input_tensor;
}

std::vector<cv::Point2f> normalize_keypoints(const std::vector<cv::Point2f>& keypts, float width, float height) {
    std::vector<cv::Point2f> normalized;
    normalized.reserve(keypts.size());

    for (const auto& kp : keypts) {
        float x = 2.0f * kp.x / width - 1.0f;
        float y = 2.0f * kp.y / height - 1.0f;
        normalized.emplace_back(x, y);
    }

    return normalized;
}

dl_extractor::dl_extractor(const std::string& model_path, const std::vector<std::vector<float>>& mask_rects, const bool use_cuda)
    : mask_rects_(mask_rects) {
    Ort::SessionOptions session_options;
#ifdef USE_ONNX_RUNTIME_CUDA
    if (use_cuda) {
        OrtCUDAProviderOptions cuda_options;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }
#endif
    session_ = Ort::Session(env_, model_path, session_options);
}

void dl_extractor::run(std::vector<cv::_InputArray>& in_images,
                       const cv::_InputArray& in_image_mask,
                       std::vector<std::vector<cv::Point2f>>& imgs_keypts,
                       std::vector<std::vector<std::vector<float>>>& imgs_descriptors,
                       std::vector<std::unordered_set<unsigned int>>& imgs_valid_indices) {
    // preprocess images

    std::vector<cv::Mat> images;
    for (const auto& in_image : in_images) {
        cv::Mat image = in_image.getMat();
        images.push_back(image);
    }

    const unsigned int width = images[0].cols;
    const unsigned int height = images[0].rows;
    const unsigned int image_num = images.size();

    std::vector<float> input_tensor_values;
    for (const auto& img : images) {
        std::vector<float> preprocessed = preprocess_for_extract(img);
        input_tensor_values.insert(input_tensor_values.end(), preprocessed.begin(), preprocessed.end());
    }

    std::vector<int64_t> input_shape = {image_num, 1, height, width};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // run the model

    const char* input_names[] = {"images"};
    const char* output_names[] = {"top_keypoints", "top_scores", "top_descriptors"};
    auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 3);

    // convert keypoints to std::vector<std::vector<cv::Point2f>>

    Ort::Value& keypts_tensor = output_tensors[0];
    int16_t* keypts_data = keypts_tensor.GetTensorMutableData<int16_t>();
    auto keypts_shape = keypts_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const unsigned int num_keypts = keypts_shape[1];

    imgs_keypts.resize(image_num);
    for (int i = 0; i < image_num; ++i) {
        const unsigned int offset_i = i * num_keypts * 2;
        imgs_keypts[i].resize(num_keypts);
        for (int j = 0; j < num_keypts; ++j) {
            const unsigned int offset_j = j * 2;
            float x = static_cast<float>(keypts_data[offset_i + offset_j]);
            float y = static_cast<float>(keypts_data[offset_i + offset_j + 1]);
            imgs_keypts[i][j] = cv::Point2f(x, y);
        }
    }

    // convert descriptors to std::vector<std::vector<std::vector<float>>>

    Ort::Value& descriptors_tensor = output_tensors[2];
    float* descriptors_data = descriptors_tensor.GetTensorMutableData<float>();
    auto descriptors_shape = descriptors_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const unsigned int descriptor_size = descriptors_shape[2];

    imgs_descriptors.resize(image_num);
    for (int i = 0; i < image_num; ++i) {
        imgs_descriptors[i].resize(num_keypts);
        const unsigned int offset_i = i * num_keypts * descriptor_size;
        for (int j = 0; j < num_keypts; ++j) {
            imgs_descriptors[i][j].resize(descriptor_size);
            const unsigned int offset_j = j * descriptor_size;
            std::copy(
                descriptors_data + offset_i + offset_j,
                descriptors_data + offset_i + offset_j + descriptor_size,
                imgs_descriptors[i][j].begin());
        }
    }

    // remove keypoints in the mask

    // mask initialization
    if (!mask_is_initialized_ && !mask_rects_.empty()) {
        create_rectangle_mask(width, height);
        mask_is_initialized_ = true;
    }

    cv::Mat mask;
    if (!in_image_mask.empty()) {
        mask = in_image_mask.getMat();
    }
    else if (!rect_mask_.empty()) {
        mask = rect_mask_;
    }
    else {
        mask = cv::Mat();
    }

    auto is_in_mask = [&mask](const unsigned int y, const unsigned int x) {
        return mask.at<unsigned char>(y, x) == 0;
    };

    imgs_valid_indices.resize(image_num);
    for (int i = 0; i < image_num; ++i) {
        for (int j = 0; j < num_keypts; ++j) {
            const auto& keypt = imgs_keypts[i][j];
            if (is_in_mask(static_cast<unsigned int>(keypt.y), static_cast<unsigned int>(keypt.x))) {
                continue;
            }
            imgs_valid_indices[i].insert(j);
        }
    }
}

void dl_extractor::create_rectangle_mask(const unsigned int cols, const unsigned int rows) {
    if (rect_mask_.empty()) {
        rect_mask_ = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(255));
    }
    // draw masks
    for (const auto& mask_rect : mask_rects_) {
        // draw black rectangle
        const unsigned int x_min = std::round(cols * mask_rect.at(0));
        const unsigned int x_max = std::round(cols * mask_rect.at(1));
        const unsigned int y_min = std::round(rows * mask_rect.at(2));
        const unsigned int y_max = std::round(rows * mask_rect.at(3));
        cv::rectangle(rect_mask_, cv::Point2i(x_min, y_min), cv::Point2i(x_max, y_max), cv::Scalar(0), -1, cv::LINE_AA);
    }
}

lg_matcher::lg_matcher(const std::string& model_path, const bool use_cuda) {
    Ort::SessionOptions session_options;
#ifdef USE_ONNX_RUNTIME_CUDA
    if (use_cuda) {
        OrtCUDAProviderOptions cuda_options;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }
#endif
    session_ = Ort::Session(env_, model_path, session_options);
}

unsigned int lg_matcher::run(std::vector<std::vector<cv::Point2f>>& imgs_keypts,
                             std::vector<std::vector<std::vector<float>>>& imgs_descriptors,
                             std::vector<std::unordered_set<unsigned int>>& imgs_valid_indices,
                             unsigned int width,
                             unsigned int height,
                             std::vector<std::pair<unsigned int, unsigned int>>& matches,
                             std::vector<float>& scores) {
    assert(imgs_keypts.size() == 2);
    assert(imgs_descriptors.size() == 2);
    assert(imgs_valid_indices.size() == 2);
    assert(imgs_keypts[0].size() == imgs_descriptors[0].size());
    assert(imgs_keypts[1].size() == imgs_descriptors[1].size());
    assert(imgs_keypts[0].size() == imgs_valid_indices[0].size());
    assert(imgs_keypts[1].size() == imgs_valid_indices[1].size());
    assert(imgs_keypts[0].size() == imgs_keypts[1].size());
    assert(imgs_keypts[0].size() == 1024);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // preprocess keypoints

    std::vector<std::vector<cv::Point2f>> normalized_imgs_keypts;
    normalized_imgs_keypts.reserve(imgs_keypts.size());
    for (const auto& keypts : imgs_keypts) {
        normalized_imgs_keypts.emplace_back(normalize_keypoints(keypts, width, height));
    }

    std::vector<float> normalized_imgs_keypts_values;
    for (const auto& keypts : normalized_imgs_keypts) {
        for (const auto& keypt : keypts) {
            normalized_imgs_keypts_values.push_back(keypt.x);
            normalized_imgs_keypts_values.push_back(keypt.y);
        }
    }

    std::vector<int64_t> normalized_imgs_keypts_shape = {
        static_cast<int64_t>(normalized_imgs_keypts.size()),
        static_cast<int64_t>(normalized_imgs_keypts[0].size()),
        2};

    Ort::Value normalized_imgs_keypts_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        normalized_imgs_keypts_values.data(),
        normalized_imgs_keypts_values.size(),
        normalized_imgs_keypts_shape.data(),
        normalized_imgs_keypts_shape.size());

    // preprocess descriptors

    std::vector<float> imgs_descriptors_values;
    imgs_descriptors_values.reserve(imgs_descriptors.size() * imgs_descriptors[0].size() * imgs_descriptors[0][0].size());
    for (const auto& descriptors : imgs_descriptors) {
        for (const auto& descriptor : descriptors) {
            imgs_descriptors_values.insert(imgs_descriptors_values.end(), descriptor.begin(), descriptor.end());
        }
    }

    std::vector<int64_t> imgs_descriptors_shape = {
        static_cast<int64_t>(imgs_descriptors.size()),
        static_cast<int64_t>(imgs_descriptors[0].size()),
        static_cast<int64_t>(imgs_descriptors[0][0].size())};

    Ort::Value imgs_descriptors_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        imgs_descriptors_values.data(),
        imgs_descriptors_values.size(),
        imgs_descriptors_shape.data(),
        imgs_descriptors_shape.size());

    std::array<Ort::Value, 2> input_tensors = {
        std::move(normalized_imgs_keypts_tensor),
        std::move(imgs_descriptors_tensor)};

    // run the model

    const char* input_names[] = {"keypoints", "descriptors"};
    const char* output_names[] = {"matches", "mscores"};

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names, input_tensors.data(), 2,
        output_names, 2);

    // convert matches to std::vector<std::pair<unsigned int, unsigned int>>
    // convert scores to std::vector<float>

    Ort::Value& matches_tensor = output_tensors[0];
    int16_t* matches_data = matches_tensor.GetTensorMutableData<int16_t>();
    auto matches_shape = matches_tensor.GetTensorTypeAndShapeInfo().GetShape();

    Ort::Value& scores_tensor = output_tensors[1];
    float* scores_data = scores_tensor.GetTensorMutableData<float>();

    const unsigned int match_num = matches_shape[0];

    unsigned int valid_match_num = 0;
    for (int i = 0; i < match_num; ++i) {
        const unsigned int offset_i = i * 3;
        int idx_1 = static_cast<unsigned int>(matches_data[offset_i + 1]);
        int idx_2 = static_cast<unsigned int>(matches_data[offset_i + 2]);
        if (imgs_valid_indices[0].count(idx_1) == 0 || imgs_valid_indices[1].count(idx_2) == 0) {
            continue;
        }
        matches.emplace_back(idx_1, idx_2);
        scores.emplace_back(scores_data[i]);
        ++valid_match_num;
    }

    return valid_match_num;
}

} // namespace feature
} // namespace stella_vslam
