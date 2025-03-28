#include "stella_vslam/feature/lightglue.h"
#include "stella_vslam/type.h"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

#include <spdlog/spdlog.h>

namespace stella_vslam {
namespace feature {

lightglue::lightglue(const std::vector<std::vector<float>>& mask_rects)
    : mask_rects_(mask_rects) {
    spdlog::debug("CONSTRUCT: feature::lightglue");

    if (!Py_IsInitialized()) {
        interpreter_ = new pybind11::scoped_interpreter(); // インタープリタを初期化
        spdlog::debug("Python Interpreter Initialized");
    }

    try {
        pybind11::module sys = pybind11::module::import("sys");
        spdlog::debug("Python version: {}", std::string(pybind11::str(sys.attr("version"))));

        std::string script_dir = "/LightGlue/cpp_bind";                            // Python スクリプトがあるディレクトリ
        std::string package_dir = "/LightGlue/venv/lib/python3.10/site-packages/"; // Python パッケージがあるディレクトリ
        sys.attr("path").attr("append")(script_dir);                               // Python のパスにディレクトリを追加
        sys.attr("path").attr("append")(package_dir);                              // Python のパスにディレクトリを追加
        lightglue_lib_ = pybind11::module::import("lightglue_lib");
    }
    catch (const pybind11::error_already_set& e) {
        throw std::runtime_error(e.what());
    }
}

lightglue::~lightglue() {
    spdlog::debug("DESTRUCT: feature::lightglue");
    if (interpreter_) {
        if (lightglue_lib_) {
            lightglue_lib_.release();
        }
        delete interpreter_;
        spdlog::debug("Python Interpreter Released");
    }
}

// cv::Mat を NumPy 配列に変換する関数
pybind11::array_t<unsigned char> lightglue::mat_to_numpy(const cv::Mat& mat) const {
    if (mat.empty()) {
        throw std::runtime_error("Empty image");
    }

    // 高さ、幅、チャンネル数を取得
    int height = mat.rows;
    int width = mat.cols;
    int channels = mat.channels();

    // NumPy の形状を定義
    std::vector<ssize_t> shape = {height, width, channels};
    std::vector<ssize_t> strides = {static_cast<ssize_t>(mat.step[0]),
                                    static_cast<ssize_t>(mat.step[1]),
                                    static_cast<ssize_t>(1)};

    // NumPy 配列を作成し、データをコピー
    return pybind11::array_t<unsigned char>(
        shape,   // 形状
        strides, // ストライド
        mat.data // データ
    );
}

template<typename T>
std::vector<T> lightglue::numpyToVector1D(pybind11::array_t<T> np_array) const {
    // NumPy配列の情報を取得
    auto buf = np_array.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input array must be 1D");
    }

    size_t rows = buf.shape[0];

    // std::vectorを作成
    std::vector<T> vec(rows);

    // NumPyのデータポインタを取得
    T* ptr = static_cast<T*>(buf.ptr);

    // 1次元配列としてコピー
    for (size_t i = 0; i < rows; ++i) {
        vec[i] = ptr[i]; // フラットな配列から値を取得
    }

    return vec;
}

template<typename T>
std::vector<std::vector<T>> lightglue::numpyToVector2D(pybind11::array_t<T> np_array) const {
    // NumPy配列の情報を取得
    auto buf = np_array.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Input array must be 2D");
    }

    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];

    // std::vectorを作成
    std::vector<std::vector<T>> vec(rows, std::vector<T>(cols));

    // NumPyのデータポインタを取得
    T* ptr = static_cast<T*>(buf.ptr);

    // 2次元配列としてコピー
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            vec[i][j] = ptr[i * cols + j]; // フラットな配列から値を取得
        }
    }

    return vec;
}

// 2D std::vector を 2D NumPy 配列に変換
template<typename T>
pybind11::array_t<T> lightglue::vectorToNumpy2D(const std::vector<std::vector<T>>& vec) const {
    if (vec.empty()) {
        throw std::runtime_error("Input vector is empty.");
    }

    size_t rows = vec.size();
    size_t cols = vec[0].size();

    // フラットな 1D 配列を作成
    std::vector<T> data(rows * cols);

    // 2D std::vector を 1D 配列に変換
    for (size_t i = 0; i < rows; ++i) {
        if (vec[i].size() != cols) {
            throw std::runtime_error("All rows must have the same number of columns.");
        }
        std::copy(vec[i].begin(), vec[i].end(), data.begin() + i * cols);
    }

    // NumPy 配列を作成
    return pybind11::array_t<T>(
        {static_cast<ssize_t>(rows), static_cast<ssize_t>(cols)}, // shape
        {sizeof(T) * cols, sizeof(T)},                            // strides
        data.data()                                               // データ
    );
}

void lightglue::feature_extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                                std::vector<cv::Point2f>& keypts, std::vector<std::vector<double>>& descripts) {
    if (in_image.empty()) {
        return;
    }

    const auto image = in_image.getMat();

    // mask initialization
    if (!mask_is_initialized_ && !mask_rects_.empty()) {
        create_rectangle_mask(image.cols, image.rows);
        mask_is_initialized_ = true;
    }

    try {
        assert(lightglue_lib_);

        pybind11::array_t<unsigned char> image_np = mat_to_numpy(image); // 画像を NumPy 配列に変換

        pybind11::tuple result = lightglue_lib_.attr("feature_extract")(image_np); // Python の関数を呼び出し

        if (result.size() != 2) {
            throw std::runtime_error("Expected 2 return values from Python function.");
        }

        pybind11::array_t<double> keypoints_np = result[0].cast<pybind11::array_t<double>>();
        pybind11::array_t<double> descriptors_np = result[1].cast<pybind11::array_t<double>>();

        // std::vector<std::vector<double>> に変換
        auto keypoints = numpyToVector2D(keypoints_np);
        auto descriptors = numpyToVector2D(descriptors_np);

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

        std::vector<cv::Point2f> not_in_mask_keypts;
        std::vector<std::vector<double>> not_in_mask_descripts;
        for (unsigned int i = 0; i < keypoints.size(); ++i) {
            const auto& keypoint = keypoints[i];
            const auto& descriptor = descriptors[i];
            if (is_in_mask(static_cast<unsigned int>(keypoint[1]), static_cast<unsigned int>(keypoint[0]))) {
                continue;
            }
            not_in_mask_keypts.emplace_back(cv::Point2f(keypoint[0], keypoint[1]));
            not_in_mask_descripts.emplace_back(descriptor);
        }

        keypts = not_in_mask_keypts;
        descripts = not_in_mask_descripts;
    }
    catch (const pybind11::error_already_set& e) {
        throw std::runtime_error(e.what());
    }
}

void lightglue::image_match(const cv::_InputArray& in_image0, const cv::_InputArray& in_image1,
                            const std::vector<cv::Point2f>& keypts1, const std::vector<cv::Point2f>& keypts2,
                            const std::vector<std::vector<double>>& desripts1, const std::vector<std::vector<double>>& desripts2,
                            std::vector<std::pair<unsigned int, unsigned int>>& matched_idx_pairs, std::vector<double>& matched_scores) const {
    std::cout << "IN feature::lightglue::image_match" << std::endl;
    try {
        assert(lightglue_lib_);

        pybind11::array_t<unsigned char> image0 = mat_to_numpy(in_image0.getMat()); // 画像を NumPy 配列に変換
        pybind11::array_t<unsigned char> image1 = mat_to_numpy(in_image1.getMat()); // 画像を NumPy 配列に変換

        // std::vector<cv::Point2f> を 2D NumPy 配列に変換
        std::vector<std::vector<double>> keypts1_vec;
        std::vector<std::vector<double>> keypts2_vec;
        keypts1_vec.reserve(keypts1.size());
        for (const auto& keypt : keypts1) {
            keypts1_vec.emplace_back(std::vector<double>{keypt.x, keypt.y});
        }
        keypts2_vec.reserve(keypts2.size());
        for (const auto& keypt : keypts2) {
            keypts2_vec.emplace_back(std::vector<double>{keypt.x, keypt.y});
        }
        pybind11::array_t<double> keypts1_np = vectorToNumpy2D(keypts1_vec);
        pybind11::array_t<double> keypts2_np = vectorToNumpy2D(keypts2_vec);

        // std::vector<std::vector<double>> を 2D NumPy 配列に変換
        pybind11::array_t<double> desripts1_np = vectorToNumpy2D(desripts1);
        pybind11::array_t<double> desripts2_np = vectorToNumpy2D(desripts2);

        pybind11::tuple result = lightglue_lib_.attr("image_match")(image0, image1, keypts1_np, keypts2_np, desripts1_np, desripts2_np); // Python の関数を呼び出し

        if (result.size() != 2) {
            throw std::runtime_error("Expected 2 return values from Python function.");
        }

        pybind11::array_t<double> matches_np = result[0].cast<pybind11::array_t<double>>();
        pybind11::array_t<double> scores_np = result[1].cast<pybind11::array_t<double>>();

        // std::vector<std::vector<double>> に変換
        auto matches = numpyToVector2D(matches_np);
        matched_scores = numpyToVector1D(scores_np);

        // std::vector<std::pair<unsigned int, unsigned int>> に変換
        matched_idx_pairs.clear();
        matched_idx_pairs.reserve(matches.size());
        for (const auto& match : matches) {
            matched_idx_pairs.emplace_back(std::make_pair(static_cast<unsigned int>(match[0]), static_cast<unsigned int>(match[1])));
        }
    }
    catch (const pybind11::error_already_set& e) {
        throw std::runtime_error(e.what());
    }
}

void lightglue::create_rectangle_mask(const unsigned int cols, const unsigned int rows) {
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

} // namespace feature
} // namespace stella_vslam
