#ifndef STELLA_VSLAM_FEATURE_LIGHTGLUE_H
#define STELLA_VSLAM_FEATURE_LIGHTGLUE_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <iostream>

namespace stella_vslam {
namespace feature {

class lightglue {
public:
    lightglue() = delete;

    //! Constructor
    lightglue(const std::vector<std::vector<float>>& mask_rects = {});

    //! Destructor
    ~lightglue();

    //! Extract features from the image
    void feature_extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                         std::vector<cv::Point2f>& keypts, std::vector<std::vector<double>>& descripts);

    //! Match features between two images
    void image_match(const cv::_InputArray& in_image0, const cv::_InputArray& in_image1,
                     const std::vector<cv::Point2f>& keypts1, const std::vector<cv::Point2f>& keypts2,
                     const std::vector<std::vector<double>>& desripts1, const std::vector<std::vector<double>>& desripts2,
                     std::vector<std::pair<unsigned int, unsigned int>>& matched_idx_pairs, std::vector<double>& matched_scores) const;

    //! A vector of keypoint area represents mask area
    //! Each areas are denoted as form of [x_min / cols, x_max / cols, y_min / rows, y_max / rows]
    std::vector<std::vector<float>> mask_rects_;

private:
    template<typename T>
    //! Convert 1D NumPy array to 1D std::vector
    std::vector<T> numpyToVector1D(pybind11::array_t<T> np_array) const;
    template<typename T>
    //! Convert 2D NumPy array to std::vector
    std::vector<std::vector<T>> numpyToVector2D(pybind11::array_t<T> np_array) const;

    //! Convert cv::Mat to NumPy array
    pybind11::array_t<unsigned char> mat_to_numpy(const cv::Mat& mat) const;

    template<typename T>
    //! Convert 2D std::vector to 2D NumPy array
    pybind11::array_t<T> vectorToNumpy2D(const std::vector<std::vector<T>>& vec) const;

    //! Create rectangle mask
    void create_rectangle_mask(const unsigned int cols, const unsigned int rows);

    //! rectangle mask has been already initialized or not
    bool mask_is_initialized_ = false;
    cv::Mat rect_mask_;

    //! Python library for LightGlue
    pybind11::module lightglue_lib_;

    //! Python interpreter
    pybind11::scoped_interpreter* interpreter_ = nullptr;
};

} // namespace feature
} // namespace stella_vslam

#endif // STELLA_VSLAM_FEATURE_LIGHTGLUE_H
