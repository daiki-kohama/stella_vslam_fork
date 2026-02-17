#ifndef STELLA_VSLAM_FEATURE_ORB_EXTRACTOR_H
#define STELLA_VSLAM_FEATURE_ORB_EXTRACTOR_H

#include "stella_vslam/feature/orb_params.h"
#include "stella_vslam/feature/orb_impl.h"
#include "stella_vslam/feature/icosahedron_unwrapper.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#ifdef USE_CUDA_EFFICIENT_DESCRIPTORS
#include <cuda_efficient_descriptors.h>
#endif

namespace stella_vslam {
namespace feature {

enum class descriptor_type {
    ORB,
    HASH_SIFT
};

struct eqr_ico_point {
    cv::Point2f eqr_pt;
    cv::Point2f ico_pt;
    unsigned int face_idx;
};

inline descriptor_type descriptor_type_from_string(const std::string& desc_type_str) {
    if (desc_type_str == "ORB") {
        return descriptor_type::ORB;
    }
    else if (desc_type_str == "HASH_SIFT" || desc_type_str == "HashSIFT") {
        return descriptor_type::HASH_SIFT;
    }
    else {
        throw std::runtime_error("Invalid descriptor_type");
    }
}

inline std::string descriptor_type_to_string(descriptor_type desc_type) {
    if (desc_type == descriptor_type::ORB) {
        return "ORB";
    }
    else if (desc_type == descriptor_type::HASH_SIFT) {
        return "HashSIFT";
    }
    else {
        throw std::runtime_error("Invalid descriptor_type");
    }
}

class orb_extractor {
public:
    orb_extractor() = delete;

    //! Constructor
    orb_extractor(const orb_params* orb_params,
                  const unsigned int min_area,
                  const descriptor_type desc_type = descriptor_type::ORB,
                  const bool use_ico_image_pyramid = false,
                  const std::vector<std::vector<float>>& mask_rects = {});

    //! Destructor
    virtual ~orb_extractor() = default;

    //! Extract keypoints and each descriptor of them
    void extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                 std::vector<cv::KeyPoint>& keypts, const cv::_OutputArray& out_descriptors);

    //! parameters for ORB extraction
    const orb_params* orb_params_;

    //! A vector of keypoint area represents mask area
    //! Each areas are denoted as form of [x_min / cols, x_max / cols, y_min / rows, y_max / rows]
    std::vector<std::vector<float>> mask_rects_;

    //! Image pyramid
    std::vector<cv::Mat> image_pyramid_;

    //! Icosahedron image pyramid
    std::vector<std::vector<cv::Mat>> ico_image_pyramid_;

private:
    //! Calculate scale factors and sigmas
    void calc_scale_factors();

    //! Create a mask matrix that constructed by rectangles
    void create_rectangle_mask(const unsigned int cols, const unsigned int rows);

    //! Compute image pyramid
    void compute_image_pyramid(const cv::Mat& image);

    //! Compute fast keypoints for cells in each image pyramid
    void compute_fast_keypoints(std::vector<std::vector<cv::KeyPoint>>& all_keypts, const cv::Mat& mask) const;

    //! Pick computed keypoints on the image uniformly
    std::vector<cv::KeyPoint> distribute_keypoints(const std::vector<cv::KeyPoint>& keypts_to_distribute,
                                                   std::vector<unsigned int>& distributed_keypts_indices,
                                                   const int min_x, const int max_x, const int min_y, const int max_y,
                                                   const float scale_factor) const;

    //! Compute orientation for each keypoint
    void compute_orientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypts) const;

    //! Correct keypoint's position to comply with the scale
    void correct_keypoint_scale(std::vector<cv::KeyPoint>& keypts_at_level, const unsigned int level) const;

    //! Compute the gradient direction of pixel intensity in a circle around the point
    float ic_angle(const cv::Mat& image, const cv::Point2f& point) const;

    //! Compute orb descriptor of a keypoint
    void compute_orb_descriptor(const cv::KeyPoint& keypt, const cv::Mat& image, uchar* desc) const;

    //! Initialize icosahedron unwrappers for each level
    void initialize_ico_unwrappers(const unsigned int cols, const unsigned int rows);

    //! Create icosahedron masks from equirectangular mask
    std::vector<cv::Mat> create_ico_masks(const cv::Mat& mask);

    //! Compute fast keypoints for icosahedron images for cells in each image pyramid
    void compute_fast_keypoints_ico(std::vector<std::vector<cv::KeyPoint>>& all_keypts, std::vector<std::vector<eqr_ico_point>>& ico_pts,
                                    const std::vector<cv::Mat>& ico_masks, const unsigned int cols, const unsigned int rows) const;
    
    //! Compute orientation for each keypoint on icosahedron images
    void compute_ico_orientation(const std::vector<cv::Mat>& ico_images, std::vector<cv::KeyPoint>& keypts,
                                 const std::vector<eqr_ico_point>& eqr_ico_pts) const;

    //! Area of node occupied by one feature point
    unsigned int min_area_sqrt_;

    //! size of maximum ORB patch radius
    static constexpr unsigned int orb_patch_radius_ = 19;

    //! rectangle mask has been already initialized or not
    bool mask_is_initialized_ = false;
    cv::Mat rect_mask_;

    descriptor_type desc_type_;

    //! flag for using icosahedron image pyramid
    bool use_ico_image_pyramid_;

    //! Icosahedron unwrappers for icosahedron image pyramid
    std::vector<eqr_ico::IcosahedronUnwrapper> ico_unwrappers_;

    //! icosahedron masks
    std::vector<cv::Mat> ico_original_masks_;

    //! flag for checking whether ico_unwrappers_ is initialized or not
    bool is_ico_unwrapper_initialized_ = false;

    //! rectangle mask for icosahedron images
    std::vector<cv::Mat> rect_ico_masks_;

    //! feature descriptor implementations
    orb_impl orb_impl_;
#ifdef USE_CUDA_EFFICIENT_DESCRIPTORS
    cv::Ptr<cv::cuda::HashSIFT> hash_sift_;
#endif
};

} // namespace feature
} // namespace stella_vslam

#endif // STELLA_VSLAM_FEATURE_ORB_EXTRACTOR_H
