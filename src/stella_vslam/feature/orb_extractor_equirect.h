#ifndef STELLA_VSLAM_FEATURE_ORB_EXTRACTOREQUIRECT_H
#define STELLA_VSLAM_FEATURE_ORB_EXTRACTOREQUIRECT_H

#include "stella_vslam/feature/orb_extractor.h"
#include "stella_vslam/type.h"
#include "stella_vslam/camera/equirectangular.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace camera {
class base;
} // namespace camera

namespace stella_vslam {
namespace feature {

class orb_extractor_equirect final : public orb_extractor {
public:
    //! Constructor
    orb_extractor_equirect(const orb_params* orb_params,
                           const unsigned int min_area,
                           camera::base* camera,
                           const descriptor_type desc_type = descriptor_type::ORB,
                           const std::vector<std::vector<float>>& mask_rects = {},
                           const unsigned int division = 4);

    //! Destructor
    virtual ~orb_extractor_equirect() = default;

    //! Extract keypoints and each descriptor of them
    void extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                 std::vector<cv::KeyPoint>& keypts, const cv::_OutputArray& out_descriptors) override;

    //! Compute map for equirectangular image rectification against the x-axis rotation
    void compute_map_angle_x(const unsigned int cols, const unsigned int rows, double angle_x, cv::Mat& map_x, cv::Mat& map_y) const;

    //! Convert points against the x-axis rotation
    std::vector<cv::Point> convert_rectified_points(std::vector<cv::Point> rectified_points,
                                                    const unsigned int cols,
                                                    const unsigned int rows,
                                                    const double angle_x) const;

    //! Rectified equirectangular image pyramid
    std::vector<std::vector<cv::Mat>> rectified_image_pyramid_;

private:
    //! Compute maps for equirectangular image rectification in each pyramid level
    void compute_rectification_map_pyramid(const unsigned int cols,
                                           const unsigned int rows,
                                           const unsigned int division,
                                           const unsigned int num_levels,
                                           std::vector<std::vector<cv::Mat>>& maps_x_pyramid,
                                           std::vector<std::vector<cv::Mat>>& maps_y_pyramid);

    //! Compute maps for equirectangular image rectification
    void compute_rectification_maps(const unsigned int cols,
                                    const unsigned int rows,
                                    const unsigned int division,
                                    std::vector<cv::Mat>& maps_x,
                                    std::vector<cv::Mat>& maps_y);

    //! Compute images for equirectangular image rectification in each pyramid level
    void compute_rectified_image_pyramid(const cv::Mat& image);

    //! Compute fast keypoints for rectified images in each image pyramid
    void compute_fast_keypoints(std::vector<std::vector<cv::KeyPoint>>& all_keypts, const cv::Mat& mask) const;

    //! Compute mask for equirectangular image rectification
    void compute_rectification_mask();

    //! Compute equirectangular curve dots
    std::vector<cv::Point> compute_equirect_curve_dots(const unsigned int cols,
                                                       const unsigned int rows,
                                                       const double angle,
                                                       const unsigned int begin_col,
                                                       const unsigned int end_col);

    //! Convert xyz to equirectangular
    cv::Point xyz_to_equirectangular(double x, double y, double z, int width, int height);

    //! Compute orientation for each keypoint
    void compute_orientation(const std::vector<cv::Mat>& image, std::vector<cv::KeyPoint>& keypts) const;

    //! camera model of equirectangular
    camera::base* camera_;

    //! division number of the equirectangular image
    unsigned int division_;

    //! angles of the equirectangular image rectification
    std::vector<double> rectification_angles_;

    //! maps for equirectangular image rectification in each pyramid level (x)
    std::vector<std::vector<cv::Mat>> maps_x_pyramid_;
    //! maps for equirectangular image rectification in each pyramid level (y)
    std::vector<std::vector<cv::Mat>> maps_y_pyramid_;

    //! mask for equirectangular image rectification
    std::vector<cv::Mat> rectification_mask_pyramid_;
};

} // namespace feature
} // namespace stella_vslam

#endif // STELLA_VSLAM_FEATURE_ORB_EXTRACTOREQUIRECT_H