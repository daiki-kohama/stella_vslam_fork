#include "stella_vslam/feature/orb_extractor_equirect.h"
#include "stella_vslam/type.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>

// #include <opencv2/opencv.hpp>

#include <iostream>

#include <spdlog/spdlog.h>

namespace stella_vslam {
namespace feature {

orb_extractor_equirect::orb_extractor_equirect(const orb_params* orb_params,
                                               const unsigned int min_area,
                                               camera::base* camera,
                                               const descriptor_type desc_type,
                                               const std::vector<std::vector<float>>& mask_rects,
                                               const unsigned int division)
    : orb_extractor(orb_params, min_area, desc_type, mask_rects), camera_(camera), division_(division) {
    assert(camera_->model_type_ == camera::model_type_t::Equirectangular);
    rectified_image_pyramid_.resize(orb_params_->num_levels_);
    for (unsigned int level = 0; level < orb_params_->num_levels_; ++level) {
        rectified_image_pyramid_.at(level).resize(division);
    }
    maps_x_pyramid_.resize(orb_params_->num_levels_);
    maps_y_pyramid_.resize(orb_params_->num_levels_);
    compute_rectification_map_pyramid(camera_->cols_, camera_->rows_, division, orb_params_->num_levels_, maps_x_pyramid_, maps_y_pyramid_);
    compute_rectification_mask();
    spdlog::debug("CONSTRUCT: orb_extractor_equirect");
}

void orb_extractor_equirect::compute_map_angle_x(const unsigned int cols,
                                                 const unsigned int rows,
                                                 double angle_x,
                                                 cv::Mat& map_x,
                                                 cv::Mat& map_y) const {
    // make grid
    double lon_unit = 2 * M_PI / cols;
    double lat_unit = M_PI / rows;
    Eigen::RowVectorXd lon = Eigen::VectorXd::LinSpaced(cols, -M_PI, M_PI - lon_unit);
    Eigen::VectorXd lat = Eigen::VectorXd::LinSpaced(rows, -M_PI / 2.0, M_PI / 2.0 - lat_unit);
    Eigen::MatrixXd lon_grid = lon.replicate(rows, 1);
    Eigen::MatrixXd lat_grid = lat.replicate(1, cols);

    // convert to cartesian
    Eigen::MatrixXd sph_x = lat_grid.array().cos() * lon_grid.array().sin();
    Eigen::MatrixXd sph_y = lat_grid.array().sin();
    Eigen::MatrixXd sph_z = lat_grid.array().cos() * lon_grid.array().cos();
    Eigen::MatrixXd sph(cols * rows, 3);
    sph.col(0) = Eigen::Map<Eigen::VectorXd>(sph_x.data(), rows * cols, 1);
    sph.col(1) = Eigen::Map<Eigen::VectorXd>(sph_y.data(), rows * cols, 1);
    sph.col(2) = Eigen::Map<Eigen::VectorXd>(sph_z.data(), rows * cols, 1);

    // rotatate around x-axis
    Eigen::Matrix3d R = Eigen::AngleAxisd(angle_x, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::MatrixXd rotated_sph = (R * sph.transpose()).transpose();

    // convert to spherical
    Eigen::VectorXd new_lon = rotated_sph.col(0).array().binaryExpr(
        rotated_sph.col(2).array(), [](double y, double x) { return std::atan2(y, x); });
    Eigen::VectorXd new_lat = rotated_sph.col(1).array().asin();

    // convert to image coordinate
    Eigen::VectorXd new_x = (new_lon.array() / (2.0 * M_PI)) * cols + cols / 2.0;
    Eigen::VectorXd new_y = (new_lat.array() / M_PI) * rows + rows / 2.0;
    Eigen::MatrixXd map_x_eigen = Eigen::Map<Eigen::MatrixXd>(new_x.data(), rows, cols);
    Eigen::MatrixXd map_y_eigen = Eigen::Map<Eigen::MatrixXd>(new_y.data(), rows, cols);

    // convert to cv::Mat
    cv::eigen2cv(map_x_eigen, map_x);
    cv::eigen2cv(map_y_eigen, map_y);
    map_x.convertTo(map_x, CV_32F);
    map_y.convertTo(map_y, CV_32F);
}

void orb_extractor_equirect::compute_rectification_maps(const unsigned int cols,
                                                        const unsigned int rows,
                                                        const unsigned int division,
                                                        std::vector<cv::Mat>& maps_x,
                                                        std::vector<cv::Mat>& maps_y) {
    maps_x.resize(division);
    maps_y.resize(division);
    rectification_angles_.resize(division);

    for (unsigned int div_idx = 0; div_idx < division; ++div_idx) {
        double angle_x = M_PI * div_idx / division;
        rectification_angles_.at(div_idx) = angle_x;
        compute_map_angle_x(cols, rows, angle_x, maps_x.at(div_idx), maps_y.at(div_idx));
    }
}

void orb_extractor_equirect::compute_rectification_map_pyramid(const unsigned int cols,
                                                               const unsigned int rows,
                                                               const unsigned int division,
                                                               const unsigned int num_levels,
                                                               std::vector<std::vector<cv::Mat>>& maps_x_pyramid,
                                                               std::vector<std::vector<cv::Mat>>& maps_y_pyramid) {
    maps_x_pyramid.resize(num_levels);
    maps_y_pyramid.resize(num_levels);

    for (unsigned int level = 0; level < num_levels; ++level) {
        const double scale = orb_params_->scale_factors_.at(level);
        const cv::Size size(std::round(cols * 1.0 / scale), std::round(rows * 1.0 / scale));
        compute_rectification_maps(size.width, size.height, division, maps_x_pyramid.at(level), maps_y_pyramid.at(level));
    }
}

cv::Point orb_extractor_equirect::xyz_to_equirectangular(double x, double y, double z, int width, int height) {
    double lat = asin(y);
    double lon = atan2(x, z);

    int u = static_cast<int>((lon / M_PI + 1.0) * 0.5 * width);
    int v = static_cast<int>((lat / M_PI + 0.5) * height);

    return {u, v};
}

std::vector<cv::Point> orb_extractor_equirect::compute_equirect_curve_dots(const unsigned int cols,
                                                                           const unsigned int rows,
                                                                           const double angle,
                                                                           const unsigned int begin_col,
                                                                           const unsigned int end_col) {
    std::vector<cv::Point> curve_dots;
    std::array<double, 3> right_dir = {1.0, 0.0, 0.0};
    std::array<double, 3> front_dir = {0.0, -sin(angle), cos(angle)};

    for (unsigned int i = begin_col; i < end_col; i++) {
        double t = static_cast<double>(i) / cols;
        double theta = t * 2 * M_PI - M_PI / 2;
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);

        double x = right_dir[0] * cos_theta + front_dir[0] * sin_theta;
        double y = right_dir[1] * cos_theta + front_dir[1] * sin_theta;
        double z = right_dir[2] * cos_theta + front_dir[2] * sin_theta;

        curve_dots.push_back(xyz_to_equirectangular(x, y, z, cols, rows));
    }

    return curve_dots;
}

void orb_extractor_equirect::compute_rectification_mask() {
    rectification_mask_pyramid_.resize(orb_params_->num_levels_);

    const double rectification_angle = M_PI / (2 * division_);

    std::vector<cv::Point> rectification_contours_left_lower = compute_equirect_curve_dots(camera_->cols_, camera_->rows_, rectification_angle, 0, camera_->cols_ / 4);
    std::vector<cv::Point> rectification_contours_left_upper = compute_equirect_curve_dots(camera_->cols_, camera_->rows_, -rectification_angle, 0, camera_->cols_ / 4);
    std::vector<cv::Point> rectification_contours_left = rectification_contours_left_lower;
    rectification_contours_left.insert(rectification_contours_left.end(), rectification_contours_left_upper.rbegin(), rectification_contours_left_upper.rend());

    std::vector<cv::Point> rectification_contours_center_upper = compute_equirect_curve_dots(camera_->cols_, camera_->rows_, rectification_angle, camera_->cols_ / 4, 3 * camera_->cols_ / 4);
    std::vector<cv::Point> rectification_contours_center_lower = compute_equirect_curve_dots(camera_->cols_, camera_->rows_, -rectification_angle, camera_->cols_ / 4, 3 * camera_->cols_ / 4);
    std::vector<cv::Point> rectification_contours_center = rectification_contours_center_upper;
    rectification_contours_center.insert(rectification_contours_center.end(), rectification_contours_center_lower.rbegin(), rectification_contours_center_lower.rend());

    std::vector<cv::Point> rectification_contours_right_lower = compute_equirect_curve_dots(camera_->cols_, camera_->rows_, rectification_angle, 3 * camera_->cols_ / 4, camera_->cols_);
    std::vector<cv::Point> rectification_contours_right_upper = compute_equirect_curve_dots(camera_->cols_, camera_->rows_, -rectification_angle, 3 * camera_->cols_ / 4, camera_->cols_);
    std::vector<cv::Point> rectification_contours_right = rectification_contours_right_lower;
    rectification_contours_right.insert(rectification_contours_right.end(), rectification_contours_right_upper.rbegin(), rectification_contours_right_upper.rend());

    cv::Mat mask = cv::Mat::zeros(camera_->rows_, camera_->cols_, CV_8UC1);
    cv::fillPoly(mask, rectification_contours_left, 255);
    cv::fillPoly(mask, rectification_contours_center, 255);
    cv::fillPoly(mask, rectification_contours_right, 255);

    rectification_mask_pyramid_.at(0) = mask.clone();
    for (unsigned int level = 1; level < orb_params_->num_levels_; ++level) {
        const double scale = orb_params_->scale_factors_.at(level);
        const cv::Size size(std::round(camera_->cols_ * 1.0 / scale), std::round(camera_->rows_ * 1.0 / scale));
        cv::resize(rectification_mask_pyramid_.at(level - 1), rectification_mask_pyramid_.at(level), size, 0, 0, cv::INTER_LINEAR);
    }
}

void orb_extractor_equirect::extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                                     std::vector<cv::KeyPoint>& keypts, const cv::_OutputArray& out_descriptors) {
    if (in_image.empty()) {
        return;
    }

    // get cv::Mat of image
    const auto image = in_image.getMat();
    assert(image.type() == CV_8UC1);

    // build image pyramid
    compute_rectified_image_pyramid(image);

    // mask initialization
    if (!mask_is_initialized_ && !mask_rects_.empty()) {
        create_rectangle_mask(image.cols, image.rows);
        mask_is_initialized_ = true;
    }

    std::vector<std::vector<cv::KeyPoint>> all_keypts;

    // select mask to use
    if (!in_image_mask.empty()) {
        // Use image_mask if it is available
        const auto image_mask = in_image_mask.getMat();
        assert(image_mask.type() == CV_8UC1);
        compute_fast_keypoints(all_keypts, image_mask);
    }
    else if (!rect_mask_.empty()) {
        // Use rectangle mask if it is available and image_mask is not used
        assert(rect_mask_.type() == CV_8UC1);
        compute_fast_keypoints(all_keypts, rect_mask_);
    }
    else {
        // Do not use any mask if all masks are unavailable
        compute_fast_keypoints(all_keypts, cv::Mat());
    }

    cv::Mat descriptors;

    unsigned int num_keypts = 0;
    for (unsigned int level = 0; level < orb_params_->num_levels_; ++level) {
        num_keypts += all_keypts.at(level).size();
    }
    if (num_keypts == 0) {
        out_descriptors.release();
    }
    else {
        out_descriptors.create(num_keypts, 32, CV_8U);
        descriptors = out_descriptors.getMat();
    }

    keypts.clear();
    keypts.reserve(num_keypts);

    unsigned int offset = 0;
    std::vector<unsigned int> offsets;
    offsets.push_back(0);
    for (unsigned int level = 0; level < orb_params_->num_levels_ - 1; ++level) {
        offset += all_keypts.at(level).size();
        offsets.push_back(offset);
    }

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (unsigned int level = 0; level < orb_params_->num_levels_; ++level) {
        auto& keypts_at_level = all_keypts.at(level);
        const auto num_keypts_at_level = keypts_at_level.size();

        if (num_keypts_at_level == 0) {
            continue;
        }

        std::vector<cv::Mat> blurred_images(division_);

        cv::Mat descriptors_at_level = descriptors.rowRange(offsets[level], offsets[level] + num_keypts_at_level);
        descriptors_at_level = cv::Mat::zeros(num_keypts_at_level, 32, CV_8UC1);

        // To enable parallelization, set the environment variable OMP_MAX_ACTIVE_LEVELS to 2.
        if (desc_type_ == feature::descriptor_type::ORB) {
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (unsigned int i = 0; i < keypts_at_level.size(); ++i) {
                const auto div_idx = keypts_at_level[i].class_id;
                if (blurred_images.at(div_idx).empty()) {
                    cv::GaussianBlur(rectified_image_pyramid_.at(level).at(div_idx),
                                     blurred_images.at(div_idx), cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
                }
                cv::KeyPoint keypt = keypts_at_level[i];
                std::vector<cv::Point> points = {keypt.pt};
                std::vector<cv::Point> rectified_points = convert_rectified_points(points,
                                                                                   blurred_images.at(div_idx).cols,
                                                                                   blurred_images.at(div_idx).rows,
                                                                                   rectification_angles_.at(keypt.class_id));
                keypt.pt = rectified_points.at(0);
                compute_orb_descriptor(keypt, blurred_images.at(div_idx), descriptors_at_level.ptr(i));

                // draw keypoints
                // cv::drawKeypoints(blurred_images.at(div_idx), {keypt}, blurred_images.at(div_idx), cv::Scalar(255, 0, 0));
            }
            // for (unsigned int div_idx = 0; div_idx < division_; ++div_idx) {
            //     cv::imwrite("orb_" + std::to_string(level) + "_" + std::to_string(div_idx) + ".jpg", blurred_images.at(div_idx));
            // }
        }
        else if (desc_type_ == feature::descriptor_type::HASH_SIFT) {
#ifdef USE_CUDA_EFFICIENT_DESCRIPTORS
            hash_sift_->compute(blurred_image, keypts_at_level, descriptors_at_level);
#else
            throw std::runtime_error("cuda_efficient_features is not available");
#endif
        }
        else {
            throw std::runtime_error("Invalid descriptor_type");
        }

        correct_keypoint_scale(keypts_at_level, level);
    }

    // Collect keypoints for every scale
    for (unsigned int level = 0; level < orb_params_->num_levels_; ++level) {
        auto& keypts_at_level = all_keypts.at(level);
        keypts.insert(keypts.end(), keypts_at_level.begin(), keypts_at_level.end());
    }
}

void orb_extractor_equirect::compute_rectified_image_pyramid(const cv::Mat& image) {
    rectified_image_pyramid_.at(0).at(0) = image;
    for (unsigned int level = 0; level < orb_params_->num_levels_; ++level) {
        if (level > 0) {
            const double scale = orb_params_->scale_factors_.at(level);
            const cv::Size size(std::round(image.cols * 1.0 / scale), std::round(image.rows * 1.0 / scale));
            cv::resize(rectified_image_pyramid_.at(level - 1).at(0), rectified_image_pyramid_.at(level).at(0), size, 0, 0, cv::INTER_LINEAR);
        }
        for (unsigned int div_idx = 1; div_idx < division_; ++div_idx) {
            cv::remap(rectified_image_pyramid_.at(level).at(0),
                      rectified_image_pyramid_.at(level).at(div_idx),
                      maps_x_pyramid_.at(level).at(div_idx),
                      maps_y_pyramid_.at(level).at(div_idx),
                      cv::INTER_LINEAR,
                      cv::BORDER_REPLICATE);
        }
    }
}

std::vector<cv::Point> orb_extractor_equirect::convert_rectified_points(std::vector<cv::Point> rectified_points,
                                                                        const unsigned int cols,
                                                                        const unsigned int rows,
                                                                        const double angle_x) const {
    // Convert to Eigen::MatrixXd
    Eigen::MatrixXd rectified_points_eigen(rectified_points.size(), 2);
    for (unsigned int i = 0; i < rectified_points.size(); ++i) {
        rectified_points_eigen(i, 0) = rectified_points.at(i).x;
        rectified_points_eigen(i, 1) = rectified_points.at(i).y;
    }

    // Convert to spherical coordinate
    Eigen::MatrixXd lon_grid = (rectified_points_eigen.col(0).array() / cols - 0.5) * 2 * M_PI;
    Eigen::MatrixXd lat_grid = (rectified_points_eigen.col(1).array() / rows - 0.5) * M_PI;

    // Convert to cartesian coordinate
    Eigen::MatrixXd sph_x = lat_grid.array().cos() * lon_grid.array().sin();
    Eigen::MatrixXd sph_y = lat_grid.array().sin();
    Eigen::MatrixXd sph_z = lat_grid.array().cos() * lon_grid.array().cos();
    Eigen::MatrixXd sph(rectified_points.size(), 3);
    sph << sph_x, sph_y, sph_z;

    // Rotate around x-axis
    Eigen::Matrix3d R = Eigen::AngleAxisd(-angle_x, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::MatrixXd rotated_sph = (R * sph.transpose()).transpose();

    // Convert to spherical coordinate
    Eigen::VectorXd new_lon = rotated_sph.col(0).array().binaryExpr(
        rotated_sph.col(2).array(), [](double y, double x) { return std::atan2(y, x); });
    Eigen::VectorXd new_lat = rotated_sph.col(1).array().asin();

    // Convert to cartesian coordinate
    Eigen::VectorXd new_x = (new_lon.array() / (2.0 * M_PI)) * cols + cols / 2.0;
    Eigen::VectorXd new_y = (new_lat.array() / M_PI) * rows + rows / 2.0;

    // Convert to cv::Point
    std::vector<cv::Point> reversed_points(rectified_points.size());
    for (unsigned int i = 0; i < rectified_points.size(); ++i) {
        reversed_points.at(i) = cv::Point(new_x(i), new_y(i));
    }

    return reversed_points;
}

void orb_extractor_equirect::compute_fast_keypoints(std::vector<std::vector<cv::KeyPoint>>& all_keypts, const cv::Mat& mask) const {
    all_keypts.resize(orb_params_->num_levels_);

    // An anonymous function which checks mask(image or rectangle)
    auto is_in_mask = [&mask](const unsigned int y, const unsigned int x, const float scale_factor) {
        return mask.at<unsigned char>(y * scale_factor, x * scale_factor) == 0;
    };

    constexpr unsigned int overlap = 6;
    constexpr unsigned int cell_size = 64;

    for (int64_t level = 0; level < orb_params_->num_levels_; ++level) {
        const float scale_factor = orb_params_->scale_factors_.at(level);

        constexpr unsigned int min_border_x = orb_patch_radius_;
        constexpr unsigned int min_border_y = orb_patch_radius_;
        const unsigned int max_border_x = rectified_image_pyramid_.at(level).at(0).cols - orb_patch_radius_;
        const unsigned int max_border_y = rectified_image_pyramid_.at(level).at(0).rows - orb_patch_radius_;

        const unsigned int min_border_div_y = (rectified_image_pyramid_.at(level).at(0).rows / (division_ * 2)) * (division_ - 1) - orb_patch_radius_;
        const unsigned int max_border_div_y = (rectified_image_pyramid_.at(level).at(0).rows / (division_ * 2)) * (division_ + 1) + orb_patch_radius_;

        const unsigned int width = max_border_x - min_border_x;
        const unsigned int height = max_border_div_y - min_border_div_y;

        const unsigned int num_cols = width / cell_size + 1;
        const unsigned int num_rows = height / cell_size + 1;

        std::vector<cv::KeyPoint> keypts_to_distribute;
        keypts_to_distribute.reserve(500);

        auto erase_outer_of_rectification_mask = [this, level](std::vector<cv::KeyPoint>& keypts) {
            keypts.erase(
                std::remove_if(
                    keypts.begin(), keypts.end(), [this, level](cv::KeyPoint& keypt) {
                        return rectification_mask_pyramid_.at(level).at<uchar>(keypt.pt.y, keypt.pt.x) == 0;
                    }),
                keypts.end());
        };

        for (int64_t div_idx = 0; div_idx < division_; ++div_idx) {
            std::vector<cv::KeyPoint> keypts_in_rectification;
            cv::Mat rectified_image = rectified_image_pyramid_.at(level).at(div_idx);

            for (int64_t i = 0; i < num_rows; ++i) {
                const unsigned int min_y = min_border_div_y + i * cell_size;
                if (max_border_div_y - overlap <= min_y) {
                    continue;
                }
                unsigned int max_y = min_y + cell_size + overlap;
                if (max_border_div_y < max_y) {
                    max_y = max_border_div_y;
                }

                for (int64_t j = 0; j < num_cols; ++j) {
                    const unsigned int min_x = min_border_x + j * cell_size;
                    if (max_border_x - overlap <= min_x) {
                        continue;
                    }
                    unsigned int max_x = min_x + cell_size + overlap;
                    if (max_border_x < max_x) {
                        max_x = max_border_x;
                    }

                    std::vector<cv::KeyPoint> keypts_in_cell;
                    cv::FAST(rectified_image.rowRange(min_y, max_y).colRange(min_x, max_x),
                             keypts_in_cell, orb_params_->ini_fast_thr_, true);
                    for (auto& keypt : keypts_in_cell) {
                        keypt.pt.x += min_border_x + j * cell_size;
                        keypt.pt.y += min_border_div_y + i * cell_size;
                    }
                    erase_outer_of_rectification_mask(keypts_in_cell);

                    // Re-compute FAST keypoint with reduced threshold if enough keypoint was not got
                    if (keypts_in_cell.empty()) {
                        cv::FAST(rectified_image.rowRange(min_y, max_y).colRange(min_x, max_x),
                                 keypts_in_cell, orb_params_->min_fast_thr_, true);
                        for (auto& keypt : keypts_in_cell) {
                            keypt.pt.x += min_border_x + j * cell_size;
                            keypt.pt.y += min_border_div_y + i * cell_size;
                        }
                        erase_outer_of_rectification_mask(keypts_in_cell);
                    }

                    keypts_in_rectification.insert(keypts_in_rectification.end(), keypts_in_cell.begin(), keypts_in_cell.end());
                }
            }

            // cv::FAST(rectified_image, keypts_in_rectification, orb_params_->ini_fast_thr_, true);
            // erase_outer_of_rectification_mask(keypts_in_rectification);
            // // Re-compute FAST keypoint with reduced threshold if enough keypoint was not got
            // if (keypts_in_rectification.empty()) {
            //     cv::FAST(rectified_image, keypts_in_rectification, orb_params_->min_fast_thr_, true);
            //     erase_outer_of_rectification_mask(keypts_in_rectification);
            // }

            if (keypts_in_rectification.empty()) {
                continue;
            }

            // draw keypoints
            // cv::Mat image_with_keypoints;
            // cv::cvtColor(rectified_image.clone(), image_with_keypoints, cv::COLOR_GRAY2BGR);
            // for (const auto& keypt : keypts_in_rectification) {
            //     cv::circle(image_with_keypoints, keypt.pt, 2, cv::Scalar(0, 0, 255), -1);
            // }
            // cv::imwrite("keypoints_" + std::to_string(level) + "_" + std::to_string(div_idx) + ".jpg", image_with_keypoints);

            std::vector<cv::Point> keypts_point;
            for (const auto& keypt : keypts_in_rectification) {
                keypts_point.push_back(keypt.pt);
            }
            std::vector<cv::Point> reversed_keypts_point = convert_rectified_points(keypts_point,
                                                                                    rectified_image.cols,
                                                                                    rectified_image.rows,
                                                                                    -rectification_angles_.at(div_idx));
            for (unsigned int i = 0; i < keypts_in_rectification.size(); ++i) {
                cv::KeyPoint& keypt = keypts_in_rectification.at(i);
                keypt.pt.x = reversed_keypts_point.at(i).x;
                keypt.pt.y = reversed_keypts_point.at(i).y;
                // borrow the class_id field to store the division index
                keypt.class_id = div_idx;
            }

            // draw keypoints
            // cv::cvtColor(rectified_image_pyramid_.at(level).at(0).clone(), image_with_keypoints, cv::COLOR_GRAY2BGR);
            // for (const auto& keypt : keypts_in_rectification) {
            //     cv::circle(image_with_keypoints, keypt.pt, 2, cv::Scalar(0, 0, 255), -1);
            // }
            // cv::imwrite("keypoints_" + std::to_string(level) + "_" + std::to_string(div_idx) + "_reversed.jpg", image_with_keypoints);

            if (!mask.empty()) {
                std::vector<cv::KeyPoint> keypts_in_rectification_masked;
                for (auto&& keypt : keypts_in_rectification) {
                    // Check if the keypoint is in the mask
                    if (is_in_mask(keypt.pt.y, keypt.pt.x, scale_factor)) {
                        continue;
                    }
                    else if (min_border_x <= keypt.pt.x && keypt.pt.x < max_border_x && min_border_y <= keypt.pt.y && keypt.pt.y < max_border_y) {
                        keypts_in_rectification_masked.push_back(std::move(keypt));
                    }
                }
                keypts_in_rectification = std::move(keypts_in_rectification_masked);
            }

            {
                keypts_to_distribute.insert(keypts_to_distribute.end(), keypts_in_rectification.begin(), keypts_in_rectification.end());
            }
        }

        std::vector<cv::KeyPoint>& keypts_at_level = all_keypts.at(level);

        // Distribute keypoints via tree
        for (auto& keypt : keypts_to_distribute) {
            keypt.pt.x -= min_border_x;
            keypt.pt.y -= min_border_y;
        }
        keypts_at_level = distribute_keypoints(keypts_to_distribute, min_border_x, max_border_x, min_border_y, max_border_y, scale_factor);
        SPDLOG_TRACE("keypts_at_level {} filtered={} raw={}", level, keypts_at_level.size(), keypts_to_distribute.size());

        // Keypoint size is patch size modified by the scale factor
        const unsigned int scaled_patch_size = orb_impl_.fast_patch_size_ * scale_factor;

        for (auto& keypt : keypts_at_level) {
            // Translation correction (scale will be corrected after ORB description)
            keypt.pt.x += min_border_x;
            keypt.pt.y += min_border_y;
            // Set the other information
            keypt.octave = level;
            keypt.size = scaled_patch_size;
        }

        // draw keypoints
        // cv::Mat image_with_keypoints;
        // cv::cvtColor(rectified_image_pyramid_.at(level).at(0).clone(), image_with_keypoints, cv::COLOR_GRAY2BGR);
        // for (const auto& keypt : keypts_at_level) {
        //     cv::circle(image_with_keypoints, keypt.pt, 2, cv::Scalar(0, 0, 255), -1);
        // }
        // cv::imwrite("keypoints_" + std::to_string(level) + "_reversed_alldiv.jpg", image_with_keypoints);

        compute_orientation(rectified_image_pyramid_.at(level), all_keypts.at(level));
    }
}

void orb_extractor_equirect::compute_orientation(const std::vector<cv::Mat>& rectified_images, std::vector<cv::KeyPoint>& keypts) const {
    for (auto& keypt : keypts) {
        keypt.angle = ic_angle(rectified_images.at(keypt.class_id), keypt.pt);
    }
}

} // namespace feature
} // namespace stella_vslam
