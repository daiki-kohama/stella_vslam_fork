#include "stella_vslam/feature/orb_extractor.h"
#include "stella_vslam/type.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>

#include <spdlog/spdlog.h>

namespace stella_vslam {
namespace feature {

orb_extractor::orb_extractor(const orb_params* orb_params,
                             const unsigned int min_area,
                             const descriptor_type desc_type,
                             const bool use_ico_image_pyramid,
                             const std::vector<std::vector<float>>& mask_rects)
    : orb_params_(orb_params), mask_rects_(mask_rects), min_area_sqrt_(std::sqrt(min_area)), desc_type_(desc_type), use_ico_image_pyramid_(use_ico_image_pyramid) {
    // resize buffers according to the number of levels
    if (use_ico_image_pyramid_) {
        ico_image_pyramid_.resize(orb_params_->num_levels_);
    } else {
        image_pyramid_.resize(orb_params_->num_levels_);
    }
#ifdef USE_CUDA_EFFICIENT_DESCRIPTORS
    hash_sift_ = cv::cuda::HashSIFT::create(1.0, cv::cuda::HashSIFT::SIZE_256_BITS);
#endif
}

void orb_extractor::extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                            std::vector<cv::KeyPoint>& keypts, const cv::_OutputArray& out_descriptors) {
    if (in_image.empty()) {
        return;
    }

    // get cv::Mat of image
    const auto image = in_image.getMat();
    assert(image.type() == CV_8UC1);

    if (use_ico_image_pyramid_ && !is_ico_unwrapper_initialized_) {
        initialize_ico_unwrappers(image.cols, image.rows);
    }

    // build image pyramid
    compute_image_pyramid(image);

    // mask initialization
    if (!mask_is_initialized_ && !mask_rects_.empty()) {
        create_rectangle_mask(image.cols, image.rows);
        mask_is_initialized_ = true;
    }

    std::vector<std::vector<cv::KeyPoint>> all_keypts;
    std::vector<std::vector<eqr_ico_point>> all_eqr_ico_pts;

    // select mask to use
    if (!in_image_mask.empty()) {
        // Use image_mask if it is available
        const auto image_mask = in_image_mask.getMat();
        assert(image_mask.type() == CV_8UC1);
        if (use_ico_image_pyramid_) {
            const auto ico_masks = create_ico_masks(image_mask);
            compute_fast_keypoints_ico(all_keypts, all_eqr_ico_pts, ico_masks, image.cols, image.rows);
        }
        else {
            compute_fast_keypoints(all_keypts, image_mask);
        }
    }
    else if (!rect_mask_.empty()) {
        // Use rectangle mask if it is available and image_mask is not used
        assert(rect_mask_.type() == CV_8UC1);
        if (use_ico_image_pyramid_) {
            compute_fast_keypoints_ico(all_keypts, all_eqr_ico_pts, rect_ico_masks_, image.cols, image.rows);
        }
        else {
            compute_fast_keypoints(all_keypts, rect_mask_);
        }
    }
    else {
        // Do not use any mask if all masks are unavailable
        if (use_ico_image_pyramid_) {
            const std::vector<cv::Mat> empty_masks(eqr_ico::IcosahedronUnwrapper::face_count_, cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255)));
            compute_fast_keypoints_ico(all_keypts, all_eqr_ico_pts, empty_masks, image.cols, image.rows);
        }
        else {
            compute_fast_keypoints(all_keypts, cv::Mat());
        }
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

#if defined(USE_OPENMP) and !defined(USE_CUDA_EFFICIENT_DESCRIPTORS)
#pragma omp parallel for schedule(dynamic)
#endif
    for (unsigned int level = 0; level < orb_params_->num_levels_; ++level) {
        auto& keypts_at_level = all_keypts.at(level);
        const auto num_keypts_at_level = keypts_at_level.size();

        if (num_keypts_at_level == 0) {
            continue;
        }

        cv::Mat blurred_image;
        std::vector<cv::Mat> blurred_ico_images(eqr_ico::IcosahedronUnwrapper::face_count_);
        if (use_ico_image_pyramid_) {
            for (unsigned int face_idx = 0; face_idx < eqr_ico::IcosahedronUnwrapper::face_count_; ++face_idx) {
                cv::GaussianBlur(ico_image_pyramid_.at(level).at(face_idx), blurred_ico_images.at(face_idx), cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
            }
        }
        else {
            cv::GaussianBlur(image_pyramid_.at(level), blurred_image, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
        }

        cv::Mat descriptors_at_level = descriptors.rowRange(offsets[level], offsets[level] + num_keypts_at_level);
        descriptors_at_level = cv::Mat::zeros(num_keypts_at_level, 32, CV_8UC1);

        // To enable parallelization, set the environment variable OMP_MAX_ACTIVE_LEVELS to 2.
        if (desc_type_ == feature::descriptor_type::ORB) {
            if (use_ico_image_pyramid_) {
                auto& eqr_ico_pts_at_level = all_eqr_ico_pts.at(level);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (unsigned int i = 0; i < keypts_at_level.size(); ++i) {
                    const auto& eqr_ico_pt = eqr_ico_pts_at_level.at(i);
                    compute_orb_descriptor(keypts_at_level[i], blurred_ico_images.at(eqr_ico_pt.face_idx), descriptors_at_level.ptr(i));
                    keypts_at_level.at(i).pt = eqr_ico_pt.eqr_pt;
                }
            } else
            {
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (unsigned int i = 0; i < keypts_at_level.size(); ++i) {
                    compute_orb_descriptor(keypts_at_level[i], blurred_image, descriptors_at_level.ptr(i));
                }
            }
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

void orb_extractor::initialize_ico_unwrappers(const unsigned int cols, const unsigned int rows) {
    const int margin_px = static_cast<int>(orb_patch_radius_);
    ico_unwrappers_.reserve(orb_params_->num_levels_);
    for (unsigned int level = 0; level < orb_params_->num_levels_; ++level) {
        const double scale = orb_params_->scale_factors_.at(level);
        const double pixelsPerUnit = cols / (2 * M_PI) / scale;
        ico_unwrappers_.emplace_back(cv::Size(cols, rows), margin_px, pixelsPerUnit);
    }
    ico_original_masks_ = ico_unwrappers_.at(0).getFaceMasks();
    is_ico_unwrapper_initialized_ = true;
}

void orb_extractor::create_rectangle_mask(const unsigned int cols, const unsigned int rows) {
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
    if (use_ico_image_pyramid_) {
        rect_ico_masks_ = create_ico_masks(rect_mask_);
    }
}

std::vector<cv::Mat> orb_extractor::create_ico_masks(const cv::Mat& mask) {
    if (!is_ico_unwrapper_initialized_) {
        initialize_ico_unwrappers(mask.cols, mask.rows);
    }
    std::vector<cv::Mat> ico_masks(eqr_ico::IcosahedronUnwrapper::face_count_);
    for (unsigned int face_idx = 0; face_idx < eqr_ico::IcosahedronUnwrapper::face_count_; ++face_idx) {
        ico_masks.at(face_idx) = ico_unwrappers_.at(0).applyFace(mask, face_idx).image;
    }
    return ico_masks;
}

void orb_extractor::compute_image_pyramid(const cv::Mat& image) {
    if (use_ico_image_pyramid_) {
        if (!is_ico_unwrapper_initialized_) {
            initialize_ico_unwrappers(image.cols, image.rows);
        }
        for (unsigned int level = 0; level < orb_params_->num_levels_; ++level) {
            ico_image_pyramid_.at(level).resize(eqr_ico::IcosahedronUnwrapper::face_count_);
            for (unsigned int face_idx = 0; face_idx < eqr_ico::IcosahedronUnwrapper::face_count_; ++face_idx) {
                ico_image_pyramid_.at(level).at(face_idx) = ico_unwrappers_.at(level).applyFace(image, face_idx).image;
            }
        }
    } else {
        image_pyramid_.at(0) = image;
        for (unsigned int level = 1; level < orb_params_->num_levels_; ++level) {
            // determine the size of an image
            const double scale = orb_params_->scale_factors_.at(level);
            const cv::Size size(std::round(image.cols * 1.0 / scale), std::round(image.rows * 1.0 / scale));
            // resize
            cv::resize(image_pyramid_.at(level - 1), image_pyramid_.at(level), size, 0, 0, cv::INTER_LINEAR);
        }
    }
}

void orb_extractor::compute_fast_keypoints(std::vector<std::vector<cv::KeyPoint>>& all_keypts, const cv::Mat& mask) const {
    all_keypts.resize(orb_params_->num_levels_);

    // An anonymous function which checks mask(image or rectangle)
    auto is_in_mask = [&mask](const unsigned int y, const unsigned int x, const float scale_factor) {
        return mask.at<unsigned char>(y * scale_factor, x * scale_factor) == 0;
    };

    constexpr unsigned int overlap = 6;
    constexpr unsigned int cell_size = 64;

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t level = 0; level < orb_params_->num_levels_; ++level) {
        const float scale_factor = orb_params_->scale_factors_.at(level);

        constexpr unsigned int min_border_x = orb_patch_radius_;
        constexpr unsigned int min_border_y = orb_patch_radius_;
        const unsigned int max_border_x = image_pyramid_.at(level).cols - orb_patch_radius_;
        const unsigned int max_border_y = image_pyramid_.at(level).rows - orb_patch_radius_;

        const unsigned int width = max_border_x - min_border_x;
        const unsigned int height = max_border_y - min_border_y;

        const unsigned int num_cols = width / cell_size + 1;
        const unsigned int num_rows = height / cell_size + 1;

        std::vector<cv::KeyPoint> keypts_to_distribute;
        keypts_to_distribute.reserve(500);

        // To enable parallelization, set the environment variable OMP_MAX_ACTIVE_LEVELS to 2.
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < num_rows; ++i) {
            const unsigned int min_y = min_border_y + i * cell_size;
            if (max_border_y - overlap <= min_y) {
                continue;
            }
            unsigned int max_y = min_y + cell_size + overlap;
            if (max_border_y < max_y) {
                max_y = max_border_y;
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

                // Pass FAST computation if one of the corners of a patch is in the mask
                // if (!mask.empty()) {
                //     if (is_in_mask(min_y, min_x, scale_factor) || is_in_mask(max_y, min_x, scale_factor)
                //         || is_in_mask(min_y, max_x, scale_factor) || is_in_mask(max_y, max_x, scale_factor)) {
                //         continue;
                //     }
                // }

                std::vector<cv::KeyPoint> keypts_in_cell;
                cv::FAST(image_pyramid_.at(level).rowRange(min_y, max_y).colRange(min_x, max_x),
                         keypts_in_cell, orb_params_->ini_fast_thr_, true);

                // Re-compute FAST keypoint with reduced threshold if enough keypoint was not got
                if (keypts_in_cell.empty()) {
                    cv::FAST(image_pyramid_.at(level).rowRange(min_y, max_y).colRange(min_x, max_x),
                             keypts_in_cell, orb_params_->min_fast_thr_, true);
                }

                if (keypts_in_cell.empty()) {
                    continue;
                }

                for (auto& keypt : keypts_in_cell) {
                    keypt.pt.x += j * cell_size;
                    keypt.pt.y += i * cell_size;
                }

                if (!mask.empty()) {
                    std::vector<cv::KeyPoint> keypts_in_cell_masked;
                    for (auto&& keypt : keypts_in_cell) {
                        // Check if the keypoint is in the mask
                        if (is_in_mask(min_border_y + keypt.pt.y, min_border_x + keypt.pt.x, scale_factor)) {
                            continue;
                        }
                        keypts_in_cell_masked.push_back(std::move(keypt));
                    }
                    keypts_in_cell = std::move(keypts_in_cell_masked);
                }

#ifdef USE_OPENMP
#pragma omp critical
#endif
                {
                    keypts_to_distribute.insert(keypts_to_distribute.end(), keypts_in_cell.begin(), keypts_in_cell.end());
                }
            }
        }

        std::vector<cv::KeyPoint>& keypts_at_level = all_keypts.at(level);

        // Distribute keypoints via tree
        std::vector<unsigned int> distributed_keypts_indices;
        keypts_at_level = distribute_keypoints(keypts_to_distribute, distributed_keypts_indices, min_border_x, max_border_x, min_border_y, max_border_y, scale_factor);
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

        compute_orientation(image_pyramid_.at(level), all_keypts.at(level));
    }
}

void orb_extractor::compute_fast_keypoints_ico(std::vector<std::vector<cv::KeyPoint>>& all_keypts, std::vector<std::vector<eqr_ico_point>>& all_eqr_ico_pts,
                                               const std::vector<cv::Mat>& ico_masks, const unsigned int cols, const unsigned int rows) const {
    all_keypts.resize(orb_params_->num_levels_);
    all_eqr_ico_pts.resize(orb_params_->num_levels_);

    // An anonymous function which checks mask(image or rectangle)
    auto is_in_mask = [this, &ico_masks](const unsigned int y, const unsigned int x, const unsigned int offset_y, const unsigned int offset_x,
                                         const float scale_factor, const unsigned int face_idx) {
        const auto unscaled_x = static_cast<unsigned int>(x * scale_factor + offset_x);
        const auto unscaled_y = static_cast<unsigned int>(y * scale_factor + offset_y);
        return ico_masks.at(face_idx).at<unsigned char>(unscaled_y, unscaled_x) == 0 || ico_original_masks_.at(face_idx).at<unsigned char>(unscaled_y, unscaled_x) == 0;
    };

    constexpr unsigned int overlap = 6;
    constexpr unsigned int cell_size = 64;

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t level = 0; level < orb_params_->num_levels_; ++level) {
        const float scale_factor = orb_params_->scale_factors_.at(level);

        std::vector<cv::KeyPoint> keypts_to_distribute;
        keypts_to_distribute.reserve(500);
        std::vector<eqr_ico_point> eqr_ico_pts_to_distribute;
        eqr_ico_pts_to_distribute.reserve(500);

        // To enable parallelization, set the environment variable OMP_MAX_ACTIVE_LEVELS to 2.
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (unsigned int face_idx = 0; face_idx < eqr_ico::IcosahedronUnwrapper::face_count_; ++face_idx) {
            std::vector<cv::KeyPoint> keypts_to_wrap;
            keypts_to_wrap.reserve(100);

            constexpr unsigned int min_border_x = orb_patch_radius_;
            constexpr unsigned int min_border_y = orb_patch_radius_;
            const unsigned int max_border_x = ico_image_pyramid_.at(level).at(face_idx).cols - orb_patch_radius_;
            const unsigned int max_border_y = ico_image_pyramid_.at(level).at(face_idx).rows - orb_patch_radius_;

            const unsigned int width = max_border_x - min_border_x;
            const unsigned int height = max_border_y - min_border_y;

            const unsigned int num_cols = width / cell_size + 1;
            const unsigned int num_rows = height / cell_size + 1;


            for (int64_t i = 0; i < num_rows; ++i) {
                const unsigned int min_y = min_border_y + i * cell_size;
                if (max_border_y - overlap <= min_y) {
                    continue;
                }
                unsigned int max_y = min_y + cell_size + overlap;
                if (max_border_y < max_y) {
                    max_y = max_border_y;
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

                    // Pass FAST computation if one of the corners of a patch is in the mask
                    // if (!mask.empty()) {
                    //     if (is_in_mask(min_y, min_x, scale_factor) || is_in_mask(max_y, min_x, scale_factor)
                    //         || is_in_mask(min_y, max_x, scale_factor) || is_in_mask(max_y, max_x, scale_factor)) {
                    //         continue;
                    //     }
                    // }

                    std::vector<cv::KeyPoint> keypts_in_cell;
                    cv::FAST(ico_image_pyramid_.at(level).at(face_idx).rowRange(min_y, max_y).colRange(min_x, max_x),
                            keypts_in_cell, orb_params_->ini_fast_thr_, true);

                    // Re-compute FAST keypoint with reduced threshold if enough keypoint was not got
                    if (keypts_in_cell.empty()) {
                        cv::FAST(ico_image_pyramid_.at(level).at(face_idx).rowRange(min_y, max_y).colRange(min_x, max_x),
                                keypts_in_cell, orb_params_->min_fast_thr_, true);
                    }

                    if (keypts_in_cell.empty()) {
                        continue;
                    }

                    for (auto& keypt : keypts_in_cell) {
                        keypt.pt.x += j * cell_size;
                        keypt.pt.y += i * cell_size;
                    }

                    if (!ico_masks.at(face_idx).empty()) {
                        std::vector<cv::KeyPoint> keypts_in_cell_masked;
                        for (auto&& keypt : keypts_in_cell) {
                            // Check if the keypoint is in the mask
                            if (is_in_mask(keypt.pt.y, keypt.pt.x, min_border_y, min_border_x, scale_factor, face_idx)) {
                                continue;
                            }
                            keypts_in_cell_masked.push_back(std::move(keypt));
                        }
                        keypts_in_cell = std::move(keypts_in_cell_masked);
                    }

    #ifdef USE_OPENMP
    #pragma omp critical
    #endif
                    {
                        keypts_to_wrap.insert(keypts_to_wrap.end(), keypts_in_cell.begin(), keypts_in_cell.end());
                    }
                }
            }
            // keypts_to_wrap を正距円筒図法の座標系に変換する
            for (auto& keypt : keypts_to_wrap) {
                keypt.pt.x += min_border_x;
                keypt.pt.y += min_border_y;
                const auto unwrapped_pt = ico_unwrappers_.at(level).facePixelToEquirectangularPixel(face_idx, keypt.pt);
                const auto scaled_unwrapped_pt = unwrapped_pt / scale_factor;
                eqr_ico_pts_to_distribute.push_back({scaled_unwrapped_pt, keypt.pt, face_idx});
                keypt.pt = scaled_unwrapped_pt;
            }

            {
                keypts_to_distribute.insert(keypts_to_distribute.end(), keypts_to_wrap.begin(), keypts_to_wrap.end());
            }
        }

        std::vector<cv::KeyPoint>& keypts_at_level = all_keypts.at(level);
        std::vector<eqr_ico_point>& eqr_ico_pts_at_level = all_eqr_ico_pts.at(level);

        // Distribute keypoints via tree
        const unsigned int min_x = 0;
        const unsigned int min_y = 0;
        const unsigned int max_x = cols;
        const unsigned int max_y = rows;
        std::vector<unsigned int> distributed_keypts_indices;
        keypts_at_level = distribute_keypoints(keypts_to_distribute, distributed_keypts_indices, min_x, max_x, min_y, max_y, scale_factor);
        SPDLOG_TRACE("keypts_at_level {} filtered={} raw={}", level, keypts_at_level.size(), keypts_to_distribute.size());

        // Update ico_pts_at_level with the corresponding ico points for the distributed keypoints
        eqr_ico_pts_at_level.clear();
        eqr_ico_pts_at_level.reserve(keypts_at_level.size());
        for (unsigned int i = 0; i < distributed_keypts_indices.size(); ++i) {
            const unsigned int idx = distributed_keypts_indices.at(i);
            eqr_ico_pts_at_level.push_back(eqr_ico_pts_to_distribute.at(idx));
            keypts_at_level.at(i).pt = eqr_ico_pts_to_distribute.at(idx).ico_pt;
        }

        // Keypoint size is patch size modified by the scale factor
        const unsigned int scaled_patch_size = orb_impl_.fast_patch_size_ * scale_factor;

        for (auto& keypt : keypts_at_level) {
            // Scale will be corrected after ORB description
            // Set the other information
            keypt.octave = level;
            keypt.size = scaled_patch_size;
        }

        compute_ico_orientation(ico_image_pyramid_.at(level), all_keypts.at(level), all_eqr_ico_pts.at(level));
    }
}

std::vector<cv::KeyPoint> orb_extractor::distribute_keypoints(const std::vector<cv::KeyPoint>& keypts_to_distribute,
                                                              std::vector<unsigned int>& distributed_keypts_indices,
                                                              const int min_x, const int max_x, const int min_y, const int max_y,
                                                              const float scale_factor) const {
    double scaled_min_area_sqrt = min_area_sqrt_ / scale_factor;
    unsigned int num_x_grid = std::ceil((max_x - min_x) / scaled_min_area_sqrt);
    unsigned int num_y_grid = std::ceil((max_y - min_y) / scaled_min_area_sqrt);
    double delta_x = static_cast<double>(max_x - min_x) / num_x_grid;
    double delta_y = static_cast<double>(max_y - min_y) / num_y_grid;
    std::vector<cv::KeyPoint> result_keypts;
    result_keypts.reserve(num_x_grid * num_y_grid);
    std::unordered_map<unsigned int, std::pair<cv::KeyPoint, double>> keypt_response_map;
    std::vector<std::vector<cv::KeyPoint>> keypts_on_grid(num_x_grid * num_y_grid);
    std::vector<std::vector<unsigned int>> keypts_indices_on_grid(num_x_grid * num_y_grid);

    for (unsigned int i = 0; i < keypts_to_distribute.size(); ++i) {
        const auto& keypt = keypts_to_distribute.at(i);
        const unsigned int ix = keypt.pt.x / delta_x;
        const unsigned int iy = keypt.pt.y / delta_y;
        const unsigned int idx = ix + iy * num_x_grid;
        keypts_on_grid[idx].push_back(keypt);
        keypts_indices_on_grid[idx].push_back(i);
    }

    for (unsigned int i = 0; i < keypts_on_grid.size(); ++i) {
        auto& keypts = keypts_on_grid[i];
        if (keypts.empty()) {
            continue;
        }
        auto& selected_keypt = keypts.at(0);
        auto& selected_keypt_index = keypts_indices_on_grid[i].at(0);
        double max_response = selected_keypt.response;

        for (unsigned int k = 1; k < keypts.size(); ++k) {
            const auto& keypt = keypts[k];
            if (keypt.response > max_response) {
                selected_keypt = keypt;
                selected_keypt_index = keypts_indices_on_grid[i].at(k);
                max_response = keypt.response;
            }
        }

        result_keypts.push_back(selected_keypt);
        distributed_keypts_indices.push_back(selected_keypt_index);
    }

    return result_keypts;
}

void orb_extractor::compute_orientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypts) const {
    for (auto& keypt : keypts) {
        keypt.angle = ic_angle(image, keypt.pt);
    }
}

void orb_extractor::compute_ico_orientation(const std::vector<cv::Mat>& ico_images, std::vector<cv::KeyPoint>& keypts,
                                            const std::vector<eqr_ico_point>& eqr_ico_pts) const {
    for (unsigned int i = 0; i < keypts.size(); ++i) {
        auto& keypt = keypts.at(i);
        const auto& eqr_ico_pt = eqr_ico_pts.at(i);
        const unsigned int face_idx = eqr_ico_pt.face_idx;
        const auto& ico_image = ico_images.at(face_idx);
        keypt.angle = ic_angle(ico_image, keypt.pt);
    }
}

void orb_extractor::correct_keypoint_scale(std::vector<cv::KeyPoint>& keypts_at_level, const unsigned int level) const {
    if (level == 0) {
        return;
    }
    const float scale_at_level = orb_params_->scale_factors_.at(level);
    for (auto& keypt_at_level : keypts_at_level) {
        keypt_at_level.pt *= scale_at_level;
    }
}

float orb_extractor::ic_angle(const cv::Mat& image, const cv::Point2f& point) const {
    return orb_impl_.ic_angle(image, point);
}

void orb_extractor::compute_orb_descriptor(const cv::KeyPoint& keypt, const cv::Mat& image, uchar* desc) const {
    orb_impl_.compute_orb_descriptor(keypt, image, desc);
}

} // namespace feature
} // namespace stella_vslam
