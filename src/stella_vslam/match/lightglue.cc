#include "stella_vslam/data/frame.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/match/lightglue.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/solve/essential_solver.h"
#include "stella_vslam/type.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace stella_vslam {
namespace match {

unsigned int lightglue::match_frame_and_frame(data::frame& frm_1, data::frame& frm_2, std::vector<cv::Point2f>& prev_matched_pts,
                                              std::vector<int>& matched_indices_2_in_frm_1, std::vector<double>& matched_scores_2_in_frm_1) const {
    std::cout << "IN match::lightglue::match_frame_and_frame" << std::endl;

    unsigned int num_matches = 0;

    matched_indices_2_in_frm_1 = std::vector<int>(frm_1.frm_obs_.lg_keypts_.size(), -1);
    matched_scores_2_in_frm_1 = std::vector<double>(frm_1.frm_obs_.lg_keypts_.size(), -1.0);
    std::vector<double> matched_scores_1_in_frm_2(frm_2.frm_obs_.lg_keypts_.size(), -1.0);

    std::vector<std::pair<unsigned int, unsigned int>> matched_idx_pairs;
    std::vector<double> matched_pair_scores;
    lightglue_->image_match(frm_1.image_, frm_2.image_, frm_1.frm_obs_.lg_keypts_, frm_2.frm_obs_.lg_keypts_,
                            frm_1.frm_obs_.lg_descriptors_, frm_2.frm_obs_.lg_descriptors_, matched_idx_pairs, matched_pair_scores);

    for (const auto& matched_idx_pair : matched_idx_pairs) {
        const auto idx_1 = matched_idx_pair.first;
        const auto idx_2 = matched_idx_pair.second;

        matched_indices_2_in_frm_1.at(idx_1) = idx_2;
        matched_scores_2_in_frm_1.at(idx_1) = matched_pair_scores.at(num_matches);
        matched_scores_1_in_frm_2.at(idx_2) = matched_pair_scores.at(num_matches);
        ++num_matches;
    }
    frm_1.add_match_score(frm_2.id_, matched_scores_2_in_frm_1);
    frm_2.add_match_score(frm_1.id_, matched_scores_1_in_frm_2);

    // Update the previous matches
    for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_frm_1.size(); ++idx_1) {
        if (0 <= matched_indices_2_in_frm_1.at(idx_1)) {
            prev_matched_pts.at(idx_1) = frm_2.frm_obs_.lg_keypts_.at(matched_indices_2_in_frm_1.at(idx_1));
        }
    }

    return num_matches;
}

unsigned int lightglue::match_current_and_last_frames(data::frame& curr_frm, const data::frame& last_frm, const float margin, std::vector<double>& matched_scores_in_cur) const {
    unsigned int num_matches = 0;

    const Mat33_t rot_cw = curr_frm.get_rot_cw();
    const Vec3_t trans_cw = curr_frm.get_trans_cw();

    const Vec3_t trans_wc = -rot_cw.transpose() * trans_cw;

    const Mat33_t rot_lw = last_frm.get_rot_cw();
    const Vec3_t trans_lw = last_frm.get_trans_cw();

    const Vec3_t trans_lc = rot_lw * trans_wc + trans_lw;

    // For non-monocular, check if the z component of the current-to-last translation vector is moving forward
    // The z component is positive going -> moving forward
    const bool assume_forward = (curr_frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
                                    ? false
                                    : trans_lc(2) > curr_frm.camera_->true_baseline_;
    // The z component is negative going -> moving backward
    const bool assume_backward = (curr_frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
                                     ? false
                                     : -trans_lc(2) > curr_frm.camera_->true_baseline_;

    std::vector<cv::Point2f> curr_frm_valid_lg_keypts;
    std::vector<std::vector<double>> curr_frm_valid_lg_descriptors;
    std::vector<unsigned int> curr_frm_idices_valid;
    for (unsigned int idx_curr = 0; idx_curr < curr_frm.frm_obs_.lg_keypts_.size(); ++idx_curr) {
        const auto& lm = curr_frm.get_landmark(idx_curr);
        if (lm && lm->has_observation()) {
            continue;
        }
        curr_frm_valid_lg_keypts.push_back(curr_frm.frm_obs_.lg_keypts_.at(idx_curr));
        curr_frm_valid_lg_descriptors.push_back(curr_frm.frm_obs_.lg_descriptors_.at(idx_curr));
        curr_frm_idices_valid.push_back(idx_curr);
    }

    std::vector<cv::Point2f> last_frm_valid_lg_keypts;
    std::vector<std::vector<double>> last_frm_valid_lg_descriptors;
    std::vector<unsigned int> last_frm_idices_valid;
    for (unsigned int idx_last = 0; idx_last < last_frm.frm_obs_.lg_keypts_.size(); ++idx_last) {
        const auto& lm = last_frm.get_landmark(idx_last);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        last_frm_valid_lg_keypts.push_back(last_frm.frm_obs_.lg_keypts_.at(idx_last));
        last_frm_valid_lg_descriptors.push_back(last_frm.frm_obs_.lg_descriptors_.at(idx_last));
        last_frm_idices_valid.push_back(idx_last);
    }

    std::vector<std::pair<unsigned int, unsigned int>> matched_idx_pairs;
    std::vector<double> matched_pair_scores;
    lightglue_->image_match(curr_frm.image_, last_frm.image_, curr_frm_valid_lg_keypts, last_frm_valid_lg_keypts,
                            curr_frm_valid_lg_descriptors, last_frm_valid_lg_descriptors, matched_idx_pairs, matched_pair_scores);

    matched_scores_in_cur.resize(curr_frm.frm_obs_.lg_keypts_.size(), -1.0);
    for (unsigned int i = 0; i < matched_idx_pairs.size(); ++i) {
        const auto matched_idx_pair = matched_idx_pairs.at(i);
        const auto idx_curr = curr_frm_idices_valid.at(matched_idx_pair.first);
        const auto idx_last = last_frm_idices_valid.at(matched_idx_pair.second);
        const auto& lm = last_frm.get_landmark(idx_last);

        // 3D point coordinates with the global reference
        const Vec3_t pos_w = lm->get_pos_in_world();

        // Reproject and compute visibility
        Vec2_t reproj;
        float x_right;
        const bool in_image = curr_frm.camera_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);

        // Ignore if it is reprojected outside the image
        if (!in_image) {
            continue;
        }

        // Acquire keypoints in the cell where the reprojected 3D points exist
        const unsigned int last_scale_level = 0;
        int min_level;
        int max_level;
        if (assume_forward) {
            min_level = last_scale_level;
            max_level = std::min(last_frm.orb_params_->num_levels_ - 1, last_scale_level + 1);
        }
        else if (assume_backward) {
            min_level = std::max(0, static_cast<int>(last_scale_level) - 1);
            max_level = last_scale_level;
        }
        else {
            min_level = std::max(0, static_cast<int>(last_scale_level) - 1);
            max_level = std::min(last_frm.orb_params_->num_levels_ - 1, last_scale_level + 1);
        }
        auto indices = curr_frm.get_keypoints_in_cell(reproj(0), reproj(1),
                                                      margin, //* curr_frm.orb_params_->scale_factors_.at(last_scale_level),
                                                      min_level, max_level);
        if (std::find(indices.begin(), indices.end(), idx_curr) == indices.end()) {
            continue;
        }

        curr_frm.add_landmark(lm, idx_curr);
        matched_scores_in_cur.at(idx_curr) = matched_pair_scores.at(i);
        ++num_matches;
    }

    return num_matches;
}

unsigned int lightglue::match_frame_and_keyframe(data::frame& frm, const std::shared_ptr<data::keyframe>& keyfrm,
                                                 std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_frm, std::vector<double>& matched_scores_in_frm,
                                                 bool use_fixed_seed) const {
    std::cout << "IN match::lightglue::match_frame_and_keyframe" << std::endl;

    // Initialization
    const auto num_frm_keypts = frm.frm_obs_.lg_keypts_.size();
    const auto keyfrm_lms = keyfrm->get_landmarks();
    unsigned int num_inlier_matches = 0;
    matched_lms_in_frm = std::vector<std::shared_ptr<data::landmark>>(num_frm_keypts, nullptr);
    matched_scores_in_frm = std::vector<double>(num_frm_keypts, 0.0);

    // Compute feature matching with LightGlue
    std::vector<std::pair<unsigned int, unsigned int>> matched_idx_pairs;
    std::vector<double> matched_scores;
    // brute_force_match(frm.frm_obs_, keyfrm, matches);
    lightglue_->image_match(frm.image_, keyfrm->image_, frm.frm_obs_.lg_keypts_, keyfrm->frm_obs_.lg_keypts_,
                            frm.frm_obs_.lg_descriptors_, keyfrm->frm_obs_.lg_descriptors_, matched_idx_pairs, matched_scores);

    for (unsigned int i = 0; i < matched_idx_pairs.size(); ++i) {
        const auto frm_idx = matched_idx_pairs.at(i).first;
        matched_scores_in_frm.at(frm_idx) = matched_scores.at(i);
    }
    frm.add_match_score(keyfrm->src_frm_id_, matched_scores_in_frm);

    cv::Mat img_matches;
    cv::vconcat(frm.image_, keyfrm->image_, img_matches);
    for (const auto& pair : matched_idx_pairs) {
        const auto idx_1 = pair.first;
        const auto idx_2 = pair.second;
        cv::circle(img_matches, frm.frm_obs_.lg_keypts_.at(idx_1), 2, cv::Scalar(255, 0, 0));
        cv::circle(img_matches, keyfrm->frm_obs_.lg_keypts_.at(idx_2) + cv::Point2f(0, frm.image_.rows), 2, cv::Scalar(255, 0, 0));
        cv::line(img_matches, frm.frm_obs_.lg_keypts_.at(idx_1), keyfrm->frm_obs_.lg_keypts_.at(idx_2) + cv::Point2f(0, frm.image_.rows), cv::Scalar(0, 255, 0));
    }
    cv::imwrite("matches.jpg", img_matches);

    std::vector<std::pair<int, int>> matches;
    keypoint_landmark_match(frm.frm_obs_, keyfrm, matched_idx_pairs, matches);

    cv::vconcat(frm.image_, keyfrm->image_, img_matches);
    for (const auto& pair : matches) {
        const auto idx_1 = pair.first;
        const auto idx_2 = pair.second;
        cv::circle(img_matches, frm.frm_obs_.lg_keypts_.at(idx_1), 2, cv::Scalar(255, 0, 0));
        cv::circle(img_matches, keyfrm->frm_obs_.lg_keypts_.at(idx_2) + cv::Point2f(0, frm.image_.rows), 2, cv::Scalar(255, 0, 0));
        cv::line(img_matches, frm.frm_obs_.lg_keypts_.at(idx_1), keyfrm->frm_obs_.lg_keypts_.at(idx_2) + cv::Point2f(0, frm.image_.rows), cv::Scalar(0, 255, 0));
    }
    cv::imwrite("matches_klmatch.jpg", img_matches);

    // Extract only inliers with RANSAC
    solve::essential_solver solver(frm.frm_obs_.lg_bearings_, keyfrm->frm_obs_.lg_bearings_, matches, use_fixed_seed);
    solver.find_via_ransac(1000, true);
    if (!solver.solution_is_valid()) {
        return 0;
    }
    const auto is_inlier_matches = solver.get_inlier_matches();

    // Save the information
    for (unsigned int i = 0; i < matches.size(); ++i) {
        if (!is_inlier_matches.at(i)) {
            continue;
        }
        const auto frm_idx = matches.at(i).first;
        const auto keyfrm_idx = matches.at(i).second;

        matched_lms_in_frm.at(frm_idx) = keyfrm_lms.at(keyfrm_idx);
        ++num_inlier_matches;
    }

    return num_inlier_matches;
}

unsigned int lightglue::keypoint_landmark_match(const data::frame_observation& frm_obs,
                                                const std::shared_ptr<data::keyframe>& keyfrm,
                                                std::vector<std::pair<unsigned int, unsigned int>> matched_idx_pairs,
                                                std::vector<std::pair<int, int>>& matches) const {
    unsigned int num_matches = 0;

    const auto num_lg_keypts_1 = frm_obs.lg_keypts_.size();
    const auto num_lg_keypts_2 = keyfrm->frm_obs_.lg_keypts_.size();
    const auto keypts_1 = frm_obs.lg_keypts_;
    const auto keypts_2 = keyfrm->frm_obs_.lg_keypts_;
    const auto lms_2 = keyfrm->get_landmarks();

    std::unordered_map<unsigned int, unsigned int> match_map_1_in_2;
    for (const auto& pair : matched_idx_pairs) {
        match_map_1_in_2[pair.second] = pair.first;
    }

    // Index 2 associated to each index 1
    auto matched_indices_2_in_1 = std::vector<int>(num_lg_keypts_1, -1);

    for (unsigned int idx_2 = 0; idx_2 < num_lg_keypts_2; ++idx_2) {
        // 3次元点が有効なもののみ対象にする
        const auto& lm_2 = lms_2.at(idx_2);
        if (!lm_2) {
            continue;
        }
        if (lm_2->will_be_erased()) {
            continue;
        }

        auto match_pair = match_map_1_in_2.find(idx_2);
        if (match_pair == match_map_1_in_2.end()) {
            continue;
        }

        matched_indices_2_in_1.at(match_pair->second) = idx_2;

        ++num_matches;
    }

    matches.clear();
    matches.reserve(num_matches);
    for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_1.size(); ++idx_1) {
        const auto idx_2 = matched_indices_2_in_1.at(idx_1);
        if (idx_2 < 0) {
            continue;
        }
        matches.emplace_back(std::make_pair(idx_1, idx_2));
    }

    return num_matches;
}

unsigned int lightglue::match_for_triangulation(const std::shared_ptr<data::keyframe>& keyfrm_1,
                                                const std::shared_ptr<data::keyframe>& keyfrm_2,
                                                const Mat33_t& E_12,
                                                std::vector<std::pair<unsigned int, unsigned int>>& matched_idx_pairs,
                                                std::vector<double>& matched_scores_in_keyfrm_1,
                                                const float residual_rad_thr) const {
    std::cout << "IN match::lightglue::match_for_triangulation; keyfrm_1: " << keyfrm_1->id_ << ", keyfrm_2: " << keyfrm_2->id_ << std::endl;
    unsigned int num_matches = 0;

    // Project the center of keyframe 1 to keyframe 2
    // to acquire the epipole coordinates of the candidate keyframe
    const Vec3_t cam_center_1 = keyfrm_1->get_trans_wc();
    const Mat33_t rot_2w = keyfrm_2->get_rot_cw();
    const Vec3_t trans_2w = keyfrm_2->get_trans_cw();
    Vec3_t epiplane_in_keyfrm_2;
    const bool valid_epiplane = keyfrm_2->camera_->reproject_to_bearing(rot_2w, trans_2w, cam_center_1, epiplane_in_keyfrm_2);

    // Acquire the 3D point information of the keframes
    const auto assoc_lms_in_keyfrm_1 = keyfrm_1->get_landmarks();
    const auto assoc_lms_in_keyfrm_2 = keyfrm_2->get_landmarks();
    const auto num_lg_keypts_1 = keyfrm_1->frm_obs_.lg_keypts_.size();
    const auto num_lg_keypts_2 = keyfrm_2->frm_obs_.lg_keypts_.size();

    std::vector<std::pair<unsigned int, unsigned int>> lg_matched_idx_pairs;
    std::vector<double> lg_matched_scores;
    lightglue_->image_match(keyfrm_1->image_, keyfrm_2->image_, keyfrm_1->frm_obs_.lg_keypts_, keyfrm_2->frm_obs_.lg_keypts_,
                            keyfrm_1->frm_obs_.lg_descriptors_, keyfrm_2->frm_obs_.lg_descriptors_, lg_matched_idx_pairs, lg_matched_scores);

    // Save the keypoint idx in keyframe 2 which is already associated to the keypoint idx in keyframe 1
    std::vector<int> matched_indices_2_in_keyfrm_1(num_lg_keypts_1, -1);
    matched_scores_in_keyfrm_1.resize(num_lg_keypts_1, -1.0);
    for (unsigned int i = 0; i < lg_matched_idx_pairs.size(); ++i) {
        const auto idx_1 = lg_matched_idx_pairs.at(i).first;
        const auto idx_2 = lg_matched_idx_pairs.at(i).second;
        matched_indices_2_in_keyfrm_1.at(idx_1) = idx_2;
        matched_scores_in_keyfrm_1.at(idx_1) = lg_matched_scores.at(i);
    }

    std::vector<int> inlier_indices_2_in_keyfrm_1(num_lg_keypts_1, -1);
    for (unsigned int idx_1 = 0; idx_1 < num_lg_keypts_1; ++idx_1) {
        if (matched_indices_2_in_keyfrm_1.at(idx_1) < 0) {
            continue;
        }
        const auto& lm_1 = assoc_lms_in_keyfrm_1.at(idx_1);
        // Ignore if the keypoint of keyframe is associated any 3D points
        if (lm_1) {
            continue;
        }
        const int idx_2 = matched_indices_2_in_keyfrm_1.at(idx_1);
        const auto& lm_2 = assoc_lms_in_keyfrm_2.at(idx_2);
        if (lm_2) {
            continue;
        }

        const Vec3_t& lg_bearing_1 = keyfrm_1->frm_obs_.lg_bearings_.at(idx_1);
        const Vec3_t& lg_bearing_2 = keyfrm_2->frm_obs_.lg_bearings_.at(idx_2);

        if (valid_epiplane) {
            // Do not use any keypoints near the epipole if both are not stereo keypoints
            const auto cos_dist = epiplane_in_keyfrm_2.dot(lg_bearing_2);
            // The threshold of the minimum angle formed by the epipole and the bearing vector is 3.0 degree
            constexpr double cos_dist_thr = 0.99862953475;

            // Do not allow to match if the formed angle is narrower that the threshold value
            if (cos_dist_thr < cos_dist) {
                continue;
            }
        }

        // Check consistency in Matrix E
        const bool is_inlier = check_epipolar_constraint(lg_bearing_1, lg_bearing_2, E_12,
                                                         residual_rad_thr,
                                                         1.0);
        if (is_inlier) {
            inlier_indices_2_in_keyfrm_1.at(idx_1) = idx_2;
            ++num_matches;
        }
    }

    matched_idx_pairs.clear();
    matched_idx_pairs.reserve(num_matches);

    bool is_a = false;

    for (unsigned int idx_1 = 0; idx_1 < inlier_indices_2_in_keyfrm_1.size(); ++idx_1) {
        if (inlier_indices_2_in_keyfrm_1.at(idx_1) < 0) {
            continue;
        }
        matched_idx_pairs.emplace_back(std::make_pair(idx_1, inlier_indices_2_in_keyfrm_1.at(idx_1)));
    }

    return num_matches;
}

template<typename T>
unsigned int lightglue::detect_duplication(const std::shared_ptr<data::keyframe>& keyfrm,
                                           const Mat33_t& rot_cw,
                                           const Vec3_t& trans_cw,
                                           const T& landmarks_to_check,
                                           const float margin,
                                           // std::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>>&, duplicated_lms_in_keyfrm,
                                           std::map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>, id_less<std::shared_ptr<data::landmark>>>& duplicated_lms_in_keyfrm,
                                           std::unordered_map<unsigned int, std::shared_ptr<data::landmark>>& new_connections,
                                           std::unordered_map<unsigned int, std::pair<std::shared_ptr<data::keyframe>, double>>& new_connections_score,
                                           bool do_reprojection_matching) const {
    std::cout << "IN match::lightglue::detect_duplication" << std::endl;
    const Vec3_t trans_wc = -rot_cw.transpose() * trans_cw;
    unsigned int num_fused = 0;
    std::unordered_set<unsigned int> already_matched_idx_in_keyfrm;

    duplicated_lms_in_keyfrm.clear();

    std::vector<std::shared_ptr<data::landmark>> lms_for_extraction;
    for (auto& lm : landmarks_to_check) {
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        if (lm->is_observed_in_keyframe(keyfrm)) {
            continue;
        }

        // 3D point coordinates with the global reference
        const Vec3_t pos_w = lm->get_pos_in_world();

        // Reproject and compute visibility
        Vec2_t reproj;
        float x_right;
        const bool in_image = keyfrm->camera_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);

        // Ignore if it is reprojected outside the image
        if (!in_image) {
            continue;
        }

        // Check if it's within ORB scale levels
        const Vec3_t cam_to_lm_vec = pos_w - trans_wc;
        const auto cam_to_lm_dist = cam_to_lm_vec.norm();

        // Compute the angle formed by the average vector of the 3D point observation,
        // and discard it if it is wider than the threshold value (60 degrees)
        const Vec3_t obs_mean_normal = lm->get_obs_mean_normal();

        if (cam_to_lm_vec.dot(obs_mean_normal) < 0.5 * cam_to_lm_dist) {
            continue;
        }

        lms_for_extraction.push_back(lm);
    }

    std::vector<std::shared_ptr<data::keyframe>> neighbors = keyfrm->graph_node_->get_top_n_covisibilities(20);
    keyfrm_lg_keypts_t keyfrm_lg_keypts;
    keyfrm_lg_descriptors_t keyfrm_lg_descriptors;

    extract_landmarks_with_keyframes(neighbors, lms_for_extraction, keyfrm_lg_keypts, keyfrm_lg_descriptors);

    for (const auto& pair : keyfrm_lg_keypts) {
        const auto lm_keyfrm = pair.first;
        const auto lms_and_lg_keypts = pair.second;
        const auto lms_and_lg_descriptors = keyfrm_lg_descriptors.at(lm_keyfrm);

        std::vector<cv::Point2f> lg_keypts;
        std::vector<std::vector<double>> lg_descriptors;
        std::vector<std::shared_ptr<data::landmark>> lms;
        for (unsigned int i = 0; i < lms_and_lg_keypts.size(); ++i) {
            lg_keypts.push_back(lms_and_lg_keypts.at(i).second);
            lg_descriptors.push_back(lms_and_lg_descriptors.at(i).second);
            lms.push_back(lms_and_lg_keypts.at(i).first);
        }

        if (lg_keypts.empty() || lg_descriptors.empty() || lms.empty()) {
            continue;
        }

        std::vector<std::pair<unsigned int, unsigned int>> matched_idx_pairs;
        std::vector<double> matched_scores;
        lightglue_->image_match(keyfrm->image_, lm_keyfrm->image_, keyfrm->frm_obs_.lg_keypts_, lg_keypts,
                                keyfrm->frm_obs_.lg_descriptors_, lg_descriptors, matched_idx_pairs, matched_scores);

        for (unsigned int i = 0; i < matched_idx_pairs.size(); ++i) {
            const auto idx_1 = matched_idx_pairs.at(i).first;
            const auto idx_2 = matched_idx_pairs.at(i).second;
            const auto lm = lms.at(idx_2);

            if (do_reprojection_matching) {
                const Vec3_t pos_w = lm->get_pos_in_world();
                Vec2_t reproj;
                float x_right;
                const bool in_image = keyfrm->camera_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);
                const auto lg_keypt = keyfrm->frm_obs_.lg_keypts_.at(idx_1);
                const auto scale_level = 0;
                if (!keyfrm->frm_obs_.stereo_x_right_.empty() && keyfrm->frm_obs_.stereo_x_right_.at(idx_1) >= 0) {
                    // Compute reprojection error with 3 degrees of freedom if a stereo match exists
                    const auto e_x = reproj(0) - lg_keypt.x;
                    const auto e_y = reproj(1) - lg_keypt.y;
                    const auto e_x_right = x_right - keyfrm->frm_obs_.stereo_x_right_.at(idx_1);
                    const auto reproj_error_sq = e_x * e_x + e_y * e_y + e_x_right * e_x_right;

                    // n=3
                    constexpr float chi_sq_3D = 7.81473;
                    if (chi_sq_3D < reproj_error_sq * keyfrm->orb_params_->inv_level_sigma_sq_.at(scale_level)) {
                        continue;
                    }
                }
                else {
                    // Compute reprojection error with 2 degrees of freedom if a stereo match does not exist
                    const auto e_x = reproj(0) - lg_keypt.x;
                    const auto e_y = reproj(1) - lg_keypt.y;
                    const auto reproj_error_sq = e_x * e_x + e_y * e_y;

                    // n=2
                    constexpr float chi_sq_2D = 5.99146;
                    if (chi_sq_2D < reproj_error_sq * keyfrm->orb_params_->inv_level_sigma_sq_.at(scale_level)) {
                        continue;
                    }
                }
            }

            auto lm_in_keyfrm = keyfrm->get_landmark(idx_1);
            if (lm_in_keyfrm) {
                // There is association between the 3D point and the keyframe
                // -> Duplication exists
                if (!lm_in_keyfrm->will_be_erased()) {
                    duplicated_lms_in_keyfrm[lm] = lm_in_keyfrm;
                }
            }
            else {
                // There is no association between the 3D point and the keyframe
                // Add the observation information
                new_connections.emplace(idx_1, lm);
                new_connections_score.emplace(idx_1, std::make_pair(lm_keyfrm, matched_scores.at(i)));
            }

            ++num_fused;
        }
    }

    return num_fused;
}

template unsigned int lightglue::detect_duplication(const std::shared_ptr<data::keyframe>&,
                                                    const Mat33_t&,
                                                    const Vec3_t&,
                                                    const std::vector<std::shared_ptr<data::landmark>>&,
                                                    const float,
                                                    // std::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>>&,
                                                    std::map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>, id_less<std::shared_ptr<data::landmark>>>&,
                                                    std::unordered_map<unsigned int, std::shared_ptr<data::landmark>>&,
                                                    std::unordered_map<unsigned int, std::pair<std::shared_ptr<data::keyframe>, double>>&,
                                                    bool) const;
template unsigned int lightglue::detect_duplication(const std::shared_ptr<data::keyframe>&,
                                                    const Mat33_t&,
                                                    const Vec3_t&,
                                                    const id_ordered_set<std::shared_ptr<data::landmark>>&,
                                                    const float,
                                                    // std::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>>&,
                                                    std::map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>, id_less<std::shared_ptr<data::landmark>>>&,
                                                    std::unordered_map<unsigned int, std::shared_ptr<data::landmark>>&,
                                                    std::unordered_map<unsigned int, std::pair<std::shared_ptr<data::keyframe>, double>>&,
                                                    bool) const;
template unsigned int lightglue::detect_duplication(const std::shared_ptr<data::keyframe>&,
                                                    const Mat33_t&,
                                                    const Vec3_t&,
                                                    const std::unordered_set<std::shared_ptr<data::landmark>>&,
                                                    const float,
                                                    // std::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>>&,
                                                    std::map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>, id_less<std::shared_ptr<data::landmark>>>&,
                                                    std::unordered_map<unsigned int, std::shared_ptr<data::landmark>>&,
                                                    std::unordered_map<unsigned int, std::pair<std::shared_ptr<data::keyframe>, double>>&,
                                                    bool) const;

unsigned int lightglue::match_frame_and_landmarks(data::frame& frm,
                                                  const std::vector<std::shared_ptr<data::landmark>>& local_landmarks,
                                                  eigen_alloc_unord_map<unsigned int, Vec2_t>& lm_to_reproj) const {
    unsigned int num_matches = 0;

    const auto neighbors = frm.ref_keyfrm_->graph_node_->get_top_n_covisibilities(10);

    keyfrm_lg_keypts_t keyfrm_lg_keypts;
    keyfrm_lg_descriptors_t keyfrm_lg_descriptors;

    std::vector<std::shared_ptr<data::landmark>> lms_for_extraction;
    for (auto local_lm : local_landmarks) {
        if (!lm_to_reproj.count(local_lm->id_)) {
            continue;
        }
        lms_for_extraction.push_back(local_lm);
    }

    extract_landmarks_with_keyframes(neighbors, lms_for_extraction, keyfrm_lg_keypts, keyfrm_lg_descriptors);

    // Match the current frame and the local landmarks for each keyframe
    for (const auto& pair : keyfrm_lg_keypts) {
        const auto keyfrm = pair.first;
        const auto lms_and_lg_keypts = pair.second;
        const auto lms_and_lg_descriptors = keyfrm_lg_descriptors.at(keyfrm);

        std::vector<cv::Point2f> lg_keypts;
        std::vector<std::vector<double>> lg_descriptors;
        std::vector<std::shared_ptr<data::landmark>> lms;
        for (unsigned int i = 0; i < lms_and_lg_keypts.size(); ++i) {
            lg_keypts.push_back(lms_and_lg_keypts.at(i).second);
            lg_descriptors.push_back(lms_and_lg_descriptors.at(i).second);
            lms.push_back(lms_and_lg_keypts.at(i).first);
        }

        if (lg_keypts.empty() || lg_descriptors.empty() || lms.empty()) {
            continue;
        }

        std::vector<std::pair<unsigned int, unsigned int>> matched_idx_pairs;
        std::vector<double> matched_scores;
        lightglue_->image_match(frm.image_, keyfrm->image_, frm.frm_obs_.lg_keypts_, lg_keypts,
                                frm.frm_obs_.lg_descriptors_, lg_descriptors, matched_idx_pairs, matched_scores);

        std::vector<double> matched_scores_in_frm(frm.frm_obs_.lg_keypts_.size(), -1.0);
        for (unsigned int i = 0; i < matched_idx_pairs.size(); ++i) {
            const auto idx_1 = matched_idx_pairs.at(i).first;
            const auto idx_2 = matched_idx_pairs.at(i).second;
            const auto lm = lms.at(idx_2);
            frm.add_landmark(lm, idx_1);
            matched_scores_in_frm.at(idx_1) = matched_scores.at(i);
            ++num_matches;
        }
        frm.add_match_score(keyfrm->src_frm_id_, matched_scores_in_frm);
    }

    return num_matches;
}

void lightglue::extract_landmarks_with_keyframes(const std::vector<std::shared_ptr<data::keyframe>>& neighbors,
                                                 const std::vector<std::shared_ptr<data::landmark>>& landmarks,
                                                 keyfrm_lg_keypts_t& keyfrm_lg_keypts, keyfrm_lg_descriptors_t& keyfrm_lg_descriptors) const {
    const auto neighbor_num = neighbors.size();

    for (auto lm : landmarks) {
        if (lm->will_be_erased()) {
            continue;
        }

        // Acquire local landmarks's LightGlue keypoints and descriptors
        const auto observed_lg_keypts = lm->get_lg_keypoints();
        const auto observed_lg_descriptors = lm->get_lg_descriptors();

        // Find the nearest local landmark's keyframe to the current frame's reference keyframe
        std::shared_ptr<data::keyframe> nearest_keyfrm = nullptr;
        unsigned int nearest_keyfrm_idx = neighbor_num;

        for (const auto& pair : observed_lg_keypts) {
            const auto keyfrm = pair.first;
            if (!keyfrm) {
                continue;
            }
            if (keyfrm->will_be_erased()) {
                continue;
            }

            for (unsigned int i = 0; i < neighbor_num; ++i) {
                const auto neighbor = neighbors.at(i);
                if (neighbor->id_ != keyfrm->id_) {
                    continue;
                }
                if (i < nearest_keyfrm_idx) {
                    nearest_keyfrm_idx = i;
                    nearest_keyfrm = neighbor;
                }
            }
        }

        if (!nearest_keyfrm) {
            continue;
        }

        keyfrm_lg_keypts[nearest_keyfrm].push_back(std::make_pair(lm, observed_lg_keypts.at(nearest_keyfrm)));
        keyfrm_lg_descriptors[nearest_keyfrm].push_back(std::make_pair(lm, observed_lg_descriptors.at(nearest_keyfrm)));
    }
}

} // namespace match
} // namespace stella_vslam
