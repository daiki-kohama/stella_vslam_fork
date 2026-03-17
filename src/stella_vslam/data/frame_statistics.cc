#include "stella_vslam/data/common.h"
#include "stella_vslam/data/frame.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/frame_statistics.h"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

double clamp_unit(const double value) {
    return std::max(-1.0, std::min(1.0, value));
}

stella_vslam::data::summary_statistics compute_summary_statistics(std::vector<double> values) {
    stella_vslam::data::summary_statistics stats;
    if (values.empty()) {
        return stats;
    }

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    stats.mean_ = sum / values.size();

    std::sort(values.begin(), values.end());
    const auto mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        stats.median_ = 0.5 * (values.at(mid - 1) + values.at(mid));
    }
    else {
        stats.median_ = values.at(mid);
    }

    stats.valid_ = true;
    return stats;
}

double compute_direction_variance(const std::vector<stella_vslam::Vec3_t>& unit_vectors, bool& valid) {
    valid = false;
    if (unit_vectors.empty()) {
        return 0.0;
    }

    stella_vslam::Vec3_t mean_vector = stella_vslam::Vec3_t::Zero();
    for (const auto& unit_vector : unit_vectors) {
        mean_vector += unit_vector;
    }
    mean_vector /= static_cast<double>(unit_vectors.size());

    valid = true;
    return 1.0 - mean_vector.norm();
}

nlohmann::json summary_statistics_to_json(const stella_vslam::data::summary_statistics& stats) {
    if (!stats.valid_) {
        return {
            {"mean", nullptr},
            {"median", nullptr},
        };
    }

    return {
        {"mean", stats.mean_},
        {"median", stats.median_},
    };
}

} // namespace

namespace stella_vslam {
namespace data {

void frame_statistics::update_frame_statistics(const data::frame& frm, const bool is_lost) {
    if (frm.pose_is_valid()) {
        const Mat44_t rel_cam_pose_from_ref_keyfrm = frm.get_pose_cw() * frm.ref_keyfrm_->get_pose_wc();

        frm_ids_of_ref_keyfrms_[frm.ref_keyfrm_].push_back(frm.id_);

        ++num_valid_frms_;
        assert(!ref_keyfrms_.count(frm.id_));
        ref_keyfrms_[frm.id_] = frm.ref_keyfrm_;
        assert(!rel_cam_poses_from_ref_keyfrms_.count(frm.id_));
        rel_cam_poses_from_ref_keyfrms_[frm.id_] = rel_cam_pose_from_ref_keyfrm;
        assert(!timestamps_.count(frm.id_));
        timestamps_[frm.id_] = frm.timestamp_;

        frame_additional_statistics additional_stats;
        std::vector<double> landmark_reproj_errors;
        std::vector<double> landmark_parallaxes_deg;
        std::vector<double> landmark_feature_responses;
        std::vector<double> all_feature_responses;
        std::vector<Vec3_t> landmark_directions;
        std::vector<Vec3_t> feature_directions;

        all_feature_responses.reserve(frm.frm_obs_.undist_keypts_.size());
        feature_directions.reserve(frm.frm_obs_.bearings_.size());
        for (const auto& keypt : frm.frm_obs_.undist_keypts_) {
            all_feature_responses.push_back(keypt.response);
        }
        for (const auto& bearing : frm.frm_obs_.bearings_) {
            const auto norm = bearing.norm();
            if (norm <= 0.0) {
                continue;
            }
            feature_directions.push_back(bearing / norm);
        }

        for (unsigned int idx = 0; idx < frm.frm_obs_.undist_keypts_.size(); ++idx) {
            const auto& lm = frm.get_landmark(idx);
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            // the observation has been considered as inlier in the pose optimization
            assert(lm->has_observation());
            // count up
            ++additional_stats.num_tracked_landmarks_;

            const auto pos_w = lm->get_pos_in_world();
            const Vec3_t cam_to_lm = pos_w - frm.get_trans_wc();
            const auto cam_to_lm_norm = cam_to_lm.norm();
            if (cam_to_lm_norm <= 0.0) {
                continue;
            }

            const Vec3_t lm_direction = cam_to_lm / cam_to_lm_norm;
            landmark_directions.push_back(lm_direction);

            const auto obs_mean_normal = lm->get_obs_mean_normal();
            const double parallax_deg = std::acos(clamp_unit(lm_direction.dot(obs_mean_normal))) * 180.0 / std::acos(-1.0);
            landmark_parallaxes_deg.push_back(parallax_deg);

            const auto& keypt = frm.frm_obs_.undist_keypts_.at(idx);
            landmark_feature_responses.push_back(keypt.response);

            Vec2_t reproj = Vec2_t::Zero();
            float reproj_x_right = -1.0f;
            if (frm.camera_->reproject_to_image(frm.get_rot_cw(), frm.get_trans_cw(), pos_w, reproj, reproj_x_right)) {
                const double dx = static_cast<double>(keypt.pt.x) - reproj(0);
                const double dy = static_cast<double>(keypt.pt.y) - reproj(1);
                double reproj_error = std::sqrt(dx * dx + dy * dy);

                if (!frm.frm_obs_.stereo_x_right_.empty()) {
                    const float observed_x_right = frm.frm_obs_.stereo_x_right_.at(idx);
                    if (0.0f <= observed_x_right && 0.0f <= reproj_x_right) {
                        const double dx_right = static_cast<double>(observed_x_right) - reproj_x_right;
                        reproj_error = std::sqrt(dx * dx + dy * dy + dx_right * dx_right);
                    }
                }

                landmark_reproj_errors.push_back(reproj_error);
            }
        }

        additional_stats.landmark_reproj_error_px_ = compute_summary_statistics(landmark_reproj_errors);
        additional_stats.landmark_parallax_deg_ = compute_summary_statistics(landmark_parallaxes_deg);
        additional_stats.landmark_feature_response_ = compute_summary_statistics(landmark_feature_responses);
        additional_stats.all_feature_response_ = compute_summary_statistics(all_feature_responses);
        additional_stats.landmark_direction_variance_ = compute_direction_variance(landmark_directions, additional_stats.has_landmark_direction_variance_);
        additional_stats.all_feature_direction_variance_ = compute_direction_variance(feature_directions, additional_stats.has_all_feature_direction_variance_);
        additional_stats_[frm.id_] = additional_stats;

        spdlog::info("frame {} stats: num_tracked_lms={}", frm.id_, additional_stats.num_tracked_landmarks_);
        if (additional_stats.landmark_reproj_error_px_.valid_) {
            spdlog::info("  landmark_reproj_error_px: mean={:.4f}, median={:.4f}",
                         additional_stats.landmark_reproj_error_px_.mean_,
                         additional_stats.landmark_reproj_error_px_.median_);
        }
        if (additional_stats.landmark_parallax_deg_.valid_) {
            spdlog::info("  landmark_parallax_deg: mean={:.4f}, median={:.4f}",
                         additional_stats.landmark_parallax_deg_.mean_,
                         additional_stats.landmark_parallax_deg_.median_);
        }
        if (additional_stats.has_landmark_direction_variance_) {
            spdlog::info("  landmark_direction_variance: {:.6f}", additional_stats.landmark_direction_variance_);
        }
        if (additional_stats.has_all_feature_direction_variance_) {
            spdlog::info("  all_feature_direction_variance: {:.6f}", additional_stats.all_feature_direction_variance_);
        }
        if (additional_stats.landmark_feature_response_.valid_) {
            spdlog::info("  landmark_feature_response: mean={:.4f}, median={:.4f}",
                         additional_stats.landmark_feature_response_.mean_,
                         additional_stats.landmark_feature_response_.median_);
        }
        if (additional_stats.all_feature_response_.valid_) {
            spdlog::info("  all_feature_response: mean={:.4f}, median={:.4f}",
                         additional_stats.all_feature_response_.mean_,
                         additional_stats.all_feature_response_.median_);
        }
    }

    assert(!is_lost_frms_.count(frm.id_));
    is_lost_frms_[frm.id_] = is_lost;
}

void frame_statistics::replace_reference_keyframe(const std::shared_ptr<data::keyframe>& old_keyfrm, const std::shared_ptr<data::keyframe>& new_keyfrm) {
    // Delete keyframes and update associations.

    assert(num_valid_frms_ == rel_cam_poses_from_ref_keyfrms_.size());
    assert(num_valid_frms_ == ref_keyfrms_.size());
    assert(num_valid_frms_ == timestamps_.size());
    assert(num_valid_frms_ <= is_lost_frms_.size());

    // Finish if no need to replace keyframes
    if (!frm_ids_of_ref_keyfrms_.count(old_keyfrm)) {
        return;
    }

    // Search frames referencing old_keyfrm which is to be deleted.
    const auto frm_ids = frm_ids_of_ref_keyfrms_.at(old_keyfrm);

    for (const auto frm_id : frm_ids) {
        assert(*ref_keyfrms_.at(frm_id) == *old_keyfrm);

        // Get pose and relative pose of the old keyframe
        const Mat44_t old_ref_cam_pose_cw = old_keyfrm->get_pose_cw();
        const Mat44_t old_rel_cam_pose_cr = rel_cam_poses_from_ref_keyfrms_.at(frm_id);

        // Replace pointer of the keyframe to new_keyfrm
        ref_keyfrms_.at(frm_id) = new_keyfrm;

        // Update relative pose
        const Mat44_t new_ref_cam_pose_cw = new_keyfrm->get_pose_cw();
        const Mat44_t new_rel_cam_pose_cr = old_rel_cam_pose_cr * old_ref_cam_pose_cw * new_ref_cam_pose_cw.inverse();
        rel_cam_poses_from_ref_keyfrms_.at(frm_id) = new_rel_cam_pose_cr;
    }

    // Update frames referencing new_keyfrm
    auto& new_frm_ids = frm_ids_of_ref_keyfrms_[new_keyfrm];
    new_frm_ids.insert(new_frm_ids.end(), frm_ids.begin(), frm_ids.end());
    // Remove frames referencing old_keyfrm
    frm_ids_of_ref_keyfrms_.erase(old_keyfrm);
}

std::unordered_map<std::shared_ptr<data::keyframe>, std::vector<unsigned int>> frame_statistics::get_frame_id_of_reference_keyframes() const {
    return frm_ids_of_ref_keyfrms_;
}

unsigned int frame_statistics::get_num_valid_frames() const {
    return num_valid_frms_;
}

std::map<unsigned int, std::shared_ptr<data::keyframe>> frame_statistics::get_reference_keyframes() const {
    return {ref_keyfrms_.begin(), ref_keyfrms_.end()};
}

eigen_alloc_map<unsigned int, Mat44_t> frame_statistics::get_relative_cam_poses() const {
    return {rel_cam_poses_from_ref_keyfrms_.begin(), rel_cam_poses_from_ref_keyfrms_.end()};
}

std::map<unsigned int, double> frame_statistics::get_timestamps() const {
    return {timestamps_.begin(), timestamps_.end()};
}

std::map<unsigned int, bool> frame_statistics::get_lost_frames() const {
    return {is_lost_frms_.begin(), is_lost_frms_.end()};
}

nlohmann::json frame_statistics::to_json() const {
    spdlog::info("encoding {} frame(s) to store", num_valid_frms_);
    std::map<std::string, nlohmann::json> frames;

    if (num_valid_frms_ == 0) {
        spdlog::warn("there are no valid frames, cannot dump frames");
        return frames;
    }

    const auto rk_itr_bgn = ref_keyfrms_.begin();
    const auto rc_itr_bgn = rel_cam_poses_from_ref_keyfrms_.begin();
    const auto rk_itr_end = ref_keyfrms_.end();
    const auto rc_itr_end = rel_cam_poses_from_ref_keyfrms_.end();
    auto rk_itr = rk_itr_bgn;
    auto rc_itr = rc_itr_bgn;

    int offset = rk_itr->first;
    unsigned int prev_frm_id = 0;
    for (unsigned int i = 0; i < num_valid_frms_; ++i, ++rk_itr, ++rc_itr) {
        // check frame ID
        assert(rk_itr->first == rc_itr->first);
        const auto frm_id = rk_itr->first;

        // check if the frame was lost or not
        if (is_lost_frms_.at(frm_id)) {
            spdlog::warn("frame {} was lost", frm_id);
            continue;
        }

        auto ref_keyfrm = rk_itr->second;
        const Mat44_t cam_pose_rw = ref_keyfrm->get_pose_cw();
        const Mat44_t rel_cam_pose_cr = rc_itr->second;

        const Mat44_t cam_pose_cw = rel_cam_pose_cr * cam_pose_rw;
        Mat44_t cam_pose_wc = util::converter::inverse_pose(cam_pose_cw);

        nlohmann::json frame_json = {
            {"ref_keyfrm_id", ref_keyfrm->id_},
            {"rot_cw", convert_rotation_to_json(cam_pose_cw.block<3, 3>(0, 0))},
            {"trans_cw", convert_translation_to_json(cam_pose_cw.block<3, 1>(0, 3))}};

        const auto additional_stats_itr = additional_stats_.find(frm_id);
        if (additional_stats_itr != additional_stats_.end()) {
            const auto& stats = additional_stats_itr->second;
            frame_json["num_tracked_landmarks"] = stats.num_tracked_landmarks_;
            frame_json["landmark_reproj_error_px"] = summary_statistics_to_json(stats.landmark_reproj_error_px_);
            frame_json["landmark_parallax_deg"] = summary_statistics_to_json(stats.landmark_parallax_deg_);
            frame_json["landmark_direction_variance"] = stats.has_landmark_direction_variance_
                                                            ? nlohmann::json(stats.landmark_direction_variance_)
                                                            : nlohmann::json(nullptr);
            frame_json["all_feature_direction_variance"] = stats.has_all_feature_direction_variance_
                                                              ? nlohmann::json(stats.all_feature_direction_variance_)
                                                              : nlohmann::json(nullptr);
            frame_json["landmark_feature_response"] = summary_statistics_to_json(stats.landmark_feature_response_);
            frame_json["all_feature_response"] = summary_statistics_to_json(stats.all_feature_response_);
        }

        frames[std::to_string(frm_id)] = std::move(frame_json);
    }

    if (rk_itr != rk_itr_end || rc_itr != rc_itr_end) {
        spdlog::error("the sizes of frame statistics are not matched");
    }

    return frames;
}

void frame_statistics::clear() {
    num_valid_frms_ = 0;
    frm_ids_of_ref_keyfrms_.clear();
    ref_keyfrms_.clear();
    rel_cam_poses_from_ref_keyfrms_.clear();
    timestamps_.clear();
    is_lost_frms_.clear();
    additional_stats_.clear();
}

} // namespace data
} // namespace stella_vslam
