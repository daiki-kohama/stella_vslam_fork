#include "stella_vslam/camera/base.h"
#include "stella_vslam/data/frame.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/match/bow_tree.h"
#include "stella_vslam/match/projection.h"
#include "stella_vslam/match/robust.h"
#include "stella_vslam/match/lightglue.h"
#include "stella_vslam/module/frame_tracker.h"
#include "stella_vslam/optimize/pose_optimizer_g2o.h"
#include "stella_vslam/feature/lightglue.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

namespace stella_vslam {
namespace module {

frame_tracker::frame_tracker(camera::base* camera, const std::shared_ptr<optimize::pose_optimizer>& pose_optimizer,
                             const unsigned int num_matches_thr, bool use_fixed_seed, float margin)
    : camera_(camera), num_matches_thr_(num_matches_thr), use_fixed_seed_(use_fixed_seed), margin_(margin), pose_optimizer_(pose_optimizer) {}

bool frame_tracker::motion_based_track(data::frame& curr_frm, const data::frame& last_frm, const Mat44_t& velocity) const {
    match::projection projection_matcher(0.9, camera_->model_type_ != camera::model_type_t::Equirectangular);

    // Set the initial pose by using the motion model
    curr_frm.set_pose_cw(velocity * last_frm.get_pose_cw());

    // Initialize the 2D-3D matches
    curr_frm.erase_landmarks();

    // Reproject the 3D points observed in the last frame and find 2D-3D matches
    auto num_matches = projection_matcher.match_current_and_last_frames(curr_frm, last_frm, margin_);

    if (num_matches < num_matches_thr_) {
        // Increment the margin, and search again
        curr_frm.erase_landmarks();
        num_matches = projection_matcher.match_current_and_last_frames(curr_frm, last_frm, 2 * margin_);
    }

    if (num_matches < num_matches_thr_) {
        spdlog::debug("motion based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // Pose optimization
    Mat44_t optimized_pose;
    std::vector<bool> outlier_flags;
    pose_optimizer_->optimize(curr_frm, optimized_pose, outlier_flags);
    curr_frm.set_pose_cw(optimized_pose);

    // Discard the outliers
    const auto num_valid_matches = discard_outliers(outlier_flags, curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("motion based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        return true;
    }
}

bool frame_tracker::bow_match_based_track(data::frame& curr_frm, const data::frame& last_frm, const std::shared_ptr<data::keyframe>& ref_keyfrm) const {
    match::bow_tree bow_matcher(0.7, camera_->model_type_ != camera::model_type_t::Equirectangular);

    // Search 2D-2D matches between the ref keyframes and the current frame
    // to acquire 2D-3D matches between the frame keypoints and 3D points observed in the ref keyframe
    std::vector<std::shared_ptr<data::landmark>> matched_lms_in_curr;
    auto num_matches = bow_matcher.match_frame_and_keyframe(ref_keyfrm, curr_frm, matched_lms_in_curr);

    if (num_matches < num_matches_thr_) {
        spdlog::debug("bow match based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // Update the 2D-3D matches
    curr_frm.set_landmarks(matched_lms_in_curr);

    // Pose optimization
    // The initial value is the pose of the previous frame
    curr_frm.set_pose_cw(last_frm.get_pose_cw());
    Mat44_t optimized_pose;
    std::vector<bool> outlier_flags;
    pose_optimizer_->optimize(curr_frm, optimized_pose, outlier_flags);
    curr_frm.set_pose_cw(optimized_pose);

    // Discard the outliers
    const auto num_valid_matches = discard_outliers(outlier_flags, curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("bow match based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        return true;
    }
}

bool frame_tracker::robust_match_based_track(data::frame& curr_frm, const data::frame& last_frm, const std::shared_ptr<data::keyframe>& ref_keyfrm) const {
    match::robust robust_matcher(0.8, true);

    // Search 2D-2D matches between the ref keyframes and the current frame
    // to acquire 2D-3D matches between the frame keypoints and 3D points observed in the ref keyframe
    std::vector<std::shared_ptr<data::landmark>> matched_lms_in_curr;
    auto num_matches = robust_matcher.match_frame_and_keyframe(curr_frm, ref_keyfrm, matched_lms_in_curr, use_fixed_seed_);

    if (num_matches < num_matches_thr_) {
        spdlog::debug("robust match based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // Update the 2D-3D matches
    curr_frm.set_landmarks(matched_lms_in_curr);

    // Pose optimization
    // The initial value is the pose of the previous frame
    curr_frm.set_pose_cw(last_frm.get_pose_cw());
    Mat44_t optimized_pose;
    std::vector<bool> outlier_flags;
    pose_optimizer_->optimize(curr_frm, optimized_pose, outlier_flags);
    curr_frm.set_pose_cw(optimized_pose);

    // Discard the outliers
    const auto num_valid_matches = discard_outliers(outlier_flags, curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("robust match based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        return true;
    }
}

bool frame_tracker::lightglue_frame_match_based_track(data::frame& curr_frm, const data::frame& last_frm, const Mat44_t& velocity, feature::lg_matcher* lg_matcher) const {
    // match::projection projection_matcher(0.9, camera_->model_type_ != camera::model_type_t::Equirectangular);
    match::lightglue lg_matching(0.8, false, lg_matcher);

    // Set the initial pose by using the motion model
    curr_frm.set_pose_cw(velocity * last_frm.get_pose_cw());

    // Initialize the 2D-3D matches
    curr_frm.erase_landmarks();

    // Reproject the 3D points observed in the last frame and find 2D-3D matches
    std::vector<float> matched_scores_in_curr;
    auto num_matches = lg_matching.match_current_and_last_frames(curr_frm, last_frm, margin_, matched_scores_in_curr);

    if (num_matches < num_matches_thr_) {
        // Increment the margin, and search again
        curr_frm.erase_landmarks();
        matched_scores_in_curr.clear();
        num_matches = lg_matching.match_current_and_last_frames(curr_frm, last_frm, 2 * margin_, matched_scores_in_curr);
    }

    if (num_matches < num_matches_thr_) {
        spdlog::debug("lightglue motion based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // Pose optimization
    Mat44_t optimized_pose;
    std::vector<bool> outlier_flags;
    pose_optimizer_->optimize_matched_scores(curr_frm, optimized_pose, outlier_flags, matched_scores_in_curr);
    curr_frm.set_pose_cw(optimized_pose);

    // Discard the outliers
    const auto num_valid_matches = discard_outliers(outlier_flags, curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("lightglue motion based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        return true;
    }
}

bool frame_tracker::lightglue_keyframe_match_based_track(data::frame& curr_frm, const data::frame& last_frm,
                                                         const std::shared_ptr<data::keyframe>& ref_keyfrm,
                                                         feature::lg_matcher* lg_matcher) const {
    match::lightglue lg_matching(0.8, true, lg_matcher);

    // Search 2D-2D matches between the ref keyframes and the current frame
    // to acquire 2D-3D matches between the frame keypoints and 3D points observed in the ref keyframe
    std::vector<std::shared_ptr<data::landmark>> matched_lms_in_curr;
    std::vector<float> matched_scores_in_curr;
    auto num_matches = lg_matching.match_frame_and_keyframe(curr_frm, ref_keyfrm, matched_lms_in_curr, matched_scores_in_curr, use_fixed_seed_);

    std::cout << "lg_matching.match_frame_and_keyframe(curr_frm.id_: " << curr_frm.id_ << ", ref_keyfrm.id_: " << ref_keyfrm->id_ << ", num_matches: " << num_matches << std::endl;

    {
        cv::Mat img = curr_frm.image_.clone();
        for (unsigned int idx = 0; idx < matched_lms_in_curr.size(); ++idx) {
            if (!matched_lms_in_curr.at(idx)) {
                continue;
            }
            cv::circle(img, curr_frm.frm_obs_.dl_keypts_.at(idx), 2, cv::Scalar(255, 0, 0), 2);
            cv::putText(img, std::to_string(matched_scores_in_curr.at(idx)), curr_frm.frm_obs_.dl_keypts_.at(idx), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }
        cv::imwrite("lg_match_track_lms_" + std::to_string(curr_frm.id_) + ".jpg", img);
    }

    if (num_matches < num_matches_thr_) {
        spdlog::debug("lightglue match based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // Update the 2D-3D matches
    curr_frm.set_landmarks(matched_lms_in_curr);

    // Pose optimization
    // The initial value is the pose of the previous frame
    curr_frm.set_pose_cw(last_frm.get_pose_cw());
    Mat44_t optimized_pose;
    std::vector<bool> outlier_flags;
    pose_optimizer_->optimize_matched_scores(curr_frm, optimized_pose, outlier_flags, matched_scores_in_curr);
    curr_frm.set_pose_cw(optimized_pose);

    // Discard the outliers
    const auto num_valid_matches = discard_outliers(outlier_flags, curr_frm);

    {
        cv::Mat img = curr_frm.image_.clone();
        const Mat33_t rot_cw = curr_frm.get_rot_cw();
        const Vec3_t trans_cw = curr_frm.get_trans_cw();
        for (const auto& lm : curr_frm.get_landmarks()) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }
            const Vec3_t pos_w = lm->get_pos_in_world();
            Vec2_t reproj;
            float x_right;
            curr_frm.camera_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);
            cv::circle(img, cv::Point2f(reproj(0), reproj(1)), 2, cv::Scalar(255, 0, 0), 2);
        }
        cv::imwrite("lg_match_track_succeed_lms_" + std::to_string(curr_frm.id_) + ".jpg", img);
    }

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("lightglue match based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        return true;
    }
}

unsigned int frame_tracker::discard_outliers(const std::vector<bool>& outlier_flags, data::frame& curr_frm) const {
    unsigned int num_valid_matches = 0;

    for (unsigned int idx = 0; idx < curr_frm.frm_obs_.dl_keypts_.size(); ++idx) {
        if (curr_frm.get_landmark(idx) == nullptr) {
            continue;
        }

        if (outlier_flags.at(idx)) {
            curr_frm.erase_landmark_with_index(idx);
        }
        else {
            ++num_valid_matches;
        }
    }

    return num_valid_matches;
}

} // namespace module
} // namespace stella_vslam
