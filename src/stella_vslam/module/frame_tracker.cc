#include "stella_vslam/camera/base.h"
#include "stella_vslam/data/frame.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/match/bow_tree.h"
#include "stella_vslam/match/projection.h"
#include "stella_vslam/match/robust.h"
#include "stella_vslam/module/frame_tracker.h"
#include "stella_vslam/optimize/pose_optimizer_g2o.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include <spdlog/spdlog.h>
#include <fstream>

namespace stella_vslam {
namespace module {

void write_matches_to_csv(const std::string& filename, unsigned int frm1_id, unsigned int frm2_id,
                          const data::frame& frm1, const data::frame& frm2,
                          const std::vector<bool>& outlier_flags,
                          const std::shared_ptr<data::keyframe>& keyfrm = nullptr) {
    // Open file in append mode
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        spdlog::warn("failed to open file: {}", filename);
        return;
    }

    // Check if file is empty to write header
    file.seekp(0, std::ios::end);
    bool is_empty = file.tellp() == 0;
    file.seekp(0, std::ios::end);

    if (is_empty) {
        file << "frm1_id,frm2_id,frm1_u,frm1_v,frm1_octave,frm1_response,frm2_u,frm2_v,frm2_octave,frm2_response,is_outlier\n";
    }

    // Write matching points
    for (unsigned int idx = 0; idx < frm1.frm_obs_.undist_keypts_.size(); ++idx) {
        const auto& lm = frm1.get_landmark(idx);
        if (lm == nullptr) {
            continue;
        }

        const auto& frm1_kp = frm1.frm_obs_.undist_keypts_.at(idx);
        int frm2_idx = -1;

        if (keyfrm != nullptr) {
            // For bow_match or robust_match based tracking
            frm2_idx = lm->get_index_in_keyframe(keyfrm);
            if (frm2_idx < 0 || static_cast<unsigned int>(frm2_idx) >= keyfrm->frm_obs_.undist_keypts_.size()) {
                continue;
            }
        } else {
            // For motion based tracking
            frm2_idx = frm2.get_landmark_idx(lm);
            if (frm2_idx < 0 || static_cast<unsigned int>(frm2_idx) >= frm2.frm_obs_.undist_keypts_.size()) {
                continue;
            }
        }

        const auto& frm2_kp = (keyfrm != nullptr) ? keyfrm->frm_obs_.undist_keypts_.at(frm2_idx) : frm2.frm_obs_.undist_keypts_.at(frm2_idx);
        const bool is_outlier = idx < outlier_flags.size() ? outlier_flags.at(idx) : false;
        file << frm1_id << "," << frm2_id << "," << frm1_kp.pt.x << "," << frm1_kp.pt.y << "," << frm1_kp.octave << "," << frm1_kp.response << "," << frm2_kp.pt.x << "," << frm2_kp.pt.y << "," << frm2_kp.octave << "," << frm2_kp.response << "," << (is_outlier ? 1 : 0) << "\n";
    }

    file.close();
}

frame_tracker::frame_tracker(camera::base* camera, const std::shared_ptr<optimize::pose_optimizer>& pose_optimizer,
                             const unsigned int num_matches_thr, bool use_fixed_seed, float margin)
    : camera_(camera), num_matches_thr_(num_matches_thr), use_fixed_seed_(use_fixed_seed), margin_(margin), pose_optimizer_(pose_optimizer) {}

bool frame_tracker::motion_based_track(data::frame& curr_frm, const data::frame& last_frm, const Mat44_t& velocity) const {
    match::projection projection_matcher(0.9, true);

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
        spdlog::debug("motion based tracking failed before optimization: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // Pose optimization
    Mat44_t optimized_pose;
    std::vector<bool> outlier_flags;
    pose_optimizer_->optimize(curr_frm, optimized_pose, outlier_flags);
    curr_frm.set_pose_cw(optimized_pose);

    constexpr auto matches_dir = "matches";
#ifdef _WIN32
    _mkdir(matches_dir);
    _mkdir(motion_based_tracking_dir);
#else
    if (mkdir(matches_dir, 0775) != 0 && errno != EEXIST) {
        spdlog::warn("failed to create directory {}", matches_dir);
    }
#endif

    // Write matches to CSV with outlier status (before outlier discard)
    write_matches_to_csv("matches/motion_based_tracking.csv", curr_frm.id_, last_frm.id_, curr_frm, last_frm, outlier_flags);

    // Discard the outliers
    const auto num_valid_matches = discard_outliers(outlier_flags, curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("motion based tracking failed after optimization: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        // spdlog::debug("motion based tracking succeeded: {} inlier matches >= {}", num_valid_matches, num_matches_thr_);
        return true;
    }
}

bool frame_tracker::bow_match_based_track(data::frame& curr_frm, const data::frame& last_frm, const std::shared_ptr<data::keyframe>& ref_keyfrm) const {
    match::bow_tree bow_matcher(0.7, true);

    // Search 2D-2D matches between the ref keyframes and the current frame
    // to acquire 2D-3D matches between the frame keypoints and 3D points observed in the ref keyframe
    std::vector<std::shared_ptr<data::landmark>> matched_lms_in_curr;
    auto num_matches = bow_matcher.match_frame_and_keyframe(ref_keyfrm, curr_frm, matched_lms_in_curr);

    if (num_matches < num_matches_thr_) {
        spdlog::debug("bow match based tracking failed before optimization: {} matches < {}", num_matches, num_matches_thr_);
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

    constexpr auto matches_dir = "matches";
#ifdef _WIN32
    _mkdir(matches_dir);
    _mkdir(bow_match_based_tracking_dir);
#else
    if (mkdir(matches_dir, 0775) != 0 && errno != EEXIST) {
        spdlog::warn("failed to create directory {}", matches_dir);
    }
#endif

    // Write matches to CSV with outlier status (before outlier discard)
    write_matches_to_csv("matches/bow_match_based_tracking.csv", curr_frm.id_, ref_keyfrm->src_frm_id_, curr_frm, last_frm, outlier_flags, ref_keyfrm);

    // Discard the outliers
    const auto num_valid_matches = discard_outliers(outlier_flags, curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("bow match based tracking failed after optimization: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        // spdlog::debug("bow match based tracking succeeded: {} inlier matches >= {}", num_valid_matches, num_matches_thr_);
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
        spdlog::debug("robust match based tracking failed before optimization: {} matches < {}", num_matches, num_matches_thr_);
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

    constexpr auto matches_dir = "matches";
#ifdef _WIN32
    _mkdir(matches_dir);
    _mkdir(robust_match_based_tracking_dir);
#else
    if (mkdir(matches_dir, 0775) != 0 && errno != EEXIST) {
        spdlog::warn("failed to create directory {}", matches_dir);
    }
#endif

    // Write matches to CSV with outlier status (before outlier discard)
    write_matches_to_csv("matches/robust_match_based_tracking.csv", curr_frm.id_, ref_keyfrm->src_frm_id_, curr_frm, last_frm, outlier_flags, ref_keyfrm);

    // Discard the outliers
    const auto num_valid_matches = discard_outliers(outlier_flags, curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("robust match based tracking failed after optimization: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        // spdlog::debug("robust match based tracking succeeded: {} inlier matches >= {}", num_valid_matches, num_matches_thr_);
        return true;
    }
}

unsigned int frame_tracker::discard_outliers(const std::vector<bool>& outlier_flags, data::frame& curr_frm) const {
    unsigned int num_valid_matches = 0;

    for (unsigned int idx = 0; idx < curr_frm.frm_obs_.undist_keypts_.size(); ++idx) {
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
