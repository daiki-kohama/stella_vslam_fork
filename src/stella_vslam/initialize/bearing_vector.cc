#include "stella_vslam/data/frame.h"
#include "stella_vslam/initialize/bearing_vector.h"
#include "stella_vslam/solve/essential_solver.h"
#include "stella_vslam/solve/triangulator.h"

#include <spdlog/spdlog.h>

namespace stella_vslam {
namespace initialize {

bearing_vector::bearing_vector(const data::frame& ref_frm,
                               const unsigned int num_ransac_iters,
                               const unsigned int min_num_triangulated,
                               const unsigned int min_num_valid_pts,
                               const float parallax_deg_thr,
                               const float reproj_err_thr,
                               bool use_fixed_seed)
    : base(ref_frm, num_ransac_iters, min_num_triangulated, min_num_valid_pts, parallax_deg_thr, reproj_err_thr),
      use_fixed_seed_(use_fixed_seed), ref_dl_keypts_(ref_frm.frm_obs_.dl_keypts_), ref_dl_bearings_(ref_frm.frm_obs_.dl_bearings_) {
    spdlog::debug("CONSTRUCT: initialize::bearing_vector");
}

bearing_vector::~bearing_vector() {
    spdlog::debug("DESTRUCT: initialize::bearing_vector");
}

bool bearing_vector::initialize(const data::frame& cur_frm, const std::vector<int>& ref_matches_with_cur) {
    // set the current camera model
    cur_camera_ = cur_frm.camera_;
    // store the keypoints and bearings
    cur_dl_keypts_ = cur_frm.frm_obs_.dl_keypts_;
    cur_dl_bearings_ = cur_frm.frm_obs_.dl_bearings_;
    // align matching information
    ref_cur_matches_.clear();
    ref_cur_matches_.reserve(cur_frm.frm_obs_.dl_keypts_.size());
    for (unsigned int ref_idx = 0; ref_idx < ref_matches_with_cur.size(); ++ref_idx) {
        const auto cur_idx = ref_matches_with_cur.at(ref_idx);
        if (0 <= cur_idx) {
            ref_cur_matches_.emplace_back(std::make_pair(ref_idx, cur_idx));
        }
    }

    // compute an E matrix
    auto essential_solver = solve::essential_solver(ref_dl_bearings_, cur_dl_bearings_, ref_cur_matches_, use_fixed_seed_);
    essential_solver.find_via_ransac(num_ransac_iters_, false);

    // reconstruct map if the solution is valid
    if (essential_solver.solution_is_valid()) {
        const Mat33_t E_ref_to_cur = essential_solver.get_best_E_21();
        const auto is_inlier_match = essential_solver.get_inlier_matches();
        return reconstruct_with_E(E_ref_to_cur, is_inlier_match);
    }
    else {
        return false;
    }
}

bool bearing_vector::reconstruct_with_E(const Mat33_t& E_ref_to_cur, const std::vector<bool>& is_inlier_match) {
    // found the most plausible pose from the FOUR hypothesis computed from the E matrix

    // decompose the E matrix
    eigen_alloc_vector<Mat33_t> init_rots;
    eigen_alloc_vector<Vec3_t> init_transes;
    if (!solve::essential_solver::decompose(E_ref_to_cur, init_rots, init_transes)) {
        return false;
    }

    assert(init_rots.size() == 4);
    assert(init_transes.size() == 4);

    const auto pose_is_found = find_most_plausible_pose(init_rots, init_transes, is_inlier_match, false);
    if (!pose_is_found) {
        return false;
    }

    spdlog::info("initialization succeeded with E");
    return true;
}

unsigned int bearing_vector::triangulate(const Mat33_t& rot_ref_to_cur, const Vec3_t& trans_ref_to_cur,
                                         const std::vector<bool>& is_inlier_match, const bool depth_is_positive,
                                         eigen_alloc_vector<Vec3_t>& triangulated_pts,
                                         std::vector<bool>& is_triangulated,
                                         unsigned int& num_triangulated_pts,
                                         float& parallax_cos) {
    // = cos(0.5deg)
    constexpr float cos_parallax_thr = 0.99996192306;
    const float reproj_err_thr_sq = reproj_err_thr_ * reproj_err_thr_;

    // resize buffers according to the number of observed keypoints in the reference
    is_triangulated.resize(ref_dl_keypts_.size(), false);
    triangulated_pts.resize(ref_dl_keypts_.size());

    std::vector<float> cos_parallaxes;
    cos_parallaxes.reserve(ref_dl_keypts_.size());

    // camera centers
    const Vec3_t ref_cam_center = Vec3_t::Zero();
    const Vec3_t cur_cam_center = -rot_ref_to_cur.transpose() * trans_ref_to_cur;

    unsigned int num_valid_pts = 0;
    num_triangulated_pts = 0;

    // for each matching, triangulate a 3D point and compute a parallax and a reprojection error
    for (unsigned int i = 0; i < ref_cur_matches_.size(); ++i) {
        if (!is_inlier_match.at(i)) {
            continue;
        }

        const Vec3_t& ref_bearing = ref_dl_bearings_.at(ref_cur_matches_.at(i).first);
        const Vec3_t& cur_bearing = cur_dl_bearings_.at(ref_cur_matches_.at(i).second);

        const Vec3_t pos_c_in_ref = solve::triangulator::triangulate(ref_bearing, cur_bearing, rot_ref_to_cur, trans_ref_to_cur);

        if (!std::isfinite(pos_c_in_ref(0))
            || !std::isfinite(pos_c_in_ref(1))
            || !std::isfinite(pos_c_in_ref(2))) {
            continue;
        }

        // compute a parallax
        const Vec3_t ref_normal = pos_c_in_ref - ref_cam_center;
        const float ref_norm = ref_normal.norm();
        const Vec3_t cur_normal = pos_c_in_ref - cur_cam_center;
        const float cur_norm = cur_normal.norm();
        const float cos_parallax = ref_normal.dot(cur_normal) / (ref_norm * cur_norm);

        const bool parallax_is_small = cos_parallax_thr < cos_parallax;

        // reject if the 3D point is in front of the cameras
        if (depth_is_positive) {
            if (!parallax_is_small && pos_c_in_ref(2) <= 0) {
                continue;
            }
            const Vec3_t pos_c_in_cur = rot_ref_to_cur * pos_c_in_ref + trans_ref_to_cur;
            if (!parallax_is_small && pos_c_in_cur(2) <= 0) {
                continue;
            }
        }

        const auto& ref_lg_keypt = ref_dl_keypts_.at(ref_cur_matches_.at(i).first);
        const auto& cur_lg_keypt = cur_dl_keypts_.at(ref_cur_matches_.at(i).second);

        // compute a reprojection error in the reference
        Vec2_t reproj_in_ref;
        float x_right_in_ref;
        const auto is_valid_ref = ref_camera_->reproject_to_image(Mat33_t::Identity(), Vec3_t::Zero(), pos_c_in_ref,
                                                                  reproj_in_ref, x_right_in_ref);
        if (!parallax_is_small && !is_valid_ref) {
            continue;
        }

        const float ref_reproj_err_sq = (reproj_in_ref - ref_lg_keypt).squaredNorm();
        if (reproj_err_thr_sq < ref_reproj_err_sq) {
            continue;
        }

        // compute a reprojection error in the current
        Vec2_t reproj_in_cur;
        float x_right_in_cur;
        const auto is_valid_cur = cur_camera_->reproject_to_image(rot_ref_to_cur, trans_ref_to_cur, pos_c_in_ref,
                                                                  reproj_in_cur, x_right_in_cur);
        if (!parallax_is_small && !is_valid_cur) {
            continue;
        }
        const float cur_reproj_err_sq = (reproj_in_cur - cur_lg_keypt).squaredNorm();
        if (reproj_err_thr_sq < cur_reproj_err_sq) {
            continue;
        }

        // triangulation is valid
        ++num_valid_pts;
        cos_parallaxes.push_back(cos_parallax);

        if (!parallax_is_small) {
            // triangulated
            triangulated_pts.at(ref_cur_matches_.at(i).first) = pos_c_in_ref;
            is_triangulated.at(ref_cur_matches_.at(i).first) = true;
            num_triangulated_pts++;
        }
    }

    if (0 < num_valid_pts) {
        // return the 50th smallest parallax
        std::sort(cos_parallaxes.begin(), cos_parallaxes.end());
        const auto idx = std::min(50, static_cast<int>(cos_parallaxes.size() - 1));
        parallax_cos = cos_parallaxes.at(idx);
    }
    else {
        parallax_cos = 1.0;
    }

    return num_valid_pts;
}

} // namespace initialize
} // namespace stella_vslam
