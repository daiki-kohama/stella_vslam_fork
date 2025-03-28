#ifndef STELLA_VSLAM_MATCH_LIGHTGLUE_H
#define STELLA_VSLAM_MATCH_LIGHTGLUE_H

#include "stella_vslam/match/base.h"
#include "stella_vslam/feature/lightglue.h"
#include "stella_vslam/type.h"

namespace feature {
class lightglue;
}

namespace stella_vslam {

namespace data {
class frame;
class keyframe;
} // namespace data

namespace match {

class lightglue final : public base {
public:
    lightglue(const float lowe_ratio, const bool check_orientation, const feature::lightglue* lightglue)
        : base(lowe_ratio, check_orientation), lightglue_(lightglue) {}

    ~lightglue() final = default;

    unsigned int match_frame_and_frame(data::frame& frm_1, data::frame& frm_2, std::vector<cv::Point2f>& prev_matched_pts,
                                       std::vector<int>& matched_indices_2_in_frm_, std::vector<double>& matched_scores_2_in_frm_1) const;

    unsigned int match_current_and_last_frames(data::frame& curr_frm, const data::frame& last_frm, const float margin, std::vector<double>& matched_scores_in_cur) const;

    unsigned int match_frame_and_keyframe(data::frame& frm, const std::shared_ptr<data::keyframe>& keyfrm,
                                          std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_frm, std::vector<double>& matched_scores_in_frm,
                                          bool use_fixed_seed) const;

    unsigned int match_for_triangulation(const std::shared_ptr<data::keyframe>& keyfrm_1,
                                         const std::shared_ptr<data::keyframe>& keyfrm_2,
                                         const Mat33_t& E_12,
                                         std::vector<std::pair<unsigned int, unsigned int>>& matched_idx_pairs,
                                         std::vector<double>& matched_scores_in_keyfrm_1,
                                         const float residual_rad_thr) const;

    //! 3次元点(landmarks_to_check)をkeyframeに再投影し，keyframeで観測している3次元点と重複しているものを探す
    template<typename T>
    unsigned int detect_duplication(const std::shared_ptr<data::keyframe>& keyfrm,
                                    const Mat33_t& rot_cw,
                                    const Vec3_t& trans_cw,
                                    const T& landmarks_to_check,
                                    const float margin,
                                    std::map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>, id_less<std::shared_ptr<data::landmark>>>& duplicated_lms_in_keyfrm,
                                    std::unordered_map<unsigned int, std::shared_ptr<data::landmark>>& new_connections,
                                    std::unordered_map<unsigned int, std::pair<std::shared_ptr<data::keyframe>, double>>& new_connections_score,
                                    bool do_reprojection_matching = false) const;

    unsigned int match_frame_and_landmarks(data::frame& frm,
                                           const std::vector<std::shared_ptr<data::landmark>>& local_landmarks,
                                           eigen_alloc_unord_map<unsigned int, Vec2_t>& lm_to_reproj) const;

private:
    unsigned int keypoint_landmark_match(const data::frame_observation& frm_obs,
                                         const std::shared_ptr<data::keyframe>& keyfrm,
                                         std::vector<std::pair<unsigned int, unsigned int>> matched_idx_pairs,
                                         std::vector<std::pair<int, int>>& matches) const;

    using keyfrm_lg_keypts_t = std::map<std::shared_ptr<data::keyframe>, std::vector<std::pair<std::shared_ptr<data::landmark>, cv::Point2f>>, id_less<std::shared_ptr<data::keyframe>>>;
    using keyfrm_lg_descriptors_t = std::map<std::shared_ptr<data::keyframe>, std::vector<std::pair<std::shared_ptr<data::landmark>, std::vector<double>>>, id_less<std::shared_ptr<data::keyframe>>>;
    void extract_landmarks_with_keyframes(const std::vector<std::shared_ptr<data::keyframe>>& neighbors,
                                          const std::vector<std::shared_ptr<data::landmark>>& landmarks,
                                          keyfrm_lg_keypts_t& keyfrm_lg_keypts, keyfrm_lg_descriptors_t& keyfrm_lg_descriptors) const;

    // LightGlue
    const feature::lightglue* lightglue_;
};

} // namespace match
} // namespace stella_vslam

#endif // STELLA_VSLAM_MATCH_LIGHTGLUE_H
