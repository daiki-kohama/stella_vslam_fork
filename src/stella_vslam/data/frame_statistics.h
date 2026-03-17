#ifndef STELLA_VSLAM_DATA_FRAME_STATISTICS_H
#define STELLA_VSLAM_DATA_FRAME_STATISTICS_H

#include "stella_vslam/type.h"

#include <vector>
#include <unordered_map>
#include <memory>
#include <map>
#include <nlohmann/json_fwd.hpp>

namespace stella_vslam {
namespace data {

class frame;
class keyframe;

struct summary_statistics {
    double mean_ = 0.0;
    double median_ = 0.0;
    bool valid_ = false;
};

struct frame_additional_statistics {
    unsigned int num_tracked_landmarks_ = 0;
    summary_statistics landmark_reproj_error_px_;
    summary_statistics landmark_parallax_deg_;
    double landmark_direction_variance_ = 0.0;
    bool has_landmark_direction_variance_ = false;
    double all_feature_direction_variance_ = 0.0;
    bool has_all_feature_direction_variance_ = false;
    summary_statistics landmark_feature_response_;
    summary_statistics all_feature_response_;
};

class frame_statistics {
public:
    /**
     * Constructor
     */
    frame_statistics() = default;

    /**
     * Destructor
     */
    virtual ~frame_statistics() = default;

    /**
     * Update frame statistics
     * @param frm
     * @param is_lost
     */
    void update_frame_statistics(const data::frame& frm, const bool is_lost);

    /**
     * Replace a keyframe which will be erased in frame statistics
     * @param old_keyfrm
     * @param new_keyfrm
     */
    void replace_reference_keyframe(const std::shared_ptr<data::keyframe>& old_keyfrm, const std::shared_ptr<data::keyframe>& new_keyfrm);

    /**
     * Get frame IDs of each of the reference keyframes
     * @return
     */
    std::unordered_map<std::shared_ptr<data::keyframe>, std::vector<unsigned int>> get_frame_id_of_reference_keyframes() const;

    /**
     * Get the number of the contained valid frames
     * @return
     */
    unsigned int get_num_valid_frames() const;

    /**
     * Get reference keyframes of each of the frames
     * @return
     */
    std::map<unsigned int, std::shared_ptr<data::keyframe>> get_reference_keyframes() const;

    /**
     * Get relative camera poses from the corresponding reference keyframes
     * @return
     */
    eigen_alloc_map<unsigned int, Mat44_t> get_relative_cam_poses() const;

    /**
     * Get timestamps
     * @return
     */
    std::map<unsigned int, double> get_timestamps() const;

    /**
     * Get lost frame flags
     * @return
     */
    std::map<unsigned int, bool> get_lost_frames() const;

    /**
     * Dump frames as JSON
     * @return
     */
    nlohmann::json to_json() const;

    /**
     * Clear frame statistics
     */
    void clear();

private:
    //! Reference keyframe, frame ID associated with the keyframe
    std::unordered_map<std::shared_ptr<data::keyframe>, std::vector<unsigned int>> frm_ids_of_ref_keyfrms_;

    //! Number of valid frames
    unsigned int num_valid_frms_ = 0;

    // Size of all the following variables is the number of frames
    //! Reference keyframes for each frame
    std::unordered_map<unsigned int, std::shared_ptr<data::keyframe>> ref_keyfrms_;
    //! Relative pose against reference keyframe for each frame
    eigen_alloc_unord_map<unsigned int, Mat44_t> rel_cam_poses_from_ref_keyfrms_;
    //! Timestamp for each frame
    std::unordered_map<unsigned int, double> timestamps_;
    //! Flag whether each frame is lost or not
    std::unordered_map<unsigned int, bool> is_lost_frms_;
    //! Additional per-frame metrics
    std::unordered_map<unsigned int, frame_additional_statistics> additional_stats_;
};

} // namespace data
} // namespace stella_vslam

#endif // STELLA_VSLAM_DATA_FRAME_STATISTICS_H
