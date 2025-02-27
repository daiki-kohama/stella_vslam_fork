#include "stella_vslam/data/frame.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/module/local_map_updater.h"

#include <spdlog/spdlog.h>

namespace stella_vslam {
namespace module {

local_map_updater::local_map_updater(const unsigned int max_num_local_keyfrms)
    : max_num_local_keyfrms_(max_num_local_keyfrms) {}

std::vector<std::shared_ptr<data::keyframe>> local_map_updater::get_local_keyframes() const {
    return local_keyfrms_;
}

std::vector<std::shared_ptr<data::landmark>> local_map_updater::get_local_landmarks() const {
    return local_lms_;
}

std::shared_ptr<data::keyframe> local_map_updater::get_nearest_covisibility() const {
    return nearest_covisibility_;
}

bool local_map_updater::acquire_local_map(const std::vector<std::shared_ptr<data::landmark>>& frm_lms,
                                          const unsigned int num_keypts,
                                          unsigned int keyframe_id_threshold) {
    // tracking_module.cc の update_local_map からの呼び出しでは keyframe_id_threshold は 0
    // 共通のキーフレームを持ったり、その二次接続を持つキーフレームを最大 max_num_local_keyfrms_ まで取得して、 local_keyfrms_ に格納
    const auto local_keyfrms_was_found = find_local_keyframes(frm_lms, num_keypts, keyframe_id_threshold);
    // local_keyfrms_ に格納されたキーフレームのランドマークで、フレームが観測していないランドマークを local_lms_ に格納
    const auto local_lms_was_found = find_local_landmarks(frm_lms, num_keypts);
    return local_keyfrms_was_found && local_lms_was_found;
}

bool local_map_updater::find_local_keyframes(const std::vector<std::shared_ptr<data::landmark>>& frm_lms,
                                             const unsigned int num_keypts,
                                             unsigned int keyframe_id_threshold) {
    // キーフレームごとのフレームと共有しているランドマークの数を取得
    const auto keyfrm_to_num_shared_lms = count_num_shared_lms(frm_lms, num_keypts, keyframe_id_threshold);
    if (keyfrm_to_num_shared_lms.empty()) {
        SPDLOG_TRACE("find_local_keyframes: empty");
        return false;
    }
    std::unordered_set<unsigned int> already_found_keyfrm_ids;
    // 最も共通するランドマーク数が多いキーフレームを記録し、一次接続(共通するランドマークを持つキーフレーム)を取得
    const auto first_local_keyfrms = find_first_local_keyframes(keyfrm_to_num_shared_lms, already_found_keyfrm_ids);
    // 一次接続と二次接続(一次接続の共視キーフレーム1つ、子ノード1つ、親ノード1つ)合わせて、 max_num_local_keyfrms_ 以下を満たすように、二次接続のキーフレームを取得
    const auto second_local_keyfrms = find_second_local_keyframes(first_local_keyfrms, already_found_keyfrm_ids);
    local_keyfrms_ = first_local_keyfrms;
    // second_local_keyfrms を local_keyfrms_ の後ろに追加
    std::copy(second_local_keyfrms.begin(), second_local_keyfrms.end(), std::back_inserter(local_keyfrms_));
    return true;
}

local_map_updater::keyframe_to_num_shared_lms_t local_map_updater::count_num_shared_lms(const std::vector<std::shared_ptr<data::landmark>>& frm_lms,
                                                                                        const unsigned int num_keypts,
                                                                                        unsigned int keyframe_id_threshold) const {
    // count the number of sharing landmarks between the current frame and each of the neighbor keyframes
    // key: keyframe, value: number of sharing landmarks
    keyframe_to_num_shared_lms_t keyfrm_to_num_shared_lms;
    for (unsigned int idx = 0; idx < num_keypts; ++idx) {
        auto& lm = frm_lms.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        const auto observations = lm->get_observations();
        for (auto obs : observations) {
            auto keyfrm = obs.first.lock();
            // tracking_module.cc の update_local_map からの呼び出しでは keyframe_id_threshold は 0 なので、常に false
            if (keyframe_id_threshold > 0 && keyfrm->id_ >= keyframe_id_threshold) {
                continue;
            }
            ++keyfrm_to_num_shared_lms[keyfrm];
        }
    }
    return keyfrm_to_num_shared_lms;
}

auto local_map_updater::find_first_local_keyframes(const keyframe_to_num_shared_lms_t& keyfrm_to_num_shared_lms,
                                                   std::unordered_set<unsigned int>& already_found_keyfrm_ids)
    -> std::vector<std::shared_ptr<data::keyframe>> {
    std::vector<std::shared_ptr<data::keyframe>> first_local_keyfrms;
    first_local_keyfrms.reserve(2 * keyfrm_to_num_shared_lms.size());

    unsigned int max_num_shared_lms = 0;
    for (auto& keyfrm_and_num_shared_lms : keyfrm_to_num_shared_lms) {
        const auto& keyfrm = keyfrm_and_num_shared_lms.first;
        const auto num_shared_lms = keyfrm_and_num_shared_lms.second;

        if (keyfrm->will_be_erased()) {
            continue;
        }

        first_local_keyfrms.push_back(keyfrm);

        // avoid duplication
        already_found_keyfrm_ids.insert(keyfrm->id_);

        // update the nearest keyframe
        if (max_num_shared_lms < num_shared_lms) {
            max_num_shared_lms = num_shared_lms;
            // 最も共通するランドマーク数が多いキーフレーム
            nearest_covisibility_ = keyfrm;
        }
    }

    return first_local_keyfrms;
}

auto local_map_updater::find_second_local_keyframes(const std::vector<std::shared_ptr<data::keyframe>>& first_local_keyframes,
                                                    std::unordered_set<unsigned int>& already_found_keyfrm_ids) const
    -> std::vector<std::shared_ptr<data::keyframe>> {
    std::vector<std::shared_ptr<data::keyframe>> second_local_keyfrms;
    second_local_keyfrms.reserve(4 * first_local_keyframes.size());

    // add the second-order keyframes to the local landmarks
    auto add_second_local_keyframe = [this, &second_local_keyfrms, &already_found_keyfrm_ids](const std::shared_ptr<data::keyframe>& keyfrm) {
        if (!keyfrm) {
            return false;
        }
        if (keyfrm->will_be_erased()) {
            return false;
        }
        // avoid duplication
        if (already_found_keyfrm_ids.count(keyfrm->id_)) {
            return false;
        }
        already_found_keyfrm_ids.insert(keyfrm->id_);
        second_local_keyfrms.push_back(keyfrm);
        return true;
    };
    for (auto iter = first_local_keyframes.cbegin(); iter != first_local_keyframes.cend(); ++iter) {
        // tracking_module.cc の update_local_map からの呼び出しでは max_num_local_keyfrms_ はデフォルトで 60
        if (max_num_local_keyfrms_ < first_local_keyframes.size() + second_local_keyfrms.size()) {
            // max_num_local_keyfrms_ よりも多くのキーフレームが見つかった場合、それ以上追加しない
            // first_local_keyframes が共通するランドマークの多い順とかになっていないので、 second_local_keyfrms に追加される際に接続のより小さいものが入るケースあり
            break;
        }

        const auto& keyfrm = *iter;

        // covisibilities of the neighbor keyframe
        // キーフレームの上位10個の共視キーフレームから1つだけ追加
        const auto neighbors = keyfrm->graph_node_->get_top_n_covisibilities(10);
        for (const auto& neighbor : neighbors) {
            if (add_second_local_keyframe(neighbor)) {
                break;
            }
        }

        // children of the spanning tree
        // キーフレームの子ノードから1つだけ追加
        const auto spanning_children = keyfrm->graph_node_->get_spanning_children();
        for (const auto& child : spanning_children) {
            if (add_second_local_keyframe(child)) {
                break;
            }
        }

        // parent of the spanning tree
        // キーフレームの親ノードを追加
        const auto& parent = keyfrm->graph_node_->get_spanning_parent();
        add_second_local_keyframe(parent);
    }

    return second_local_keyfrms;
}

bool local_map_updater::find_local_landmarks(const std::vector<std::shared_ptr<data::landmark>>& frm_lms,
                                             const unsigned int num_keypts) {
    local_lms_.clear();
    local_lms_.reserve(50 * local_keyfrms_.size());

    std::unordered_set<unsigned int> already_found_lms_ids;
    for (unsigned int idx = 0; idx < num_keypts; ++idx) {
        auto& lm = frm_lms.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        already_found_lms_ids.insert(lm->id_);
    }
    for (const auto& keyfrm : local_keyfrms_) {
        const auto& lms = keyfrm->get_landmarks();

        for (const auto& lm : lms) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            // avoid duplication
            if (already_found_lms_ids.count(lm->id_)) {
                continue;
            }
            already_found_lms_ids.insert(lm->id_);

            // フレームで観測されていないかつ、 local_keyframes_ に含まれるキーフレームで観測されているランドマークを local_lms_ に追加
            local_lms_.push_back(lm);
        }
    }

    return true;
}

} // namespace module
} // namespace stella_vslam
