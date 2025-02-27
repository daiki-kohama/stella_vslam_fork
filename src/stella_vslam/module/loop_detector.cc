#include "stella_vslam/data/bow_database.h"
#include "stella_vslam/data/bow_vocabulary.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/match/bow_tree.h"
#include "stella_vslam/match/projection.h"
#include "stella_vslam/match/robust.h"
#include "stella_vslam/module/loop_detector.h"
#include "stella_vslam/optimize/pose_optimizer_factory.h"
#include "stella_vslam/solve/pnp_solver.h"
#include "stella_vslam/util/converter.h"
#include "stella_vslam/util/fancy_index.h"

#include <spdlog/spdlog.h>

namespace stella_vslam {
namespace module {

loop_detector::loop_detector(data::bow_database* bow_db, data::bow_vocabulary* bow_vocab, const YAML::Node& yaml_node, const bool fix_scale_in_Sim3_estimation)
// camera_type が monocular なら fix_scale_in_Sim3_estimation = false
    : bow_db_(bow_db), bow_vocab_(bow_vocab), transform_optimizer_(fix_scale_in_Sim3_estimation), pose_optimizer_(optimize::pose_optimizer_factory::create(yaml_node)),
      loop_detector_is_enabled_(yaml_node["enabled"].as<bool>(true)),
      fix_scale_in_Sim3_estimation_(fix_scale_in_Sim3_estimation),
      num_final_matches_thr_(yaml_node["num_final_matches_threshold"].as<unsigned int>(40)),
      min_continuity_(yaml_node["min_continuity"].as<unsigned int>(3)),
      reject_by_graph_distance_(yaml_node["reject_by_graph_distance"].as<bool>(false)),
      min_distance_on_graph_(yaml_node["min_distance_on_graph"].as<unsigned int>(50)),
      num_matches_thr_(yaml_node["num_matches_thr"].as<unsigned int>(20)),
      num_matches_thr_brute_force_(yaml_node["num_matches_thr_robust_matcher"].as<unsigned int>(0)),
      num_optimized_inliers_thr_(yaml_node["num_optimized_inliers_thr"].as<unsigned int>(20)),
      top_n_covisibilities_to_search_(yaml_node["top_n_covisibilities_to_search"].as<unsigned int>(0)),
      use_fixed_seed_(yaml_node["use_fixed_seed"].as<bool>(false)) {
    spdlog::debug("CONSTRUCT: loop_detector");
}

void loop_detector::enable_loop_detector() {
    loop_detector_is_enabled_ = true;
}

void loop_detector::disable_loop_detector() {
    loop_detector_is_enabled_ = false;
}

bool loop_detector::is_enabled() const {
    return loop_detector_is_enabled_;
}

void loop_detector::set_current_keyframe(const std::shared_ptr<data::keyframe>& keyfrm) {
    cur_keyfrm_ = keyfrm;
}

bool loop_detector::detect_loop_candidates() {
    auto succeeded = detect_loop_candidates_impl();
    // register to the BoW database
    bow_db_->add_keyframe(cur_keyfrm_);
    return succeeded;
}

void loop_detector::add_loop_candidate(const std::shared_ptr<data::keyframe>& keyfrm) {
    loop_candidates_to_validate_.insert(keyfrm);
}

bool loop_detector::detect_loop_candidates_impl() {
    // if the loop detector is disabled or the loop has been corrected recently,
    // cannot perfrom the loop correction
    if (!loop_detector_is_enabled_ || cur_keyfrm_->id_ < prev_loop_correct_keyfrm_id_ + 10) {
        return false;
    }

    // 1. search loop candidates by inquiring to the BoW dataqbase

    // 1-1. before inquiring, compute the minimum score of BoW similarity between the current and each of the covisibilities

    // キーフレーム間でBoW類似度の最小値を計算
    const float min_score = compute_min_score_in_covisibilities(cur_keyfrm_);

    // 1-2. inquiring to the BoW database about the similar keyframe whose score is lower than min_score

    // Not searching near frames of query_keyframe
    std::set<std::shared_ptr<data::keyframe>> keyfrms_to_reject;
    if (!reject_by_graph_distance_) {
        // 現在のキーフレームとその接続しているキーフレームを除外
        keyfrms_to_reject = cur_keyfrm_->graph_node_->get_connected_keyframes();
        keyfrms_to_reject.insert(cur_keyfrm_);
    }
    else {
        std::vector<std::pair<std::shared_ptr<data::keyframe>, int>> targets;
        targets.emplace_back(cur_keyfrm_, 0);
        keyfrms_to_reject.insert(cur_keyfrm_);
        while (!targets.empty()) {
            auto keyfrm_distance_pair = targets.back();
            targets.pop_back();
            auto& keyfrm = keyfrm_distance_pair.first;
            auto& distance = keyfrm_distance_pair.second;
            if (distance + 1 < min_distance_on_graph_) {
                // search parent
                const auto parent = keyfrm->graph_node_->get_spanning_parent();
                if (parent && !static_cast<bool>(keyfrms_to_reject.count(parent))) {
                    keyfrms_to_reject.insert(parent);
                    targets.emplace_back(parent, distance + 1);
                }
                // search loop_edges
                for (const auto& node : keyfrm->graph_node_->get_loop_edges()) {
                    if (static_cast<bool>(keyfrms_to_reject.count(node))) {
                        continue;
                    }
                    keyfrms_to_reject.insert(node);
                    targets.emplace_back(node, distance + 1);
                }
                // search children
                for (const auto& child : keyfrm->graph_node_->get_spanning_children()) {
                    if (static_cast<bool>(keyfrms_to_reject.count(child))) {
                        continue;
                    }
                    keyfrms_to_reject.insert(child);
                    targets.emplace_back(child, distance + 1);
                }
            }
        }
    }

    // BoWをベースに、keyfrms_to_reject を除いて最初の候補を探索する
    const auto init_loop_candidates = bow_db_->acquire_keyframes(cur_keyfrm_->bow_vec_, min_score, keyfrms_to_reject);

    // 1-3. if no candidates are found, cannot perform the loop correction

    if (init_loop_candidates.empty()) {
        // clear the buffer because any candidates are not found
        cont_detected_keyfrm_sets_.clear();
        return false;
    }

    // 2. From now on, we treat each of the candidates as "keyframe set" in order to improve robustness of loop detection
    //    the number of each of the candidate keyframe sets that detected are counted every time when this member functions is called
    //    if the keyframe sets were detected at the previous call, it is contained in `cont_detected_keyfrm_sets_`
    //    (note that "match of two keyframe sets" means the intersection of the two sets is NOT empty)

    // cont_detected_keyfrm_sets_ の各 keyfrm_set について、候補の keyfrm_set (接続のあるキーフレーム群) と重複するキーフレームがある場合、cont_detected_keyfrm_sets を更新
    // 候補の keyfrm_set からみて重複するキーフレームがない場合、新しい keyfrm_set を追加
    const auto curr_cont_detected_keyfrm_sets = find_continuously_detected_keyframe_sets(cont_detected_keyfrm_sets_, init_loop_candidates);

    // 3. if the number of the detection is equal of greater than the threshold (`min_continuity_`),
    //    adopt it as one of the loop candidates

    loop_candidates_to_validate_.clear();
    for (auto& curr : curr_cont_detected_keyfrm_sets) {
        const auto candidate_keyfrm = curr.lead_keyfrm_;
        const auto continuity = curr.continuity_;
        // check if the number of the detection is equal of greater than the threshold
        // 局所大域的(接続フレーム分の大域性あり)に連続して min_continuity_ (デフォルトで3) 回以上検出された場合、curr.lead_keyfrm_ を候補として採用
        // 連続とは言いつつも、curr.lead_keyfrm_ は変わっていく気がするので、これでいいのか？
        // -> 現在のキーフレームと接続するものは除いているから大丈夫
        if (min_continuity_ <= continuity) {
            // adopt as the candidates
            loop_candidates_to_validate_.insert(candidate_keyfrm);
        }
    }

    // 4. Update the members for the next call of this function

    cont_detected_keyfrm_sets_ = curr_cont_detected_keyfrm_sets;

    // 5. Add top n covisibilities to the candidates

    // top_n_covisibilities_to_search_ はデフォルトで0
    // 0より大きい場合、候補に上位 top_n_covisibilities_to_search_ 個の共視キーフレームを追加。ただし、keyfrms_to_reject に含まれるものは除外
    if (top_n_covisibilities_to_search_ > 0) {
        auto candidates = loop_candidates_to_validate_;
        for (auto& keyfrm : candidates) {
            auto covisibilities = keyfrm->graph_node_->get_top_n_covisibilities(top_n_covisibilities_to_search_);
            for (const auto& covisibility : covisibilities) {
                if (!static_cast<bool>(keyfrms_to_reject.count(covisibility))) {
                    loop_candidates_to_validate_.insert(covisibility);
                }
            }
        }
    }

    // return any candidate is found or not
    return !loop_candidates_to_validate_.empty();
}

bool loop_detector::validate_candidates() {
    // disallow the removal of the candidates
    for (const auto& candidate : loop_candidates_to_validate_) {
        // 削除されないフラグを立てる
        candidate->set_not_to_be_erased();
    }

    auto succeeded = validate_candidates_impl();
    if (succeeded) {
        // allow the removal of the candidates except for the selected one
        for (const auto& loop_candidate : loop_candidates_to_validate_) {
            if (*loop_candidate == *selected_candidate_) {
                continue;
            }
            loop_candidate->set_to_be_erased();
        }
    }
    else {
        // allow the removal of all of the candidates
        for (const auto& loop_candidate : loop_candidates_to_validate_) {
            loop_candidate->set_to_be_erased();
        }
    }
    return succeeded;
}

bool loop_detector::validate_candidates_impl() {
    // 1. for each of the candidates, estimate and validate the Sim3 between it and the current keyframe using the observed landmarks
    //    then, select ONE candaite

    const bool candidate_is_found = select_loop_candidate_via_Sim3(loop_candidates_to_validate_, selected_candidate_,
                                                                   g2o_Sim3_world_to_curr_, curr_match_lms_observed_in_cand_);
    Sim3_world_to_curr_ = util::converter::to_eigen_mat(g2o_Sim3_world_to_curr_);

    if (!candidate_is_found) {
        return false;
    }

    spdlog::debug("detect loop candidate via Sim3 estimation: keyframe {} - keyframe {}", selected_candidate_->id_, cur_keyfrm_->id_);

    // 2. reproject the landmarks observed in covisibilities of the selected candidate to the current keyframe,
    //    then acquire the extra 2D-3D matches

    // matches between the keypoints in the current and the landmarks observed in the covisibilities of the selected candidate
    // 現在のキーフレームの特徴点と選択された候補の共視キーフレームのランドマークのマッチング
    curr_match_lms_observed_in_cand_covis_.clear();

    auto cand_covisibilities = selected_candidate_->graph_node_->get_covisibilities();
    cand_covisibilities.push_back(selected_candidate_);

    // acquire all of the landmarks observed in the covisibilities of the candidate
    // check the already inserted landmarks
    std::unordered_set<std::shared_ptr<data::landmark>> already_inserted;
    for (const auto& covisibility : cand_covisibilities) {
        const auto lms_in_covisibility = covisibility->get_landmarks();
        for (const auto& lm : lms_in_covisibility) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            if (already_inserted.count(lm)) {
                // 重複するランドマークはスキップ
                continue;
            }
            curr_match_lms_observed_in_cand_covis_.push_back(lm);
            already_inserted.insert(lm);
        }
    }

    // reproject the landmarks observed in the covisibilities of the candidate to the current keyframe using Sim3 `Sim3_world_to_curr_`,
    // then, acquire the extra 2D-3D matches
    // however, landmarks in `curr_match_lms_observed_in_cand_` are already matched with keypoints in the current keyframe,
    // thus they are excluded from the reprojection
    match::projection projection_matcher(0.75);
    // 候補キーフレームと共視関係にあるキーフレームのランドマークも含めて、現在のキーフレームに投影し、ORB特徴量で追加のマッチングを行う
    projection_matcher.match_by_Sim3_transform(cur_keyfrm_, Sim3_world_to_curr_, curr_match_lms_observed_in_cand_covis_,
                                               curr_match_lms_observed_in_cand_, 10);

    // count up the matches
    unsigned int num_final_matches = 0;
    for (const auto& curr_assoc_lm_in_cand : curr_match_lms_observed_in_cand_) {
        if (curr_assoc_lm_in_cand) {
            ++num_final_matches;
        }
    }

    spdlog::debug("acquired {} matches after projection-match", num_final_matches);

    if (num_final_matches_thr_ <= num_final_matches) {
        return true;
    }
    else {
        spdlog::debug("destruct loop candidate because enough matches not acquired (< {})", num_final_matches_thr_);
        return false;
    }
}

float loop_detector::compute_min_score_in_covisibilities(const std::shared_ptr<data::keyframe>& keyfrm) const {
    // the maximum of score is 1.0
    float min_score = 1.0;

    // search the mininum score among covisibilities
    const auto covisibilities = keyfrm->graph_node_->get_covisibilities();
    const auto& bow_vec_1 = keyfrm->bow_vec_;
    for (const auto& covisibility : covisibilities) {
        if (covisibility->will_be_erased()) {
            continue;
        }
        const auto& bow_vec_2 = covisibility->bow_vec_;

        const auto score = data::bow_vocabulary_util::score(bow_vocab_, bow_vec_1, bow_vec_2);
        if (score < min_score) {
            min_score = score;
        }
    }

    return min_score;
}

keyframe_sets loop_detector::find_continuously_detected_keyframe_sets(const keyframe_sets& prev_cont_detected_keyfrm_sets,
                                                                      const std::vector<std::shared_ptr<data::keyframe>>& keyfrms_to_search) const {
    // count up the number of the detection of each of the keyframe sets

    // buffer to store continuity and keyframe set
    keyframe_sets curr_cont_detected_keyfrm_sets;

    // check the already counted keyframe sets to prevent from counting the same set twice
    std::map<std::set<std::shared_ptr<data::keyframe>>, bool> already_checked;
    for (const auto& prev : prev_cont_detected_keyfrm_sets) {
        already_checked[prev.keyfrm_set_] = false;
    }

    for (const auto& keyfrm_to_search : keyfrms_to_search) {
        // enlarge the candidate to the "keyframe set"
        // 接続のあるキーフレームを取得
        const auto keyfrm_set = keyfrm_to_search->graph_node_->get_connected_keyframes();

        // check if the initialization of the buffer is needed or not
        bool initialization_is_needed = true;

        // check continuity for each of the previously detected keyframe set
        for (const auto& prev : prev_cont_detected_keyfrm_sets) {
            // prev.keyfrm_set_: keyframe set
            // prev.lead_keyfrm_: the leader keyframe of the set
            // prev.continuity_: continuity

            // check if the keyframe set is already counted or not
            if (already_checked.at(prev.keyfrm_set_)) {
                continue;
            }

            // compute intersection between the previous set and the current set, then check if it is empty or not
            // 両方に属するキーフレームがなければ、continue
            if (prev.intersection_is_empty(keyfrm_set)) {
                continue;
            }

            // initialization is not needed because any candidate is found
            initialization_is_needed = false;

            // create the new statistics by incrementing the continuity
            // keyfrm_set, keyfrm_to_search を更新して、continuityを+1して追加
            const auto curr_continuity = prev.continuity_ + 1;
            curr_cont_detected_keyfrm_sets.emplace_back(
                keyframe_set{keyfrm_set, keyfrm_to_search, curr_continuity});

            // this keyframe set is already checked
            // 一度チェックしたら、もうチェックされない
            already_checked.at(prev.keyfrm_set_) = true;
        }

        // if initialization is needed, add the new statistics
        if (initialization_is_needed) {
            curr_cont_detected_keyfrm_sets.emplace_back(
                keyframe_set{keyfrm_set, keyfrm_to_search, 0});
        }
    }

    return curr_cont_detected_keyfrm_sets;
}

bool loop_detector::select_loop_candidate_via_Sim3(const std::unordered_set<std::shared_ptr<data::keyframe>>& loop_candidates,
                                                   std::shared_ptr<data::keyframe>& selected_candidate,
                                                   g2o::Sim3& g2o_Sim3_world_to_curr,
                                                   std::vector<std::shared_ptr<data::landmark>>& curr_match_lms_observed_in_cand) const {
    // estimate and the Sim3 between the current keyframe and each of the candidates using the observed landmarks
    // the Sim3 is estimated both in linear and non-linear ways
    // if the inlier after the estimation is lower than the threshold, discard tha candidate

    match::robust robust_matcher(0.75, false);
    match::bow_tree bow_matcher(0.75, false);
    match::projection projection_matcher(0.75, false);

    for (const auto& candidate : loop_candidates) {
        if (candidate->will_be_erased()) {
            continue;
        }

        // estimate the matches between the keypoints in the current keyframe and the landmarks observed in the candidate
        curr_match_lms_observed_in_cand.clear();
        // ORB特徴量ベースでマッチするランドマークのインデックスを curr_match_lms_observed_in_cand に記録
        const auto num_matches = bow_matcher.match_keyframes(cur_keyfrm_, candidate, curr_match_lms_observed_in_cand);

        // check the threshold
        // num_matches_thr_ はデフォルトで20
        if (num_matches < num_matches_thr_) {
            continue;
        }

        spdlog::debug("Checking if the loop candidate is appropriate: keyframe {} - keyframe {} (num_matches: {})", candidate->id_, cur_keyfrm_->id_, num_matches);

        // num_matches_thr_brute_force_ はデフォルトで0
        if (num_matches_thr_brute_force_ > 0) {
            // Look for more correspondence over more time
            const auto num_matches_brute_force = robust_matcher.match_keyframes(cur_keyfrm_, candidate, curr_match_lms_observed_in_cand, false);

            spdlog::debug("num_matches_brute_force: {}", num_matches_brute_force);

            if (num_matches_brute_force < num_matches_thr_brute_force_) {
                continue;
            }
        }

        // 現在のキーフレームでマッチした有効なランドマークのインデックス
        std::vector<unsigned int> valid_indices;
        valid_indices.reserve(curr_match_lms_observed_in_cand.size());
        for (unsigned int idx = 0; idx < curr_match_lms_observed_in_cand.size(); ++idx) {
            auto lm = curr_match_lms_observed_in_cand.at(idx);
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }
            valid_indices.push_back(idx);
        }

        // Resample valid elements
        // マッチしたランドマークの方向ベクトル
        const auto valid_bearings = util::resample_by_indices(cur_keyfrm_->frm_obs_.bearings_, valid_indices);
        // マッチしたランドマークのキーポイント
        const auto valid_keypts = util::resample_by_indices(cur_keyfrm_->frm_obs_.undist_keypts_, valid_indices);
        // マッチしたランドマークの特徴点のオクターブ(スケールレベル)
        std::vector<int> octaves(valid_indices.size());
        for (unsigned int i = 0; i < valid_indices.size(); ++i) {
            octaves.at(i) = valid_keypts.at(i).octave;
        }
        // マッチした候補側の(共通する)ランドマーク
        const auto valid_assoc_lms = util::resample_by_indices(curr_match_lms_observed_in_cand, valid_indices);
        // マッチした候補側の(共通する)ランドマークの3D座標
        eigen_alloc_vector<Vec3_t> valid_points(valid_indices.size());
        for (unsigned int i = 0; i < valid_indices.size(); ++i) {
            valid_points.at(i) = valid_assoc_lms.at(i)->get_pos_in_world();
        }
        // Setup PnP solver
        auto pnp_solver = std::unique_ptr<solve::pnp_solver>(new solve::pnp_solver(valid_bearings, octaves, valid_points,
                                                                                   cur_keyfrm_->orb_params_->scale_factors_,
                                                                                   10, use_fixed_seed_));

        // EPnP と RANSAC を用いて、マッチしたランドマークに対する現在のキーフレームのカメラポーズを推定
        pnp_solver->find_via_ransac(30, false);
        if (!pnp_solver->solution_is_valid()) {
            spdlog::debug("solution is not valid.");
            continue;
        }

        // RANSAC において inlier とされたランドマークのインデックス
        const auto inlier_indices = util::resample_by_indices(valid_indices, pnp_solver->get_inlier_flags());

        // Set 2D-3D matches for the pose optimization
        auto lms_in_cand = std::vector<std::shared_ptr<data::landmark>>(cur_keyfrm_->frm_obs_.num_keypts_, nullptr);
        for (const auto idx : inlier_indices) {
            // Set only the valid 3D points to the current frame
            lms_in_cand.at(idx) = curr_match_lms_observed_in_cand.at(idx);
        }
        // inlier のランドマークのみにする
        curr_match_lms_observed_in_cand = lms_in_cand;

        // Pose optimization
        std::vector<bool> outlier_flags;
        Mat44_t optimized_pose;
        // g2oで、inlier となったランドマークに対する現在のキーフレームのカメラポーズを最適化して optimized_pose, outlier_flags に記録し inlier の数を返す
        auto num_valid_obs = pose_optimizer_->optimize(pnp_solver->get_best_cam_pose(), cur_keyfrm_->frm_obs_, cur_keyfrm_->orb_params_, cur_keyfrm_->camera_,
                                                       curr_match_lms_observed_in_cand, optimized_pose, outlier_flags);

        // Discard the candidate if the number of the inliers is less than the threshold
        const int min_num_matches_after_pose_optimize = 10;
        // 最適化後の inlier の数が min_num_matches_after_pose_optimize(=10) 未満の場合、候補を破棄
        if (num_valid_obs < min_num_matches_after_pose_optimize) {
            spdlog::debug("1. Number of inliers ({}) < threshold ({})", num_valid_obs, min_num_matches_after_pose_optimize);
            continue;
        }

        // Reject outliers
        // lms_in_cand で outlier とされたランドマークを nullptr にする
        for (unsigned int idx = 0; idx < cur_keyfrm_->frm_obs_.num_keypts_; idx++) {
            if (!outlier_flags.at(idx)) {
                continue;
            }
            lms_in_cand.at(idx) = nullptr;
        }

        // 最適化時に inlier とされたランドマーク
        std::set<std::shared_ptr<data::landmark>> already_found_landmarks;
        for (const auto idx : inlier_indices) {
            if (outlier_flags.at(idx)) {
                continue;
            }
            // Record the 3D points already associated to the frame keypoints
            already_found_landmarks.insert(curr_match_lms_observed_in_cand.at(idx));
        }

        // Projection match based on the pre-optimized camera pose
        // ループ候補のキーフレームのランドマークを現在のキーフレームに投影して、特徴点とマッチしたランドマークを curr_match_lms_observed_in_cand に追加
        auto num_found = projection_matcher.match_frame_and_keyframe(optimized_pose, cur_keyfrm_->camera_, cur_keyfrm_->frm_obs_,
                                                                     cur_keyfrm_->orb_params_, curr_match_lms_observed_in_cand,
                                                                     candidate, already_found_landmarks, 10, 100);
        // Discard the candidate if the number of the inliers is less than the threshold
        const unsigned int min_num_valid_obs1 = 25;
        // 元々マッチしてたランドマーク数と、新たにマッチしたランドマーク数の合計が25未満なら破棄
        if (already_found_landmarks.size() + num_found < min_num_valid_obs1) {
            spdlog::debug("2. Number of matches ({}) < threshold ({})",
                          already_found_landmarks.size() + num_found, min_num_valid_obs1);
            continue;
        }

        Mat44_t optimized_pose1;
        std::vector<bool> outlier_flags1;
        // もう一度、g2oで最適化して、optimized_pose1, outlier_flags1 に記録し inlier の数を返す
        auto num_valid_obs1 = pose_optimizer_->optimize(optimized_pose,
                                                        cur_keyfrm_->frm_obs_, cur_keyfrm_->orb_params_, cur_keyfrm_->camera_,
                                                        curr_match_lms_observed_in_cand, optimized_pose1, outlier_flags1);

        // 最適化後の inlier の数が min_num_valid_obs1(=25) 未満の場合、候補を破棄
        if (num_valid_obs1 < min_num_valid_obs1) {
            spdlog::debug("2. Number of inliers ({}) < threshold ({})", num_valid_obs1, min_num_valid_obs1);
            continue;
        }

        // Exclude the already-associated landmarks
        std::set<std::shared_ptr<data::landmark>> already_found_landmarks1;
        for (unsigned int idx = 0; idx < cur_keyfrm_->frm_obs_.num_keypts_; ++idx) {
            if (!curr_match_lms_observed_in_cand.at(idx)) {
                continue;
            }
            already_found_landmarks1.insert(curr_match_lms_observed_in_cand.at(idx));
        }
        // Apply projection match again, then set the 2D-3D matches
        // g2oで再最適化した姿勢を使って、再度投影マッチングを行い、curr_match_lms_observed_in_cand に追加
        auto num_additional = projection_matcher.match_frame_and_keyframe(optimized_pose1, cur_keyfrm_->camera_, cur_keyfrm_->frm_obs_,
                                                                          cur_keyfrm_->orb_params_, curr_match_lms_observed_in_cand,
                                                                          candidate, already_found_landmarks, 3, 64);

        const unsigned int min_num_valid_obs2 = 40;
        // Discard if the number of the observations is less than the threshold
        // 再投影の前にマッチしてたランドマーク数と、新たにマッチしたランドマーク数の合計が40未満なら破棄
        if (num_valid_obs1 + num_additional < min_num_valid_obs2) {
            spdlog::debug("3. Number of matches ({}) < threshold ({})", num_valid_obs1 + num_additional, min_num_valid_obs2);
            // 他の候補は考えずに return しちゃう?
            return false;
        }

        // Perform optimization again
        Mat44_t optimized_pose2;
        std::vector<bool> outlier_flags2;
        // ランドマークを増やして、g2o で再々最適化
        auto num_valid_obs2 = pose_optimizer_->optimize(optimized_pose1,
                                                        cur_keyfrm_->frm_obs_, cur_keyfrm_->orb_params_, cur_keyfrm_->camera_,
                                                        curr_match_lms_observed_in_cand, optimized_pose2, outlier_flags2);

        // Discard if falling below the threshold
        // 再々最適化時の onlier の数が40未満なら破棄
        if (num_valid_obs2 < min_num_valid_obs2) {
            spdlog::debug("3. Number of inliers ({}) < threshold ({})", num_valid_obs2, min_num_valid_obs2);
            // 他の候補は考えずに return しちゃう?
            return false;
        }

        // Reject outliers
        for (unsigned int idx = 0; idx < cur_keyfrm_->frm_obs_.num_keypts_; ++idx) {
            if (!outlier_flags2.at(idx)) {
                continue;
            }
            // outlier のランドマークを nullptr にする
            curr_match_lms_observed_in_cand.at(idx) = nullptr;
        }

        const Mat44_t pose_1w_in_cand = optimized_pose2;
        const Mat33_t rot_1w_in_cand = pose_1w_in_cand.block<3, 3>(0, 0);
        const Vec3_t trans_1w_in_cand = pose_1w_in_cand.block<3, 1>(0, 3);
        auto lms_curr = cur_keyfrm_->get_landmarks();
        std::vector<float> scales;
        for (unsigned int idx = 0; idx < lms_curr.size(); ++idx) {
            // 現在のキーフレームで観測したランドマーク
            auto& lm_curr = lms_curr.at(idx);
            // ループ候補のキーフレームで観測したランドマーク
            auto& lm_cand = curr_match_lms_observed_in_cand.at(idx);
            if (!lm_cand || !lm_curr) {
                continue;
            }
            if (lm_cand->will_be_erased() || lm_curr->will_be_erased()) {
                continue;
            }
            const Vec3_t pos_w_lm_cand = lm_cand->get_pos_in_world();
            const Vec3_t pos_w_lm_curr = lm_curr->get_pos_in_world();
            // 最適化後のカメラポーズで、ループ候補のキーフレームで観測したランドマークを、現在のキーフレーム座標系に変換
            const Vec3_t pos_1_in_cand = rot_1w_in_cand * pos_w_lm_cand + trans_1w_in_cand;
            // 最適化前のカメラポーズで、現在のキーフレームで観測したランドマークを、現在のキーフレーム座標系に変換
            const Vec3_t pos_1_in_curr = cur_keyfrm_->get_rot_cw() * pos_w_lm_curr + cur_keyfrm_->get_trans_cw();
            const float norm_pos_1_in_cand = pos_1_in_cand.norm();
            const float norm_pos_1_in_curr = pos_1_in_curr.norm();
            // 最適化前の現在のキーフレームのカメラ位置から見たランドマークと、最適化後にマッチしたランドマークを見た時の視差
            const float cos_parallax = pos_1_in_cand.dot(pos_1_in_curr) / (norm_pos_1_in_cand * norm_pos_1_in_curr);
            // = cos(0.5deg)
            constexpr float cos_parallax_thr = 0.99996192306;
            const bool parallax_is_small = cos_parallax_thr < cos_parallax;
            // 視差が0.5度以上であればcontinue
            if (!parallax_is_small) {
                continue;
            }
            // スケールを算出するのに用いる
            scales.push_back(norm_pos_1_in_curr / norm_pos_1_in_cand);
        }
        if (scales.size() < 1) {
            spdlog::debug("not enough scale references {}", scales.size());
            continue;
        }
        // ループ候補キーフレームの座標系から最適化後の現在のキーフレーム座標系に変換する行列
        const Mat33_t rot_12 = rot_1w_in_cand * candidate->get_rot_cw().transpose();
        const Vec3_t trans_12 = -rot_12 * candidate->get_trans_cw() + trans_1w_in_cand;
        std::sort(scales.begin(), scales.end());
        // scales の中央値
        const float scale_12 = scales[(scales.size() - 1) / 2];

        // perforn non-linear optimization of the estimated Sim3

        // マッチしていないランドマークについて、両方向からマッチングを行い、相互参照していれば、curr_match_lms_observed_in_cand に追加
        projection_matcher.match_keyframes_mutually(cur_keyfrm_, candidate, curr_match_lms_observed_in_cand,
                                                    scale_12, rot_12, trans_12, 7.5);

        // ループ候補キーフレームの座標系から最適化後の現在のキーフレーム座標系に変換する行列(スケールを含む)
        g2o::Sim3 g2o_sim3_12(rot_12, trans_12, scale_12);
        // 現在のキーフレームからみたランドマークをループ候補のキーフレームから見た際と、その逆の際の両方で、再投影誤差を最適化して、 g2o_sim3_12 に記録し、最適化後の inlier の数を返す
        const auto num_optimized_inliers = transform_optimizer_.optimize(cur_keyfrm_, candidate, curr_match_lms_observed_in_cand,
                                                                         g2o_sim3_12, 10);

        // check the threshold
        // num_optimized_inliers_thr_ はデフォルトで20
        if (num_optimized_inliers < num_optimized_inliers_thr_) {
            continue;
        }

        spdlog::debug("found loop candidate via nonlinear Sim3 optimization: keyframe {} - keyframe {} (num_optimized_inliers: {})", candidate->id_, cur_keyfrm_->id_, num_optimized_inliers);

        selected_candidate = candidate;
        // convert the estimated Sim3 from "candidate -> current" to "world -> current"
        // this Sim3 indicates the correct camera pose oof the current keyframe after loop correction
        // 世界座標系から、ループ候補キーフレームの座標系を経由して、最適化後の現在のキーフレーム座標系への変換行列
        g2o_Sim3_world_to_curr = g2o_sim3_12 * g2o::Sim3(candidate->get_rot_cw(), candidate->get_trans_cw(), 1.0);

        return true;
    }

    return false;
}

std::shared_ptr<data::keyframe> loop_detector::get_selected_candidate_keyframe() const {
    return selected_candidate_;
}

g2o::Sim3 loop_detector::get_Sim3_world_to_current() const {
    return g2o_Sim3_world_to_curr_;
}

std::vector<std::shared_ptr<data::landmark>> loop_detector::current_matched_landmarks_observed_in_candidate() const {
    return curr_match_lms_observed_in_cand_;
}

std::vector<std::shared_ptr<data::landmark>> loop_detector::current_matched_landmarks_observed_in_candidate_covisibilities() const {
    return curr_match_lms_observed_in_cand_covis_;
}

void loop_detector::set_loop_correct_keyframe_id(const unsigned int loop_correct_keyfrm_id) {
    prev_loop_correct_keyfrm_id_ = loop_correct_keyfrm_id;
}

} // namespace module
} // namespace stella_vslam
