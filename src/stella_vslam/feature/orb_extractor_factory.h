#ifndef STELLA_VSLAM_FEATURE_ORB_EXTRACTOR_FACTORY_H
#define STELLA_VSLAM_FEATURE_ORB_EXTRACTOR_FACTORY_H

#include "stella_vslam/feature/orb_extractor.h"
#include "stella_vslam/feature/orb_extractor_equirect.h"

namespace stella_vslam {

namespace feature {

class orb_extractor_factory {
public:
    static feature::orb_extractor* create(const orb_params* orb_params,
                                          const unsigned int min_area,
                                          camera::base* camera,
                                          const descriptor_type desc_type = descriptor_type::ORB,
                                          const std::vector<std::vector<float>>& mask_rects = {},
                                          const unsigned int division = 4) {
        feature::orb_extractor* extractor = nullptr;
        switch (camera->model_type_) {
            case camera::model_type_t::Equirectangular: {
                extractor = new feature::orb_extractor_equirect(orb_params, min_area, camera, desc_type, mask_rects, division);
                break;
            }
            default: {
                extractor = new feature::orb_extractor(orb_params, min_area, desc_type, mask_rects);
                break;
            }
        }

        return extractor;
    }
};

} // namespace feature
} // namespace stella_vslam

#endif // STELLA_VSLAM_FEATURE_ORB_EXTRACTOR_FACTORY_H
