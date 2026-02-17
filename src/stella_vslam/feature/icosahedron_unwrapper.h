#ifndef STELLA_VSLAM_FEATURE_ICOSAHEDRON_UNWRAPPER_H
#define STELLA_VSLAM_FEATURE_ICOSAHEDRON_UNWRAPPER_H

#include <opencv2/opencv.hpp>
#include <array>
#include <vector>

namespace stella_vslam {
namespace eqr_ico {

struct FaceOutput {
    cv::Mat image;                    // BGR
    cv::Mat mask;                     // 8U (0/255)
    std::array<cv::Point2f, 3> triPx; // triangle vertices in output pixel coords
};

struct FaceRemap {
    cv::Mat mapX;                     // CV_32FC1
    cv::Mat mapY;                     // CV_32FC1
    cv::Mat mask;                     // 8U (0/255)
    std::array<cv::Point2f, 3> triPx; // triangle vertices in output pixel coords
    cv::Size size;                    // output image size
    cv::Vec3d basisX;                 // gnomonic camera basis x-axis
    cv::Vec3d basisY;                 // gnomonic camera basis y-axis
    cv::Vec3d basisZ;                 // gnomonic camera basis z-axis
    double planeMinX = 0.0;           // gnomonic bbox min x
    double planeMinY = 0.0;           // gnomonic bbox min y
    int marginPx = 0;                 // output image margin in pixels
    double pixelsPerUnit = 0.0;       // pixel density on gnomonic plane
};

template <typename T>
inline const T& clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

class IcosahedronUnwrapper {
public:
    IcosahedronUnwrapper(const cv::Size& srcSize, int marginPx = 16, double pixelsPerUnit = 1200.0);

    void rebuild(const cv::Size& srcSize, int marginPx, double pixelsPerUnit);

    int faceCount() const;

    FaceOutput applyFace(const cv::Mat& srcEqrBGR, int faceIdx) const;

    std::vector<FaceOutput> applyAll(const cv::Mat& srcEqrBGR) const;

    cv::Point2f facePixelToEquirectangularPixel(int faceIdx, const cv::Point2f& facePx) const;

    std::vector<cv::Mat> getFaceMasks() const {
        std::vector<cv::Mat> masks;
        masks.reserve(remaps_.size());
        for (const auto& remap : remaps_) {
            masks.push_back(remap.mask);
        }
        return masks;
    }

    static constexpr unsigned int face_count_ = 20;

private:
    struct Icosahedron {
        std::vector<cv::Vec3d> V;          // 12 vertices on unit sphere
        std::vector<std::array<int, 3>> F; // 20 faces (indices)
    };

    static cv::Vec3d normalize3(const cv::Vec3d& v);
    static double wrap01(double x);
    static Icosahedron buildIcosahedron();
    static void makeCameraBasis(const cv::Vec3d& z, cv::Vec3d& x, cv::Vec3d& y);
    static cv::Point2d projectGnomonic(const cv::Vec3d& v, const cv::Vec3d& x, const cv::Vec3d& y, const cv::Vec3d& z);
    static FaceRemap buildFaceRemap(const Icosahedron& ico, int faceIdx, int marginPx, double pixelsPerUnit, const cv::Size& srcSize);

private:
    Icosahedron ico_;
    cv::Size srcSize_;
    int marginPx_ = 16;
    double pixelsPerUnit_ = 1200.0;
    std::vector<FaceRemap> remaps_;
};

} // namespace eqr_ico
} // namespace stella_vslam

#endif // STELLA_VSLAM_FEATURE_ICOSAHEDRON_UNWRAPPER_H
