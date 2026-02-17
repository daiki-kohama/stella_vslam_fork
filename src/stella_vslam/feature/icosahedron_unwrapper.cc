#include "stella_vslam/feature/icosahedron_unwrapper.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace stella_vslam {
namespace eqr_ico {

cv::Vec3d IcosahedronUnwrapper::normalize3(const cv::Vec3d& v) {
    double n = std::sqrt(v.dot(v));
    return (n > 0) ? (v / n) : cv::Vec3d(0, 0, 0);
}

double IcosahedronUnwrapper::wrap01(double x) {
    x = x - std::floor(x);
    return x;
}

IcosahedronUnwrapper::Icosahedron IcosahedronUnwrapper::buildIcosahedron() {
    Icosahedron ico;
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;

    std::vector<cv::Vec3d> verts = {
        {-1, phi, 0}, {1, phi, 0}, {-1, -phi, 0}, {1, -phi, 0}, {0, -1, phi}, {0, 1, phi}, {0, -1, -phi}, {0, 1, -phi}, {phi, 0, -1}, {phi, 0, 1}, {-phi, 0, -1}, {-phi, 0, 1}};

    ico.V.reserve(12);
    for (auto& v : verts) {
        ico.V.push_back(normalize3(v));
    }

    ico.F = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};

    return ico;
}

void IcosahedronUnwrapper::makeCameraBasis(const cv::Vec3d& z, cv::Vec3d& x, cv::Vec3d& y) {
    cv::Vec3d up(0, 1, 0);
    if (std::abs(z.dot(up)) > 0.95) {
        up = cv::Vec3d(0, 0, 1);
    }
    x = normalize3(up.cross(z));
    y = z.cross(x);  // - をつけると自然な向きになるが、左手系になってしまう
}

cv::Point2d IcosahedronUnwrapper::projectGnomonic(const cv::Vec3d& v, const cv::Vec3d& x, const cv::Vec3d& y, const cv::Vec3d& z) {
    double dz = v.dot(z);
    return cv::Point2d(v.dot(x) / dz, v.dot(y) / dz);
}

FaceRemap IcosahedronUnwrapper::buildFaceRemap(
    const Icosahedron& ico,
    int faceIdx,
    int marginPx,
    double pixelsPerUnit,
    const cv::Size& srcSize) {
    const auto f = ico.F.at(faceIdx);
    const cv::Vec3d v0 = ico.V[f[0]];
    const cv::Vec3d v1 = ico.V[f[1]];
    const cv::Vec3d v2 = ico.V[f[2]];

    cv::Vec3d z = normalize3(v0 + v1 + v2);
    cv::Vec3d x, y;
    makeCameraBasis(z, x, y);

    cv::Point2d p0 = projectGnomonic(v0, x, y, z);
    cv::Point2d p1 = projectGnomonic(v1, x, y, z);
    cv::Point2d p2 = projectGnomonic(v2, x, y, z);

    double minx = std::min({p0.x, p1.x, p2.x});
    double maxx = std::max({p0.x, p1.x, p2.x});
    double miny = std::min({p0.y, p1.y, p2.y});
    double maxy = std::max({p0.y, p1.y, p2.y});

    int w = (int)std::ceil((maxx - minx) * pixelsPerUnit) + 2 * marginPx;
    int h = (int)std::ceil((maxy - miny) * pixelsPerUnit) + 2 * marginPx;
    w = std::max(w, 2);
    h = std::max(h, 2);

    cv::Mat mask(h, w, CV_8UC1, cv::Scalar(0));
    cv::Mat mapX(h, w, CV_32FC1);
    cv::Mat mapY(h, w, CV_32FC1);

    auto planeFromPixel = [&](int px, int py) -> cv::Point2d {
        double ux = minx + ((double)(px - marginPx)) / pixelsPerUnit;
        double uy = miny + ((double)(py - marginPx)) / pixelsPerUnit;
        return {ux, uy};
    };

    for (int py = 0; py < h; ++py) {
        for (int px = 0; px < w; ++px) {
            cv::Point2d u = planeFromPixel(px, py);
            cv::Vec3d dir = normalize3(u.x * x + u.y * y + 1.0 * z);

            double lon = std::atan2(dir[0], dir[2]);
            double lat = std::asin(clamp(dir[1], -1.0, 1.0));

            double fx = (lon + CV_PI) / (2.0 * CV_PI);
            double fy = (CV_PI / 2.0 - lat) / CV_PI;

            mapX.at<float>(py, px) = static_cast<float>(wrap01(fx) * srcSize.width);
            mapY.at<float>(py, px) = static_cast<float>(clamp(fy * srcSize.height, 0.0, (double)(srcSize.height - 1)));
        }
    }

    auto toPix = [&](const cv::Point2d& p) -> cv::Point2f {
        float px = (float)((p.x - minx) * pixelsPerUnit + marginPx);
        float py = (float)((p.y - miny) * pixelsPerUnit + marginPx);
        return {px, py};
    };

    cv::Point2f q0 = toPix(p0);
    cv::Point2f q1 = toPix(p1);
    cv::Point2f q2 = toPix(p2);

    std::vector<std::vector<cv::Point>> poly(1);
    poly[0].push_back(cv::Point((int)std::lround(q0.x), (int)std::lround(q0.y)));
    poly[0].push_back(cv::Point((int)std::lround(q1.x), (int)std::lround(q1.y)));
    poly[0].push_back(cv::Point((int)std::lround(q2.x), (int)std::lround(q2.y)));
    cv::fillPoly(mask, poly, cv::Scalar(255));

    FaceRemap fr;
    fr.mapX = mapX;
    fr.mapY = mapY;
    fr.mask = mask;
    fr.triPx = {q0, q1, q2};
    fr.size = cv::Size(w, h);
    fr.basisX = x;
    fr.basisY = y;
    fr.basisZ = z;
    fr.planeMinX = minx;
    fr.planeMinY = miny;
    fr.marginPx = marginPx;
    fr.pixelsPerUnit = pixelsPerUnit;
    return fr;
}

IcosahedronUnwrapper::IcosahedronUnwrapper(const cv::Size& srcSize, int marginPx, double pixelsPerUnit)
    : ico_(buildIcosahedron()) {
    rebuild(srcSize, marginPx, pixelsPerUnit);
}

void IcosahedronUnwrapper::rebuild(const cv::Size& srcSize, int marginPx, double pixelsPerUnit) {
    srcSize_ = srcSize;
    marginPx_ = marginPx;
    pixelsPerUnit_ = pixelsPerUnit;

    remaps_.clear();
    remaps_.reserve(ico_.F.size());
    for (int i = 0; i < (int)ico_.F.size(); ++i) {
        remaps_.push_back(buildFaceRemap(ico_, i, marginPx_, pixelsPerUnit_, srcSize_));
    }
}

int IcosahedronUnwrapper::faceCount() const {
    return (int)remaps_.size();
}

FaceOutput IcosahedronUnwrapper::applyFace(const cv::Mat& srcEqrBGR, int faceIdx) const {
    if (srcEqrBGR.size() != srcSize_) {
        throw std::invalid_argument("Input image size does not match remap cache size.");
    }

    const FaceRemap& remap = remaps_.at(faceIdx);

    FaceOutput fo;
    fo.image = cv::Mat(remap.size, CV_8UC3);
    cv::remap(srcEqrBGR, fo.image, remap.mapX, remap.mapY, cv::INTER_LINEAR, cv::BORDER_WRAP);
    fo.mask = remap.mask;
    fo.triPx = remap.triPx;
    return fo;
}

std::vector<FaceOutput> IcosahedronUnwrapper::applyAll(const cv::Mat& srcEqrBGR) const {
    if (srcEqrBGR.size() != srcSize_) {
        throw std::invalid_argument("Input image size does not match remap cache size.");
    }

    std::vector<FaceOutput> out;
    out.reserve(remaps_.size());
    for (int i = 0; i < (int)remaps_.size(); ++i) {
        out.push_back(applyFace(srcEqrBGR, i));
    }
    return out;
}

cv::Point2f IcosahedronUnwrapper::facePixelToEquirectangularPixel(int faceIdx, const cv::Point2f& facePx) const {
    const FaceRemap& remap = remaps_.at(faceIdx);

    if (remap.pixelsPerUnit <= 0.0) {
        throw std::runtime_error("Invalid pixelsPerUnit in face remap.");
    }

    const double ux = remap.planeMinX + (static_cast<double>(facePx.x) - remap.marginPx) / remap.pixelsPerUnit;
    const double uy = remap.planeMinY + (static_cast<double>(facePx.y) - remap.marginPx) / remap.pixelsPerUnit;

    const cv::Vec3d dir = normalize3(ux * remap.basisX + uy * remap.basisY + remap.basisZ);

    const double lon = std::atan2(dir[0], dir[2]);
    const double lat = std::asin(clamp(dir[1], -1.0, 1.0));

    const double fx = wrap01((lon + CV_PI) / (2.0 * CV_PI));
    const double fy = (CV_PI / 2.0 - lat) / CV_PI;

    const float eqrX = static_cast<float>(fx * srcSize_.width);
    const float eqrY = static_cast<float>(clamp(fy * srcSize_.height, 0.0, static_cast<double>(srcSize_.height - 1)));
    return cv::Point2f(eqrX, eqrY);
}

} // namespace eqr_ico
} // namespace stella_vslam