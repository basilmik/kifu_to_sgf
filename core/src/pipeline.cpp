#include "kifu.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

#include <opencv2/opencv.hpp>

namespace kifu {

std::vector<std::uint8_t> reverse_pixels(const std::vector<std::uint8_t>& rgba) {
    if (rgba.size() % 4 != 0) {
        throw std::invalid_argument("RGBA buffer size must be a multiple of 4");
    }
    std::vector<std::uint8_t> out(rgba.size());
    const std::size_t n = rgba.size() / 4;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t src = (n - 1 - i) * 4;
        const std::size_t dst = i * 4;
        out[dst + 0] = rgba[src + 0];
        out[dst + 1] = rgba[src + 1];
        out[dst + 2] = rgba[src + 2];
        out[dst + 3] = rgba[src + 3];
    }
    return out;
}

namespace {

bool is_reasonable_quad(const std::vector<cv::Point>& pts) {
    if (pts.size() != 4) return false;
    double sides[4];
    for (int i = 0; i < 4; ++i) {
        cv::Point2f d = cv::Point2f(pts[(i + 1) % 4]) - cv::Point2f(pts[i]);
        sides[i] = std::sqrt(d.x * d.x + d.y * d.y);
    }
    std::sort(sides, sides + 4);
    if (sides[0] < 20.0) return false;
    if (sides[3] / sides[0] > 5.0) return false;
    return true;
}

std::vector<cv::Point> order_corners(const std::vector<cv::Point>& pts) {
    cv::Point tl, tr, br, bl;
    int min_sum = INT_MAX, max_sum = INT_MIN;
    int min_diff = INT_MAX, max_diff = INT_MIN;
    for (const auto& p : pts) {
        int s = p.x + p.y;
        int d = p.x - p.y;
        if (s < min_sum) { min_sum = s; tl = p; }
        if (s > max_sum) { max_sum = s; br = p; }
        if (d > max_diff) { max_diff = d; tr = p; }
        if (d < min_diff) { min_diff = d; bl = p; }
    }
    return {tl, tr, br, bl};
}

}  // namespace

DetectResult detect_board(const std::vector<std::uint8_t>& rgba, int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("width/height must be positive");
    }
    if (rgba.size() != static_cast<std::size_t>(width) * height * 4) {
        throw std::invalid_argument("RGBA buffer size does not match width*height*4");
    }

    DetectResult result;

    cv::Mat rgba_mat(height, width, CV_8UC4, const_cast<std::uint8_t*>(rgba.data()));
    cv::Mat bgr;
    cv::cvtColor(rgba_mat, bgr, cv::COLOR_RGBA2BGR);

    cv::Mat gray, blurred, edges;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edges, 50, 150);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(edges, edges, kernel);
    cv::erode(edges, edges, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::sort(contours.begin(), contours.end(),
              [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                  return cv::contourArea(a) > cv::contourArea(b);
              });

    const double img_area = static_cast<double>(width) * height;
    std::vector<cv::Point> best;
    const std::size_t limit = std::min<std::size_t>(contours.size(), 10);
    for (std::size_t i = 0; i < limit; ++i) {
        const auto& c = contours[i];
        double area = cv::contourArea(c);
        if (area < img_area * 0.10) break;

        double peri = cv::arcLength(c, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.02 * peri, true);

        if (approx.size() == 4 && is_reasonable_quad(approx)) {
            best = approx;
            break;
        }
    }

    cv::Mat marked = bgr.clone();

    if (!best.empty()) {
        auto ordered = order_corners(best);

        std::vector<cv::Point> poly(ordered.begin(), ordered.end());
        cv::polylines(marked, poly, true, cv::Scalar(0, 0, 255), 4);

        for (std::size_t i = 0; i < ordered.size(); ++i) {
            cv::circle(marked, ordered[i], 10, cv::Scalar(0, 255, 0), -1);
            cv::putText(marked,
                        std::to_string(i),
                        ordered[i] + cv::Point(12, -12),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1.0,
                        cv::Scalar(0, 255, 0),
                        2);
            result.corners.emplace_back(ordered[i].x, ordered[i].y);
        }
        result.found = true;
    }

    cv::Mat marked_rgba;
    cv::cvtColor(marked, marked_rgba, cv::COLOR_BGR2RGBA);
    result.marked_rgba.assign(marked_rgba.data,
                              marked_rgba.data + marked_rgba.total() * marked_rgba.channels());

    return result;
}

}  // namespace kifu