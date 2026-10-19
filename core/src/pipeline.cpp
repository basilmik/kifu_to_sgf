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

// Перевод cv::Mat (любой глубины/каналов) в RGBA-байты
std::vector<std::uint8_t> mat_to_rgba(const cv::Mat& src) {
    cv::Mat tmp;
    if (src.channels() == 1) {
        cv::cvtColor(src, tmp, cv::COLOR_GRAY2RGBA);
    } else if (src.channels() == 3) {
        cv::cvtColor(src, tmp, cv::COLOR_BGR2RGBA);
    } else if (src.channels() == 4) {
        tmp = src.clone();
    } else {
        tmp = src.clone();
    }
    return std::vector<std::uint8_t>(
        tmp.data, tmp.data + tmp.total() * tmp.channels());
}

// Общая внутренняя функция. Если debug != nullptr — заполняет его.
struct DetectInternalResult {
    bool found = false;
    std::vector<cv::Point> ordered_corners;
};

DetectInternalResult run_detect(const cv::Mat& bgr,
                                double min_area_ratio,
                                DebugResult* debug) {
    DetectInternalResult res;

    cv::Mat gray, blurred, edges, edges_processed;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edges, 50, 150);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(edges, edges_processed, kernel);
    cv::erode(edges_processed, edges_processed, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges_processed, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    std::sort(contours.begin(), contours.end(),
              [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                  return cv::contourArea(a) > cv::contourArea(b);
              });

    const double img_area = static_cast<double>(bgr.cols) * bgr.rows;

    if (debug) {
        debug->img_area = img_area;
        debug->images.push_back({"gray", gray.cols, gray.rows, mat_to_rgba(gray)});
        debug->images.push_back({"edges", edges.cols, edges.rows, mat_to_rgba(edges)});
        debug->images.push_back({"edges_processed", edges_processed.cols,
                                 edges_processed.rows, mat_to_rgba(edges_processed)});
    }

    std::vector<cv::Point> best;
    const std::size_t limit = std::min<std::size_t>(contours.size(), 10);
    for (std::size_t i = 0; i < limit; ++i) {
        const auto& c = contours[i];
        double area = cv::contourArea(c);
        double peri = cv::arcLength(c, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.03 * peri, true);

        bool big_enough = area >= img_area * min_area_ratio;
        bool quad_ok = approx.size() == 4 && is_reasonable_quad(approx);

        if (debug) {
            ContourInfo ci;
            ci.area = area;
            ci.perimeter = peri;
            ci.approx_points = static_cast<int>(approx.size());
            ci.reasonable = quad_ok;
            for (const auto& p : approx) ci.points.emplace_back(p.x, p.y);
            debug->top_contours.push_back(ci);
        }

        if (big_enough && quad_ok && best.empty()) {
            best = approx;
        }
    }

    if (!best.empty()) {
        res.found = true;
        res.ordered_corners = order_corners(best);
    }

    if (debug) {
        if (contours.empty()) {
            debug->note = "no contours found";
        } else if (debug->top_contours.empty()) {
            debug->note = "no contours processed (unexpected)";
        } else {
            // формулируем, почему не нашли
            bool any_big = false, any_quad = false;
            for (const auto& ci : debug->top_contours) {
                if (ci.area >= img_area * min_area_ratio) any_big = true;
                if (ci.approx_points == 4 && ci.reasonable) any_quad = true;
            }
            if (!any_big) {
                debug->note = "no contour larger than min_area_ratio";
            } else if (!any_quad) {
                debug->note = "no reasonable 4-point quad among top contours";
            } else if (!res.found) {
                debug->note = "candidate found but rejected";
            } else {
                debug->note = "found";
            }
        }
    }

    return res;
}

}  // namespace

DetectResult detect_board(const std::vector<std::uint8_t>& rgba, int width, int height) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("width/height must be positive");
    if (rgba.size() != static_cast<std::size_t>(width) * height * 4)
        throw std::invalid_argument("RGBA buffer size does not match width*height*4");

    cv::Mat rgba_mat(height, width, CV_8UC4, const_cast<std::uint8_t*>(rgba.data()));
    cv::Mat bgr;
    cv::cvtColor(rgba_mat, bgr, cv::COLOR_RGBA2BGR);

    auto internal = run_detect(bgr, 0.10, nullptr);

    DetectResult result;
    result.found = internal.found;

    cv::Mat marked = bgr.clone();
    if (internal.found) {
        std::vector<cv::Point> poly = internal.ordered_corners;
        cv::polylines(marked, poly, true, cv::Scalar(0, 0, 255), 4);
        for (std::size_t i = 0; i < poly.size(); ++i) {
            cv::circle(marked, poly[i], 10, cv::Scalar(0, 255, 0), -1);
            cv::putText(marked, std::to_string(i), poly[i] + cv::Point(12, -12),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            result.corners.emplace_back(poly[i].x, poly[i].y);
        }
    }
    result.marked_rgba = mat_to_rgba(marked);
    return result;
}

DebugResult detect_board_debug(const std::vector<std::uint8_t>& rgba, int width, int height) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("width/height must be positive");
    if (rgba.size() != static_cast<std::size_t>(width) * height * 4)
        throw std::invalid_argument("RGBA buffer size does not match width*height*4");

    cv::Mat rgba_mat(height, width, CV_8UC4, const_cast<std::uint8_t*>(rgba.data()));
    cv::Mat bgr;
    cv::cvtColor(rgba_mat, bgr, cv::COLOR_RGBA2BGR);

    DebugResult result;
    auto internal = run_detect(bgr, 0.10, &result);

    result.detect.found = internal.found;

    cv::Mat marked = bgr.clone();
    if (internal.found) {
        std::vector<cv::Point> poly = internal.ordered_corners;
        cv::polylines(marked, poly, true, cv::Scalar(0, 0, 255), 4);
        for (std::size_t i = 0; i < poly.size(); ++i) {
            cv::circle(marked, poly[i], 10, cv::Scalar(0, 255, 0), -1);
            cv::putText(marked, std::to_string(i), poly[i] + cv::Point(12, -12),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            result.detect.corners.emplace_back(poly[i].x, poly[i].y);
        }
    }
    result.detect.marked_rgba = mat_to_rgba(marked);
    result.images.push_back({"marked", marked.cols, marked.rows, mat_to_rgba(marked)});

    return result;
}

}  // namespace kifu