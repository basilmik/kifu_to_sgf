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

// Сводит 5–6 точек к 4: отбрасывает точку, удаление которой меньше всего
// меняет площадь многоугольника (то есть шумовую).
std::vector<cv::Point> reduce_to_quad(const std::vector<cv::Point>& pts) {
    if (pts.size() == 4) return pts;
    if (pts.size() < 4) return {};
    double full_area = cv::contourArea(pts);
    double best_loss = 1e18;
    std::size_t best_idx = 0;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        std::vector<cv::Point> tmp;
        tmp.reserve(pts.size() - 1);
        for (std::size_t j = 0; j < pts.size(); ++j) {
            if (j != i) tmp.push_back(pts[j]);
        }
        double a = cv::contourArea(tmp);
        double loss = std::abs(full_area - a);
        if (loss < best_loss) { best_loss = loss; best_idx = i; }
    }
    std::vector<cv::Point> out;
    out.reserve(pts.size() - 1);
    for (std::size_t j = 0; j < pts.size(); ++j) {
        if (j != best_idx) out.push_back(pts[j]);
    }
    return out;
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

struct DetectInternalResult {
    bool found = false;
    std::vector<cv::Point> ordered_corners;
};


DetectInternalResult run_detect(const cv::Mat& bgr,
                                double min_area_ratio,
                                DebugResult* debug) {
    DetectInternalResult res;

    // 1. Grayscale
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    // 2. CLAHE: локальное выравнивание контраста
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat gray_eq;
    clahe->apply(gray, gray_eq);

    // 3. Лёгкое размытие, чтобы убрать зерно
    cv::Mat blurred;
    cv::GaussianBlur(gray_eq, blurred, cv::Size(3, 3), 0);

    // 4. Адаптивная бинаризация: инвертированная, чтобы линии были белыми на чёрном
    cv::Mat binary;
    cv::adaptiveThreshold(
        blurred, binary,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        31,   // blockSize — нечётный, ~размер линий
        10    // C — смещение; больше = меньше пикселей считаются линиями
    );

    // 5. Морфология: утолщаем и замыкаем линии
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat edges_processed;
    cv::dilate(binary, edges_processed, kernel, cv::Point(-1, -1), 1);
    cv::erode(edges_processed, edges_processed, kernel, cv::Point(-1, -1), 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges_processed, contours, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);

    std::sort(contours.begin(), contours.end(),
              [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                  return cv::contourArea(a) > cv::contourArea(b);
              });

    const double img_area = static_cast<double>(bgr.cols) * bgr.rows;

    if (debug) {
        debug->img_area = img_area;
        debug->images.push_back({"gray", gray.cols, gray.rows, mat_to_rgba(gray)});
        debug->images.push_back({"gray_clahe", gray_eq.cols, gray_eq.rows, mat_to_rgba(gray_eq)});
        debug->images.push_back({"binary", binary.cols, binary.rows, mat_to_rgba(binary)});
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

        bool big_enough   = area >= img_area * min_area_ratio;
        bool small_enough = area <= img_area * 0.90;

        bool quad_ok = false;
        std::vector<cv::Point> quad;
        if (approx.size() >= 4 && approx.size() <= 6) {
            quad = reduce_to_quad(approx);
            if (quad.size() == 4 && is_reasonable_quad(quad)) {
                quad_ok = true;
            }
        }

        if (debug) {
            ContourInfo ci;
            ci.area = area;
            ci.perimeter = peri;
            ci.approx_points = static_cast<int>(approx.size());
            ci.reasonable = quad_ok;
            for (const auto& p : approx) ci.points.emplace_back(p.x, p.y);
            debug->top_contours.push_back(ci);
        }

        if (big_enough && small_enough && quad_ok && best.empty()) {
            best = quad;
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
            bool any_big = false, any_small = false, any_quad = false;
            for (const auto& ci : debug->top_contours) {
                if (ci.area >= img_area * min_area_ratio) any_big = true;
                if (ci.area <= img_area * 0.90) any_small = true;
                if (ci.reasonable) any_quad = true;
            }
            if (!any_big) {
                debug->note = "no contour larger than min_area_ratio";
            } else if (!any_small) {
                debug->note = "contour is too big (likely full page)";
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


bool fit_grid_axis(const cv::Mat& line_mask, bool vertical, int grid_size, std::vector<float>& positions, double& quality) {
    positions.clear();
    quality = 0.0;

    const int length = vertical
        ? line_mask.cols
        : line_mask.rows;

    const int line_length = vertical
        ? line_mask.rows
        : line_mask.cols;

    // 1. Строим профиль количества пикселей линий.
    std::vector<float> profile(length, 0.0f);

    for (int i = 0; i < length; ++i) {
        if (vertical) {
            profile[i] = static_cast<float>(
                cv::countNonZero(line_mask.col(i))
            );
        } else {
            profile[i] = static_cast<float>(
                cv::countNonZero(line_mask.row(i))
            );
        }
    }
    // 2. Сглаживаем профиль.
    std::vector<float> smooth(length, 0.0f);
    for (int i = 0; i < length; ++i) {
        float sum = 0.0f;
        int count = 0;

        for (int d = -3; d <= 3; ++d) {
            const int j = i + d;

            if (j >= 0 && j < length) {
                sum += profile[j];
                ++count;
            }
        }
        smooth[i] = sum / count;
    }
    const float max_signal = *std::max_element(
        smooth.begin(), smooth.end()
    );
    // Нет достаточно длинных линий.
    if (max_signal < line_length * 0.08f) {
        return false;
    }
    // 3. Перебираем модели регулярной сетки.
    const float min_step =
        0.50f * length / (grid_size - 1);

    const float max_step =
        0.98f * length / (grid_size - 1);

    double best_score = -1.0;

    float best_start = 0.0f;
    float best_step = 0.0f;
    double best_quality = 0.0;
    const int min_hits = static_cast<int>(
        std::ceil(grid_size * 0.70)
    );

    for (float step = min_step; step <= max_step; step += 0.25f) {
        const int max_start = static_cast<int>(
            std::floor(
                length - 1 - (grid_size - 1) * step
            )
        );

        for (int start = 0; start <= max_start; ++start) {
            double sum = 0.0;
            int hits = 0;

            for (int i = 0; i < grid_size; ++i) {
                const int pos = static_cast<int>(
                    std::round(start + i * step)
                );

                const float signal = smooth[pos];

                sum += signal;

                if (signal >= max_signal * 0.20f) {
                    ++hits;
                }
            }
            // Не менее 70% линий должны иметь заметное подтверждение на изображении.
            if (hits < min_hits) {
                continue;
            }
            // Дополнительно поощряем модели, подтверждённые большим количеством линий.
            const double score =
                sum + hits * max_signal * 0.08;

            if (score > best_score) {
                best_score = score;
                best_start = start;
                best_step = step;

                best_quality =
                    sum / (grid_size * max_signal);
            }
        }
    }
    if (best_score < 0.0 || best_quality < 0.25) {
        return false;
    }
    // 4. Возвращаем регулярную модель.
    // Даже если часть реальных линий закрыта цифрами,
    // координаты восстанавливаются по общему шагу.
    for (int i = 0; i < grid_size; ++i) {
        positions.push_back(
            best_start + i * best_step
        );
    }
    quality = best_quality;
    return true;
}


GridResult rectify_grid(const std::vector<std::uint8_t>& rgba, int width, int height, const std::vector<std::pair<float, float>>& manual_corners, int grid_size, int output_side) 
{
    if (width <= 0 || height <= 0 ||
        rgba.size() !=
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height) * 4) {
        throw std::invalid_argument("invalid RGBA image");
    }

    if (grid_size != 9 &&
        grid_size != 13 &&
        grid_size != 19) {
        throw std::invalid_argument("unsupported grid size");
    }

    if (output_side < 512 || output_side > 4096) {
        throw std::invalid_argument("invalid output_side");
    }

    if (!manual_corners.empty() &&
        manual_corners.size() != 4) {
        throw std::invalid_argument(
            "manual_corners must contain exactly 4 points"
        );
    }
    GridResult result;
    result.grid_size = grid_size;
    
    // 1. Конвертация входного RGBA в BGR.
    cv::Mat rgba_mat(
        height,
        width,
        CV_8UC4,
        const_cast<std::uint8_t*>(rgba.data())
    );

    cv::Mat bgr;
    cv::cvtColor(rgba_mat, bgr, cv::COLOR_RGBA2BGR);

    // 2. Получение четырёх углов.
    std::vector<cv::Point2f> src;

    const bool manual = !manual_corners.empty();

    if (manual) {
        for (const auto& p : manual_corners) {
            src.emplace_back(p.first, p.second);
        }
    } else {
        auto detected = run_detect(bgr, 0.10, nullptr);

        if (!detected.found) {
            result.note = "board quadrilateral not found";
            return result;
        }

        for (const auto& p : detected.ordered_corners) {
            src.emplace_back(
                static_cast<float>(p.x),
                static_cast<float>(p.y)
            );
        }
    }

    // Углы должны задавать выпуклый четырёхугольник.
    if (!cv::isContourConvex(src) ||
        std::abs(cv::contourArea(src)) < 100.0) {
        throw std::invalid_argument(
            "invalid or unordered board corners"
        );
    }

    for (const auto& p : src) {
        result.corners.emplace_back(p.x, p.y);
    }

    // 3. Задаём прямоугольник назначения.
    //
    // Небольшое поле по краям нужно, чтобы
    // крайние линии сетки не обрезались.

    const float pad = output_side * 0.045f;
    const float last = output_side - 1.0f - pad;

    std::vector<cv::Point2f> dst = {
        {pad, pad},
        {last, pad},
        {last, last},
        {pad, last}
    };

    // 4. Перспективное преобразование.
    cv::Mat H = cv::getPerspectiveTransform(src, dst);

    cv::Mat rectified;

    cv::warpPerspective(
        bgr,
        rectified,
        H,
        cv::Size(output_side, output_side),
        cv::INTER_CUBIC,
        cv::BORDER_CONSTANT,
        cv::Scalar(255, 255, 255)
    );

    result.rectified_side = output_side;
    result.rectified_rgba = mat_to_rgba(rectified);

    // 5. Получение координат линий.
    if (manual) {
        // В ручном режиме четыре угла относятся именно
        // к крайним пересечениям доски.
        //
        // После перспективного преобразования
        // сетку можно задать геометрически.

        const float span = last - pad;

        for (int i = 0; i < grid_size; ++i) {
            const float position =
                pad + span * i / (grid_size - 1);

            result.x_lines.push_back(position);
            result.y_lines.push_back(position);
        }

        result.note = "grid from manual corners";
    } else {
        // Автоматический режим:
        // ищем реальные линии на выпрямленном фото.

        cv::Mat gray;
        cv::cvtColor(
            rectified,
            gray,
            cv::COLOR_BGR2GRAY
        );

        auto clahe = cv::createCLAHE(
            2.0,
            cv::Size(8, 8)
        );

        cv::Mat enhanced;
        clahe->apply(gray, enhanced);

        cv::Mat binary;

        cv::adaptiveThreshold(
            enhanced,
            binary,
            255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY_INV,
            41,
            8
        );

        // Выделяем длинные горизонтальные
        // и вертикальные элементы.
        const int kernel_length = std::max(
            15,
            output_side / (grid_size * 3)
        );

        cv::Mat vertical_kernel =
            cv::getStructuringElement(
                cv::MORPH_RECT,
                cv::Size(1, kernel_length)
            );

        cv::Mat horizontal_kernel =
            cv::getStructuringElement(
                cv::MORPH_RECT,
                cv::Size(kernel_length, 1)
            );

        cv::Mat vertical_lines;
        cv::Mat horizontal_lines;

        cv::morphologyEx(
            binary,
            vertical_lines,
            cv::MORPH_OPEN,
            vertical_kernel
        );

        cv::morphologyEx(
            binary,
            horizontal_lines,
            cv::MORPH_OPEN,
            horizontal_kernel
        );

        // Подгоняем модель регулярной сетки.
        const bool x_ok = fit_grid_axis(
            vertical_lines,
            true,
            grid_size,
            result.x_lines,
            result.x_quality
        );

        const bool y_ok = fit_grid_axis(
            horizontal_lines,
            false,
            grid_size,
            result.y_lines,
            result.y_quality
        );

        if (!x_ok || !y_ok) {
            result.x_lines.clear();
            result.y_lines.clear();

            result.note =
                "quadrilateral found, but grid not confirmed";

            return result;
        }

        result.note = "automatic grid fitted";
    }

    // 6. Строим все пересечения
    // в координатах выпрямленного изображения.
    std::vector<cv::Point2f> rectified_points;

    for (int row = 0; row < grid_size; ++row) {
        for (int col = 0; col < grid_size; ++col) {
            rectified_points.emplace_back(
                result.x_lines[col],
                result.y_lines[row]
            );
        }
    }

    // 7. Обратное перспективное преобразование.
    //
    // Нужно, чтобы фронтенд мог показать
    // пересечения поверх исходного фото.

    cv::Mat H_inv = H.inv();

    std::vector<cv::Point2f> image_points;

    cv::perspectiveTransform(
        rectified_points,
        image_points,
        H_inv
    );

    for (const auto& p : image_points) {
        result.grid_points_image.emplace_back(
            p.x,
            p.y
        );
    }

    result.found = true;

    return result;
}
}  // namespace kifu