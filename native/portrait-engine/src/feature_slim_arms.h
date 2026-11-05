#pragma once

#include "feature.h"

namespace pe {

std::unique_ptr<Feature> create_slim_arms_feature(const EngineConfig& config);

/* Pure geometry helpers exposed for synthetic unit tests (no ONNX). */
namespace slim_arms_test {

struct Point {
    float x = 0;
    float y = 0;
    Point add(Point o) const { return {x + o.x, y + o.y}; }
    Point sub(Point o) const { return {x - o.x, y - o.y}; }
    Point mul(float s) const { return {x * s, y * s}; }
    float dot(Point o) const { return x * o.x + y * o.y; }
    float length() const;
    Point normalized() const;
};

struct ArmSegment {
    Point start;
    Point end;
    float half_width = 0;
    bool fade_start = false;
    bool fade_end = false;
};

enum class CrossLevel { None, Contact, Cross };

PE_API float smooth(float value);
PE_API bool segments_intersect(Point a0, Point a1, Point b0, Point b1);
PE_API CrossLevel classify_cross(const ArmSegment* segments, const uint8_t* ownership,
                                 size_t cells);
PE_API Point source_point(Point point, ArmSegment segment, float strength, bool* ok);

}  // namespace slim_arms_test

}  // namespace pe
