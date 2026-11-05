#include <cmath>
#include <cstdio>
#include <vector>

#include "../src/feature_slim_arms.h"

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

using pe::slim_arms_test::ArmSegment;
using pe::slim_arms_test::CrossLevel;
using pe::slim_arms_test::Point;
using pe::slim_arms_test::classify_cross;
using pe::slim_arms_test::segments_intersect;
using pe::slim_arms_test::smooth;
using pe::slim_arms_test::source_point;

}  // namespace

int main() {
    check(smooth(0.0f) == 0.0f, "smooth(0)");
    check(smooth(1.0f) == 1.0f, "smooth(1)");
    check(smooth(0.5f) > 0.0f && smooth(0.5f) < 1.0f, "smooth(0.5)");

    check(segments_intersect({0, 0}, {10, 10}, {0, 10}, {10, 0}), "crossing segments");
    check(!segments_intersect({0, 0}, {10, 0}, {0, 5}, {10, 5}), "parallel segments");

    ArmSegment segment{{10, 10}, {110, 10}, 12.0f, true, true};
    bool ok = false;
    Point near_joint = source_point({10, 16}, segment, 0.15f, &ok);
    check(ok && std::fabs(near_joint.y - 16.0f) < 1e-2f, "shoulder fade nearly fixed");
    Point mid = source_point({60, 16}, segment, 0.15f, &ok);
    check(ok && mid.y > 16.5f, "mid-segment moves outward for inverse sample");
    Point far = source_point({60, 40}, segment, 0.15f, &ok);
    check(!ok, "outside envelope rejected");

    ArmSegment separated[4] = {
        {{40, 15}, {40, 55}, 8, true, false},
        {{40, 55}, {40, 105}, 8, false, true},
        {{120, 15}, {120, 55}, 8, true, false},
        {{120, 55}, {120, 105}, 8, false, true},
    };
    std::vector<uint8_t> empty(512 * 512, 0);
    check(classify_cross(separated, empty.data(), empty.size()) == CrossLevel::None,
          "separated arms => None");

    ArmSegment near[4] = {
        {{50, 15}, {50, 55}, 8, true, false},
        {{50, 55}, {50, 105}, 8, false, true},
        {{54, 15}, {54, 55}, 8, true, false},
        {{54, 55}, {54, 105}, 8, false, true},
    };
    check(classify_cross(near, empty.data(), empty.size()) == CrossLevel::Contact,
          "near arms => Contact");

    ArmSegment crossed[4] = {
        {{40, 15}, {40, 55}, 8, true, false},
        {{40, 55}, {120, 105}, 8, false, true},
        {{120, 15}, {120, 55}, 8, true, false},
        {{120, 55}, {40, 105}, 8, false, true},
    };
    check(classify_cross(crossed, empty.data(), empty.size()) == CrossLevel::Cross,
          "crossing bones => Cross");

    // Synthetic multi-person mask: two large blobs -> helper not available here, so only
    // document that analyze maps that case to bypass reason multiple_people via SCHP path.

    if (failures == 0) {
        std::printf("portrait engine geometry test passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
}
