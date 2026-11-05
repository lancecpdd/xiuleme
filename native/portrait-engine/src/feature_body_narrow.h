#pragma once

#include "feature.h"

namespace pe {

std::unique_ptr<Feature> create_slim_waist_feature(const EngineConfig& config);
std::unique_ptr<Feature> create_slim_legs_feature(const EngineConfig& config);

}  // namespace pe
