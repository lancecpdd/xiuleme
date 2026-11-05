/**
 * XiuLeMe product C ABI around GPUPixel FaceReshape + beauty stack.
 * Filter order: lipstick → blusher → reshape → beauty → sink,
 * then CPU sclera brighten / redness on the RGBA readout (VNN 106 landmarks).
 *
 * Windows / Linux: GLFW + glad (upstream desktop path).
 * macOS: NSOpenGLContext via GPUPixelContext (no GLFW).
 */
#define GPUPIXEL_RR_EXPORTS
#include "gpupixel_rr.h"

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#else
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "gpupixel.h"

#if defined(_WIN32)
#include <direct.h>
#define gpu_rr_chdir _chdir
#else
#include <unistd.h>
#define gpu_rr_chdir chdir
#endif

using namespace gpupixel;

namespace {

// VNN / GPUPixel 106 eye rings + big-eye iris anchors (same as FaceReshapeFilter).
constexpr int kLeftEyeLid[] = {52, 53, 54, 55, 56, 57};
constexpr int kRightEyeLid[] = {58, 59, 60, 61, 62, 63};
constexpr int kLeftIrisCenter = 74;
constexpr int kRightIrisCenter = 77;
constexpr int kLeftIrisRadiusRef = 72;
constexpr int kRightIrisRadiusRef = 75;
constexpr int kLeftInnerCanthus = 52;
constexpr int kRightInnerCanthus = 58;

inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

inline float smoothstep(float edge0, float edge1, float x) {
  float t = clampf((x - edge0) / (edge1 - edge0 + 1e-6f), 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}

inline float luma_srgb(float r, float g, float b) {
  return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

bool point_in_poly(float x, float y, const float* pts, const int* idx, int n) {
  bool inside = false;
  for (int i = 0, j = n - 1; i < n; j = i++) {
    float xi = pts[idx[i] * 2];
    float yi = pts[idx[i] * 2 + 1];
    float xj = pts[idx[j] * 2];
    float yj = pts[idx[j] * 2 + 1];
    bool intersect =
        ((yi > y) != (yj > y)) &&
        (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-8f) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

float poly_edge_dist(float x, float y, const float* pts, const int* idx, int n) {
  float best = 1e9f;
  for (int i = 0, j = n - 1; i < n; j = i++) {
    float ax = pts[idx[j] * 2];
    float ay = pts[idx[j] * 2 + 1];
    float bx = pts[idx[i] * 2];
    float by = pts[idx[i] * 2 + 1];
    float abx = bx - ax;
    float aby = by - ay;
    float apx = x - ax;
    float apy = y - ay;
    float denom = abx * abx + aby * aby;
    float t = denom > 1e-12f ? clampf((apx * abx + apy * aby) / denom, 0.f, 1.f) : 0.f;
    float dx = x - (ax + abx * t);
    float dy = y - (ay + aby * t);
    best = std::min(best, std::sqrt(dx * dx + dy * dy));
  }
  return best;
}

float sclera_weight(float x, float y, const float* pts, int n_pts,
                    const int* lid, int lid_n, int iris_c, int iris_ref,
                    int canthus) {
  if (n_pts <= std::max(iris_c, std::max(iris_ref, canthus)) || lid_n < 3) return 0.f;
  if (!point_in_poly(x, y, pts, lid, lid_n)) return 0.f;

  float cx = pts[iris_c * 2];
  float cy = pts[iris_c * 2 + 1];
  float rx = pts[iris_ref * 2];
  float ry = pts[iris_ref * 2 + 1];
  float iris_r = std::sqrt((cx - rx) * (cx - rx) + (cy - ry) * (cy - ry));
  // VNN 虹膜半径常偏小；再放大挖空，避免白边啃进虹膜。
  iris_r = std::max(iris_r * 1.35f, 0.014f);
  float d_iris = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
  // 硬保护区到 1.05R，羽化到 1.22R。
  float iris_keep = smoothstep(iris_r * 1.05f, iris_r * 1.22f, d_iris);

  float edge = poly_edge_dist(x, y, pts, lid, lid_n);
  float feather = clampf(iris_r * 0.22f, 0.003f, 0.014f);
  float lid_w = smoothstep(0.f, feather, edge);

  // 肉阜轻抑制（外眼角仍可处理）。
  float can_x = pts[canthus * 2];
  float can_y = pts[canthus * 2 + 1];
  float d_can = std::sqrt((x - can_x) * (x - can_x) + (y - can_y) * (y - can_y));
  float can_r = clampf(iris_r * 0.55f, 0.008f, 0.024f);
  float canthus_keep = 1.f - (1.f - smoothstep(can_r * 0.3f, can_r, d_can)) * 0.45f;

  return clampf(lid_w * iris_keep * canthus_keep, 0.f, 1.f);
}

// 在眼 ROI 内对标量场做盒模糊（仅用于血丝平滑）。
void box_blur_roi(std::vector<float>& field, int w, int h, int x0, int y0, int x1,
                  int y1, int radius) {
  if (radius < 1 || field.size() != static_cast<size_t>(w) * h) return;
  std::vector<float> tmp(field.size(), 0.f);
  const int rw = x1 - x0 + 1;
  const int rh = y1 - y0 + 1;
  if (rw < 3 || rh < 3) return;
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      float acc = 0.f;
      int cnt = 0;
      for (int dy = -radius; dy <= radius; ++dy) {
        int yy = clampi(y + dy, y0, y1);
        for (int dx = -radius; dx <= radius; ++dx) {
          int xx = clampi(x + dx, x0, x1);
          acc += field[static_cast<size_t>(yy) * w + xx];
          ++cnt;
        }
      }
      tmp[static_cast<size_t>(y) * w + x] = acc / std::max(cnt, 1);
    }
  }
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      field[static_cast<size_t>(y) * w + x] = tmp[static_cast<size_t>(y) * w + x];
    }
  }
}

void apply_eye_white(uint8_t* rgba, int width, int height,
                     const std::vector<float>& landmarks,
                     float brighten01, float redness01) {
  if (!rgba || width < 8 || height < 8) return;
  if (brighten01 <= 1e-4f && redness01 <= 1e-4f) return;
  if (landmarks.size() < 106 * 2) return;

  const float* pts = landmarks.data();
  const int n_pts = static_cast<int>(landmarks.size() / 2);

  // 提亮联动弱祛红/去黄；力度保守，避免漂白与过曝。
  const float coupled_red = std::max(redness01, brighten01 * 0.35f);
  const float deyellow = brighten01 * 0.75f;

  struct EyeBox {
    const int* lid;
    int lid_n;
    int iris_c;
    int iris_ref;
    int canthus;
    int x0, y0, x1, y1;
  };
  EyeBox eyes[2] = {
      {kLeftEyeLid, 6, kLeftIrisCenter, kLeftIrisRadiusRef, kLeftInnerCanthus,
       width, height, 0, 0},
      {kRightEyeLid, 6, kRightIrisCenter, kRightIrisRadiusRef, kRightInnerCanthus,
       width, height, 0, 0},
  };

  for (auto& eye : eyes) {
    for (int i = 0; i < eye.lid_n; ++i) {
      int id = eye.lid[i];
      if (id >= n_pts) continue;
      int px = static_cast<int>(pts[id * 2] * width);
      int py = static_cast<int>(pts[id * 2 + 1] * height);
      eye.x0 = std::min(eye.x0, px);
      eye.y0 = std::min(eye.y0, py);
      eye.x1 = std::max(eye.x1, px);
      eye.y1 = std::max(eye.y1, py);
    }
    int pad = std::max(6, (eye.x1 - eye.x0) / 3);
    eye.x0 = std::max(0, eye.x0 - pad);
    eye.y0 = std::max(0, eye.y0 - pad);
    eye.x1 = std::min(width - 1, eye.x1 + pad);
    eye.y1 = std::min(height - 1, eye.y1 + pad);
  }

  for (const auto& eye : eyes) {
    if (eye.x1 <= eye.x0 || eye.y1 <= eye.y0) continue;
    const int bw = eye.x1 - eye.x0 + 1;
    const int bh = eye.y1 - eye.y0 + 1;
    const int eye_w_px = std::max(bw, 8);

    std::vector<float> mask(static_cast<size_t>(bw) * bh, 0.f);
    std::vector<float> excess(static_cast<size_t>(bw) * bh, 0.f);
    std::vector<float> yellow(static_cast<size_t>(bw) * bh, 0.f);
    std::vector<float> src_r(static_cast<size_t>(bw) * bh, 0.f);
    std::vector<float> src_g(static_cast<size_t>(bw) * bh, 0.f);
    std::vector<float> src_b(static_cast<size_t>(bw) * bh, 0.f);

    double ref_r = 0, ref_g = 0, ref_b = 0, ref_w = 0;
    std::vector<float> excess_samples;
    excess_samples.reserve(256);

    for (int y = eye.y0; y <= eye.y1; ++y) {
      for (int x = eye.x0; x <= eye.x1; ++x) {
        const int li = (y - eye.y0) * bw + (x - eye.x0);
        float u = (x + 0.5f) / width;
        float v = (y + 0.5f) / height;
        float m = sclera_weight(u, v, pts, n_pts, eye.lid, eye.lid_n, eye.iris_c,
                                eye.iris_ref, eye.canthus);
        mask[li] = m;
        if (m < 0.02f) continue;
        const uint8_t* p = rgba + (static_cast<size_t>(y) * width + x) * 4;
        float r = p[0] / 255.f;
        float g = p[1] / 255.f;
        float b = p[2] / 255.f;
        src_r[li] = r;
        src_g[li] = g;
        src_b[li] = b;
        float yv = luma_srgb(r, g, b);
        float e = std::max(0.f, r - 0.5f * (g + b));
        float yel = std::max(0.f, 0.5f * (r + g) - b);
        excess[li] = e;
        yellow[li] = yel;
        if (m >= 0.25f && yv > 0.12f && yv < 0.88f && e < 0.05f && yel < 0.06f) {
          float w = m * (1.f - e * 12.f) * (1.f - yel * 10.f);
          ref_r += r * w;
          ref_g += g * w;
          ref_b += b * w;
          ref_w += w;
        }
        if (m > 0.30f && yv > 0.10f && yv < 0.92f) {
          excess_samples.push_back(e);
        }
      }
    }

    // 参考色：贴近本图干净巩膜，仅轻度去黄，不要推向纯冷白。
    float ref_rf = 0.74f, ref_gf = 0.74f, ref_bf = 0.76f;
    if (ref_w > 1e-3) {
      ref_rf = static_cast<float>(ref_r / ref_w);
      ref_gf = static_cast<float>(ref_g / ref_w);
      ref_bf = static_cast<float>(ref_b / ref_w);
      float yv = luma_srgb(ref_rf, ref_gf, ref_bf);
      ref_rf = ref_rf * 0.88f + yv * 0.12f;
      ref_gf = ref_gf * 0.88f + yv * 0.12f;
      ref_bf = ref_bf * 0.82f + std::min(yv * 1.02f, 1.f) * 0.18f;
      // 限制参考亮度，防止后续 pull 把巩膜拉到过曝。
      float ref_y = luma_srgb(ref_rf, ref_gf, ref_bf);
      if (ref_y > 0.82f) {
        float s = 0.82f / ref_y;
        ref_rf *= s;
        ref_gf *= s;
        ref_bf *= s;
      }
    }

    float baseline_excess = 0.015f;
    if (!excess_samples.empty()) {
      std::sort(excess_samples.begin(), excess_samples.end());
      size_t idx = static_cast<size_t>(excess_samples.size() * 0.12);
      if (idx >= excess_samples.size()) idx = excess_samples.size() - 1;
      baseline_excess = excess_samples[idx];
    }

    std::vector<float> excess_blur = excess;
    int blur_r = static_cast<int>(clampf(static_cast<float>(eye_w_px) * 0.035f, 1.f, 3.f));
    if (coupled_red > 0.05f) {
      box_blur_roi(excess_blur, bw, bh, 0, 0, bw - 1, bh - 1, blur_r);
    }

    for (int y = eye.y0; y <= eye.y1; ++y) {
      for (int x = eye.x0; x <= eye.x1; ++x) {
        const int li = (y - eye.y0) * bw + (x - eye.x0);
        float m = mask[li];
        if (m < 0.02f) continue;

        float r = src_r[li];
        float g = src_g[li];
        float b = src_b[li];
        float yv0 = luma_srgb(r, g, b);
        // 亮度门控：暗像素（虹膜/睫毛阴影）禁止提亮与漂白。
        float sclera_like = smoothstep(0.22f, 0.38f, yv0);
        if (sclera_like < 0.05f) continue;
        m *= sclera_like;

        float e = excess[li];
        float e_blur = excess_blur[li];
        float yel = yellow[li];

        // --- 祛红 / 细丝 ---
        if (coupled_red > 1e-4f) {
          float delta_e = std::max(0.f, e - baseline_excess);
          float local_boost = smoothstep(0.f, 0.02f, std::max(0.f, e - e_blur));
          float color_anom =
              std::max(smoothstep(0.002f, 0.03f, delta_e),
                       smoothstep(0.03f, 0.085f, e) * 0.75f);
          float vessel =
              clampf(color_anom * (0.50f + 0.50f * local_boost) * m, 0.f, 1.f);
          float smear = vessel * coupled_red * 0.65f;
          float target_e = e_blur * (1.f - smear) + baseline_excess * smear;
          float suppress = clampf((e - target_e) * coupled_red * 1.05f * m, 0.f, 0.18f);
          r -= suppress;
          g += std::min(suppress * 0.22f, 0.05f);
          b += std::min(suppress * 0.30f, 0.06f);
          float pull = vessel * coupled_red * 0.18f;
          r += (ref_rf - r) * pull;
          g += (ref_gf - g) * pull;
          b += (ref_bf - b) * pull;
        }

        // --- 去黄 ---
        if (deyellow > 1e-4f && yel > 0.01f) {
          float y_w = smoothstep(0.01f, 0.055f, yel) * m * deyellow;
          float drop = y_w * 0.35f;
          r -= drop * 0.28f;
          g -= drop * 0.16f;
          b += drop * 0.40f;
          r += (ref_rf - r) * y_w * 0.12f;
          g += (ref_gf - g) * y_w * 0.12f;
          b += (ref_bf - b) * y_w * 0.14f;
        }

        // --- 提亮：有限 headroom，硬顶，禁止冲到纯白 ---
        if (brighten01 > 1e-4f) {
          float yv = luma_srgb(r, g, b);
          const float soft_ceiling = 0.86f;
          float gap = std::max(0.f, soft_ceiling - yv);
          float gap_curve =
              gap * (0.55f + 0.45f * clampf(gap / soft_ceiling, 0.f, 1.f));
          float shadow = smoothstep(0.20f, 0.36f, yv);
          float highlight = 1.f - smoothstep(soft_ceiling - 0.08f, soft_ceiling, yv);
          float lift = m * brighten01 * 0.42f * gap_curve * shadow * highlight;
          lift = std::min(lift, 0.12f * m * brighten01);
          r += lift;
          g += lift;
          b += lift;
        }

        // 最终钳制：巩膜不要顶满。
        r = std::min(r, 0.92f);
        g = std::min(g, 0.92f);
        b = std::min(b, 0.93f);

        uint8_t* p = rgba + (static_cast<size_t>(y) * width + x) * 4;
        p[0] = static_cast<uint8_t>(clampf(r, 0.f, 1.f) * 255.f + 0.5f);
        p[1] = static_cast<uint8_t>(clampf(g, 0.f, 1.f) * 255.f + 0.5f);
        p[2] = static_cast<uint8_t>(clampf(b, 0.f, 1.f) * 255.f + 0.5f);
      }
    }
  }
}

class Session {
 public:
  Session() = default;
  ~Session() { release_output(); }

  bool initialize(const char* resource_dir) {
    if (resource_dir && resource_dir[0]) {
      if (gpu_rr_chdir(resource_dir) != 0) {
        std::cerr << "gpupixel_rr: chdir failed: " << resource_dir << std::endl;
        return false;
      }
    }

#if defined(__APPLE__)
    auto* ns_ctx = GPUPixelContext::getInstance()->getOpenGLContext();
    if (ns_ctx == nullptr) {
      std::cerr << "gpupixel_rr: failed to create NSOpenGL context" << std::endl;
      return false;
    }
    [ns_ctx makeCurrentContext];
#else
    window_ = GPUPixelContext::getInstance()->GetGLContext();
    if (window_ == nullptr) {
      std::cerr << "gpupixel_rr: failed to create GLFW/GL context" << std::endl;
      return false;
    }
    glfwMakeContextCurrent(window_);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
      std::cerr << "gpupixel_rr: gladLoadGLLoader failed" << std::endl;
      return false;
    }
#endif

    source_ = SourceRawDataInput::create();
    sink_ = TargetRawDataOutput::create();
    beauty_ = BeautyFaceFilter::create();
    reshape_ = FaceReshapeFilter::create();
    lipstick_ = LipstickFilter::create();
    blusher_ = BlusherFilter::create();

    source_->addTarget(lipstick_)
        ->addTarget(blusher_)
        ->addTarget(reshape_)
        ->addTarget(beauty_)
        ->addTarget(sink_);
    return true;
  }

  void set_callbacks() {
    source_->RegLandmarkCallback([this](std::vector<float> landmarks) {
      lipstick_->SetFaceLandmarks(landmarks);
      blusher_->SetFaceLandmarks(landmarks);
      reshape_->SetFaceLandmarks(landmarks);
      landmarks_ = std::move(landmarks);
      has_face_ = !landmarks_.empty();
    });
    sink_->setPixelsCallbck(
        [this](const uint8_t* data, int w, int h, int64_t) {
          if (data == nullptr || w != width_ || h != height_ || channels_ != 4) {
            output_ready_ = false;
            return;
          }
          const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h) *
                           static_cast<size_t>(channels_);
          if (!output_ || output_cap_ < n) {
            delete[] output_;
            output_ = new uint8_t[n];
            output_cap_ = n;
          }
          std::memcpy(output_, data, n);
          output_ready_ = true;
        });
  }

  static float clamp10(float v) {
    if (v < 0.f) return 0.f;
    if (v > 10.f) return 10.f;
    return v;
  }

  void set_parameters(float beauty, float white, float thin, float eye,
                      float lipstick, float blusher, float eye_brighten,
                      float eye_redness) {
    beauty = clamp10(beauty);
    white = clamp10(white);
    thin = clamp10(thin);
    eye = clamp10(eye);
    lipstick = clamp10(lipstick);
    blusher = clamp10(blusher);
    eye_brighten_ = clamp10(eye_brighten);
    eye_redness_ = clamp10(eye_redness);
    beauty_->setBlurAlpha(beauty / 10.f);
    beauty_->setWhite(white / 20.f);
    reshape_->setFaceSlimLevel(thin / 200.f);
    reshape_->setEyeZoomLevel(eye / 100.f);
    lipstick_->setBlendLevel(lipstick / 10.f);
    blusher_->setBlendLevel(blusher / 10.f);
  }

  uint8_t* run(const uint8_t* rgba, int width, int height, int channels) {
    if (!rgba || width <= 0 || height <= 0 || channels != 4) {
      return nullptr;
    }
    width_ = width;
    height_ = height;
    channels_ = channels;
    has_face_ = false;
    landmarks_.clear();
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height) *
                     static_cast<size_t>(channels);
    if (!output_ || output_cap_ < n) {
      delete[] output_;
      output_ = new uint8_t[n];
      output_cap_ = n;
    }
    output_ready_ = false;
    source_->uploadBytes(rgba, width, height, width);
    if (!output_ready_ || !output_) {
      return nullptr;
    }
    if (has_face_ && (eye_brighten_ > 1e-4f || eye_redness_ > 1e-4f)) {
      apply_eye_white(output_, width_, height_, landmarks_,
                      eye_brighten_ / 10.f, eye_redness_ / 10.f);
    }
    return output_;
  }

  void release_output() {
    delete[] output_;
    output_ = nullptr;
    output_cap_ = 0;
    output_ready_ = false;
  }

 private:
#if !defined(__APPLE__)
  GLFWwindow* window_ = nullptr;
#endif
  std::shared_ptr<SourceRawDataInput> source_;
  std::shared_ptr<TargetRawDataOutput> sink_;
  std::shared_ptr<BeautyFaceFilter> beauty_;
  std::shared_ptr<FaceReshapeFilter> reshape_;
  std::shared_ptr<LipstickFilter> lipstick_;
  std::shared_ptr<BlusherFilter> blusher_;
  std::vector<float> landmarks_;
  bool has_face_ = false;
  float eye_brighten_ = 0.f;
  float eye_redness_ = 0.f;
  uint8_t* output_ = nullptr;
  size_t output_cap_ = 0;
  bool output_ready_ = false;
  int width_ = 0;
  int height_ = 0;
  int channels_ = 4;
};

}  // namespace

extern "C" {

GpRrHandle gpupixel_rr_create(void) { return new Session(); }

int gpupixel_rr_initialize(GpRrHandle handle, const char* resource_dir) {
  if (!handle) return 0;
  return reinterpret_cast<Session*>(handle)->initialize(resource_dir) ? 1 : 0;
}

void gpupixel_rr_set_callbacks(GpRrHandle handle) {
  if (!handle) return;
  reinterpret_cast<Session*>(handle)->set_callbacks();
}

void gpupixel_rr_set_parameters(GpRrHandle handle, float beauty, float white,
                                float thin_face, float big_eye, float lipstick,
                                float blusher, float eye_brighten,
                                float eye_redness) {
  if (!handle) return;
  reinterpret_cast<Session*>(handle)->set_parameters(
      beauty, white, thin_face, big_eye, lipstick, blusher, eye_brighten,
      eye_redness);
}

void gpupixel_rr_set_parameters_v1(GpRrHandle handle, float beauty, float white,
                                   float thin_face, float big_eye, float lipstick,
                                   float blusher) {
  gpupixel_rr_set_parameters(handle, beauty, white, thin_face, big_eye, lipstick,
                             blusher, 0.f, 0.f);
}

uint8_t* gpupixel_rr_run(GpRrHandle handle, const uint8_t* rgba, int width,
                         int height, int channels) {
  if (!handle) return nullptr;
  return reinterpret_cast<Session*>(handle)->run(rgba, width, height, channels);
}

void gpupixel_rr_release_output(GpRrHandle handle) {
  if (!handle) return;
  reinterpret_cast<Session*>(handle)->release_output();
}

void gpupixel_rr_destroy(GpRrHandle handle) {
  delete reinterpret_cast<Session*>(handle);
}

}  // extern "C"
