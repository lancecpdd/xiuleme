# XiuLeMe

高性能、非破坏性、GPU 加速的 RAW 图像编辑器。基于 Tauri（Rust）+ React，面向 Windows、macOS、Linux 与 Android。

> 应用标识：`com.xiuleme.app` · 产品名 / 窗口标题：**XiuLeMe** · 非破坏性旁车：`.rrdata`

### 上游关系

XiuLeMe 是基于开源项目 **[RapidRAW](https://github.com/CyberTimon/RapidRAW)**（作者 [CyberTimon](https://github.com/CyberTimon) / Timon Käch）的**修改与衍生作品**。本仓库在 RapidRAW 的架构与大量核心能力之上进行了品牌化、人像处理等扩展，并继续以 **GNU Affero General Public License v3.0（AGPL-3.0）** 发布。许可与致谢详见文末 [许可与致谢](#许可与致谢)。

## 特性概要

- 非破坏性编辑（`.rrdata` sidecar）
- wgpu GPU 渲染管线与丰富调整工具
- 蒙版、预设、导出与批处理
- 本地 AI 能力（蒙版 / 人像 / 降噪等；模型与 ONNX Runtime 使用内置或本地资源，不联网下载）
- 可选相机连线拍摄（`tethering` feature）

## 从源码运行

需要已安装 [Rust](https://www.rust-lang.org/tools/install) 与 [Node.js](https://nodejs.org/)。

```bash
npm install
npm start
```

### 构建资源准备

ONNX Runtime 动态库与 AI / 人像模型体积较大，已被 `.gitignore` 忽略，不在仓库中。`build.rs` 会校验 ONNX Runtime 的 SHA-256，缺失或不匹配时构建直接失败。

构建前先运行准备脚本，从私有制品目录补齐缺失文件并逐一校验哈希：

```bash
pwsh scripts/prepare-release-assets.ps1 -SourceDir <资源目录> -Target <平台>
```

`-Target` 可选值：`windows-x86_64`、`windows-aarch64`、`linux-x86_64`、`linux-aarch64`、`macos-x86_64`、`macos-aarch64`、`android-aarch64`（默认当前机器平台）。不传 `-SourceDir` 时只做校验。

| 文件 | 放置位置 | 来源目录中的位置 |
|---|---|---|
| `onnxruntime.dll` / `libonnxruntime.so` / `libonnxruntime.dylib` | `src-tauri/resources/` | `ort/<平台>/` |
| Android 的 `libonnxruntime.so` | `src-tauri/libs/arm64-v8a/` | `ort/android-aarch64/` |
| 通用 AI 模型（SAM、U2Net、Depth Anything、CLIP、降噪、LaMa）与 `clip_tokenizer.json` | `src-tauri/resources/` | 根目录 |
| 人像模型（清单与哈希见 `src-tauri/resources/portrait-models.json`） | `src-tauri/resources/` | 根目录 |

人像模型的来源与许可见 `src-tauri/resources/portrait-NOTICE.txt`，发布前须确认其中的授权状态。

### 发布构建

```bash
npm run tauri build
# 产物示例：src-tauri/target/release/XiuLeMe
```

### 代码签名

未签名的安装包在 Windows 上会触发 SmartScreen 警告，在 macOS 上会被 Gatekeeper 拦截。

- **Windows**：将代码签名证书导入本机证书库，然后在 `src-tauri/tauri.conf.json` 的 `bundle.windows` 中填入 `"certificateThumbprint": "<证书指纹>"`。摘要算法（`sha256`）和时间戳服务器已经配置好。
- **macOS**：在 Mac 上构建前设置以下环境变量，Tauri 会自动完成签名和公证：

  ```bash
  export APPLE_SIGNING_IDENTITY="Developer ID Application: <名称> (<TEAM_ID>)"
  export APPLE_ID="<Apple ID 邮箱>"
  export APPLE_PASSWORD="<App 专用密码>"
  export APPLE_TEAM_ID="<TEAM_ID>"
  ```

### Android

Android release 包需要签名：在 `src-tauri/gen/android/` 下放置 `keystore.properties`（参照同目录的 `keystore.properties.example`），然后执行：

```bash
npm run tauri android build
```

带相机连线支持（macOS / Linux 需先安装 libgphoto2）：

```bash
# macOS
brew install libgphoto2 pkg-config

# Linux (Debian/Ubuntu)
sudo apt-get install -y libgphoto2-dev pkg-config

npm run start:tethering
# 或：npm run tauri build -- --features tethering
```

## 命令行导出

```bash
xiuleme export /path/to/photos --output /path/to/output_dir --format jpeg --quality 90
xiuleme export /path/to/photo.raw --output /path/to/output.png --format png
xiuleme export /path/to/photos --output /path/to/output_dir --adjustments /path/to/preset.json
```

## 技术栈

| 层 | 技术 |
|---|---|
| 桌面框架 | Tauri v2 |
| 前端 | React + TypeScript |
| 渲染 | wgpu + WGSL |
| RAW 解码 | rawler（dnglab 系） |
| AI | ONNX Runtime（本地模型） |

## 许可与致谢

### 本项目许可

本项目（XiuLeMe）以 **GNU Affero General Public License v3.0（AGPL-3.0）** 许可发布，全文见根目录 [LICENSE](LICENSE)。

### 上游版权与归属

| 项目 | 说明 |
|---|---|
| [RapidRAW](https://github.com/CyberTimon/RapidRAW) | 上游开源 RAW 编辑器；作者 CyberTimon（Timon Käch）。本仓库基于其源代码修改与扩展。许可：**AGPL-3.0**。 |
| [RapidRAW-DngLab](https://github.com/CyberTimon/RapidRAW-DngLab) | 本项目 RAW 解码所依赖的 `rawler` 上游 fork（dnglab 系）。使用时须遵守其仓库声明的许可条款。 |

RapidRAW 选择 AGPL-3.0，是为确保其自身及衍生作品对社区保持开源；XiuLeMe 作为衍生作品，**同样以 AGPL-3.0 发布**，并保留对上游作者与项目的署名。

应用内「设置 → 特别鸣谢」亦列出 RapidRAW 及其他开源贡献者，与本节相互补充，不能替代本节的法律披露。

### AGPL-3.0 合规要点（分发与二次开发）

二次分发或基于本仓库继续开发时，请至少遵守以下义务（完整条款以 [LICENSE](LICENSE) 为准）：

1. **提供对应源代码**：分发本软件的二进制／安装包时，须向接收方提供与该构建对应的完整对应源代码（含构建脚本与你所作修改）。
2. **保持 copyleft**：基于本项目的衍生作品须继续以 AGPL-3.0（或上游允许的兼容方式）许可，不得将本项目核心代码闭源并入专有软件。
3. **保留声明**：保留版权声明、许可文本及本 README / NOTICE 中的归属与第三方说明。
4. **网络使用（AGPL §13）**：若以网络服务方式向用户提供本软件的修改版本，须按 AGPL 第 13 条向用户提供获取对应源代码的途径。

### 主要第三方组件与资源

下表为**关键**第三方，并非完整依赖清单。npm / Cargo 传递依赖的许可以各包自身声明及锁文件为准。

| 组件 / 资源 | 许可（摘要） | 说明 |
|---|---|---|
| [RapidRAW](https://github.com/CyberTimon/RapidRAW) | AGPL-3.0 | 上游应用架构与核心编辑能力 |
| [rawler / RapidRAW-DngLab](https://github.com/CyberTimon/RapidRAW-DngLab) | 见上游仓库 | RAW 解码基础 |
| Tauri、React、wgpu 等 | 各组件自有许可 | 桌面壳、前端与 GPU 渲染栈 |
| [spektrafilm](https://github.com/andreavolpato/spektrafilm)（Andrea Volpato） | CC BY-SA 4.0 | 内置胶片模拟 LUT |
| [NegPy](https://github.com/marcinz606/NegPy) | GPL-3.0 | 负片反转逻辑的灵感来源 |
| ONNX Runtime、人像 / 蒙版等 AI 模型、GPUPixel、OpenCV 等 | 见 NOTICE | 来源、版权与再分发条件见下方文件 |

**详细 NOTICE（须随发行物一并保留／查阅）：**

- 人像与相关 AI 资源：[src-tauri/resources/portrait-NOTICE.txt](src-tauri/resources/portrait-NOTICE.txt)（及其中引用的 MIT / Apache-2.0 许可全文文件）
- GPUPixel 运行时工具链：[tools/gpupixel-ref/NOTICE.md](tools/gpupixel-ref/NOTICE.md)

### 模型与再分发警示

[portrait-NOTICE.txt](src-tauri/resources/portrait-NOTICE.txt) 记录的是上游**公开许可声明**，**不构成**对全部权重、训练数据或再分发链的最终法律授权结论。

- 项目开源、非商用**并不自动**取得模型权重的再分发权。
- 部分条目在 NOTICE 中仍为 **pending-review（待确认）**；发布安装包或对外分发捆绑模型前，维护者须逐项核实适用许可与数据条款。
- ONNX Runtime 动态库与多数模型文件体积较大，默认由 `.gitignore` 排除；构建／发布前须按本文「构建资源准备」补齐并校验哈希。

### 商标

Adobe Lightroom 及其他第三方名称、标志均为其各自所有者的商标。提及 RapidRAW、XiuLeMe 或任何第三方项目，**不表示**存在官方隶属、赞助或背书关系（除非另有书面授权）。
