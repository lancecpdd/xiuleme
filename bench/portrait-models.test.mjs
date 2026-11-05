import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

const resources = resolve(
  process.env.PORTRAIT_RESOURCE_DIR || fileURLToPath(new URL('../src-tauri/resources', import.meta.url)),
);
const manifest = JSON.parse(readFileSync(resolve(resources, 'portrait-models.json'), 'utf8'));

// 仅解析 ONNX protobuf 的元数据字段，不依赖 Python、不执行或修改模型。
function* fields(bytes) {
  let offset = 0;
  const varint = () => {
    let value = 0;
    let shift = 0;
    for (let i = 0; i < 10; i++) {
      assert.ok(offset < bytes.length, '截断的 protobuf');
      const byte = bytes[offset++];
      value += (byte & 127) * 2 ** shift;
      if (!(byte & 128)) return value;
      shift += 7;
    }
    throw new Error('非法 varint');
  };
  while (offset < bytes.length) {
    const tag = varint();
    const number = Math.floor(tag / 8);
    const wire = tag & 7;
    if (wire === 0) {
      yield [number, varint()];
    } else {
      const length = wire === 2 ? varint() : wire === 1 ? 8 : wire === 5 ? 4 : -1;
      assert.ok(length >= 0 && offset + length <= bytes.length, '非法字段长度');
      yield [number, bytes.subarray(offset, offset + length)];
      offset += length;
    }
  }
}
const field = (bytes, key) => [...fields(bytes)].find(([number]) => number === key)?.[1];
const text = (bytes) => bytes?.toString('utf8');
const dimension = (actual, expected) => {
  if (typeof actual === 'number') assert.equal(actual, expected);
  else assert.ok(typeof actual === 'string' && actual.length > 0, '动态维度必须有符号名称');
};
function tensorInfo(bytes) {
  const tensor = field(field(bytes, 2), 1);
  const shape = field(tensor, 2);
  return {
    name: text(field(bytes, 1)),
    dtype: field(tensor, 1),
    shape: [...fields(shape)].map(([, dim]) => field(dim, 1) ?? text(field(dim, 2))),
  };
}

test('安装资源随附许可、来源声明及待确认状态', () => {
  for (const name of ['portrait-NOTICE.txt', 'portrait-MIT-LICENSE.txt', 'portrait-Apache-2.0-LICENSE.txt']) {
    assert.ok(existsSync(resolve(resources, name)), `缺少随附许可文件 ${name}`);
  }
  const notice = readFileSync(resolve(resources, 'portrait-NOTICE.txt'), 'utf8');
  for (const term of [
    'RetinaFace',
    'Face Landmarker',
    'BiSeNet',
    'Selfie',
    'BlazeFace',
    'MIT License',
    'Apache License 2.0',
    '待确认',
  ]) {
    assert.ok(notice.includes(term), `来源声明缺少 ${term}`);
  }
  for (const model of manifest.models) {
    assert.ok(notice.includes(model.source), `声明缺少 ${model.id} 下载来源`);
  }
  const mit = readFileSync(resolve(resources, 'portrait-MIT-LICENSE.txt'), 'utf8');
  assert.ok(mit.includes('Copyright (c) 2024 Yakhyokhuja Valikhujaev'));
  const apache = readFileSync(resolve(resources, 'portrait-Apache-2.0-LICENSE.txt'), 'utf8');
  assert.ok(apache.includes('Version 2.0, January 2004') && apache.includes('END OF TERMS AND CONDITIONS'));
});

test('安装资源不含原始训练权重与已移除的 3DMM 资产', () => {
  // tauri.conf.json 整目录打包 resources；原始权重放 .research/weights。
  const forbidden = readdirSync(resources).filter(
    (name) => /\.(pth|pt|pb)$/i.test(name) || /^face_3dmm/i.test(name) || /^bfm/i.test(name),
  );
  assert.deepEqual(forbidden, [], `resources/ 不应包含：${forbidden.join(', ')}`);
});

test('重塑资源：审批绑定与代码常量一致，不暗用旧美肤资产', () => {
  assert.equal(manifest.version, 1);
  assert.equal(manifest.reshape.version, 1);
  const approved = manifest.reshape.approvedModels;
  assert.deepEqual(
    approved.map((binding) => [binding.capability, binding.modelId, binding.adapter]),
    [
      ['landmarks', 'landmarker', 'face-mesh-478-v1'],
      ['protection', 'skin', 'selfie-multiclass-logits-v1'],
      ['detection', 'detector_blazeface', 'blazeface-192-anchors-v1'],
    ],
  );
  for (const binding of approved) {
    assert.equal(binding.approvalStatus, 'approved');
    assert.ok(binding.reviewer.trim().length > 0, `${binding.capability} 缺少审批人`);
    assert.ok(binding.evidence.trim().length > 0, `${binding.capability} 缺少审批证据`);
  }
  const source = readFileSync(new URL('../src-tauri/src/portrait_models.rs', import.meta.url), 'utf8');
  for (const [constant, id] of [
    ['RESHAPE_LANDMARK_HASH', 'landmarker'],
    ['RESHAPE_PROTECTION_HASH', 'skin'],
    ['RESHAPE_DETECTION_HASH', 'detector_blazeface'],
  ]) {
    const match = source.match(new RegExp(`const ${constant}: &str =\\s*"([a-f0-9]{64})"`));
    assert.ok(match, `缺少适配器权重绑定 ${constant}`);
    assert.equal(match[1], manifest.models.find((model) => model.id === id).sha256);
  }
});

test('瘦手臂资源：姿态与语义解析审批、哈希和适配器一致', () => {
  assert.equal(manifest.body.version, 1);
  assert.deepEqual(
    manifest.body.approvedModels.map((binding) => [
      binding.capability,
      binding.modelId,
      binding.adapter,
      binding.approvalStatus,
    ]),
    [
      ['body-pose', 'body_pose_rtmpose_s', 'rtmpose-s-simcc-v1', 'approved'],
      ['arm-parsing', 'arm_parsing_schp_pascal7', 'schp-pascal7-int8-v1', 'approved'],
    ],
  );
  const source = readFileSync(
    new URL('../native/portrait-engine/src/feature_slim_arms.cpp', import.meta.url),
    'utf8',
  );
  for (const [constant, id] of [
    ['kPoseSha', 'body_pose_rtmpose_s'],
    ['kParsingSha', 'arm_parsing_schp_pascal7'],
  ]) {
    const match = source.match(new RegExp(`constexpr char ${constant}\\[\\] = \"([a-f0-9]{64})\"`));
    assert.ok(match, `缺少原生引擎瘦手臂权重绑定 ${constant}`);
    assert.equal(match[1], manifest.models.find((model) => model.id === id).sha256);
  }
});

for (const model of manifest.models) {
  test(`内置模型完整性和张量契约：${model.id}`, () => {
    const filename = [model.filename, ...model.aliases].map((name) => resolve(resources, name)).find(existsSync);
    assert.ok(filename, `缺少 ${model.filename}（接受别名：${model.aliases.join(', ')}）`);
    const bytes = readFileSync(filename);
    assert.equal(createHash('sha256').update(bytes).digest('hex'), model.sha256, '模型版本或文件完整性不匹配');
    const graph = field(bytes, 7);
    assert.ok(graph, '不是有效的 ONNX ModelProto');
    const graphFields = [...fields(graph)];
    const inputs = graphFields.filter(([n]) => n === 11).map(([, v]) => tensorInfo(v));
    const outputs = graphFields.filter(([n]) => n === 12).map(([, v]) => tensorInfo(v));
    assert.equal(inputs.length, model.id === 'retouch_inpainting' ? 2 : 1);
    for (const input of inputs) assert.equal(input.dtype, 1, '必须使用 float32 输入');
    for (const output of outputs) {
      assert.equal(output.dtype, model.id === 'retouch_skin' && output.name === 'output_png:0' ? 2 : 1);
    }
    console.log(model.id, bytes.length, JSON.stringify({ inputs, outputs }));
    if (model.id === 'retouch_skin') {
      assert.equal(inputs[0].name, 'input_image:0');
      assert.equal(inputs[0].shape.length, 3);
      assert.equal(inputs[0].shape[2], 3);
      assert.deepEqual(
        outputs.map((o) => o.name),
        ['output_png:0', 'if_person:0'],
      );
      assert.equal(outputs[0].shape.length, 2);
      assert.equal(outputs[1].shape.length, 0);
    } else if (model.id.startsWith('retouch_')) {
      assert.deepEqual(
        inputs.map((i) => i.name),
        model.id === 'retouch_inpainting' ? ['input', 'mask'] : ['input'],
      );
      for (const [i, input] of inputs.entries()) {
        assert.equal(input.shape.length, 4);
        input.shape.forEach((d, axis) => dimension(d, [1, i === 0 ? 3 : 1, 512, 512][axis]));
      }
      assert.equal(outputs.length, 1);
      assert.equal(outputs[0].name, 'output');
      assert.equal(outputs[0].shape.length, 4);
      outputs[0].shape.forEach((d, axis) =>
        dimension(d, [1, model.id === 'retouch_detection' ? 1 : 3, 512, 512][axis]),
      );
    } else if (model.id === 'parsing') {
      assert.deepEqual(inputs[0].shape.slice(1), [3, 512, 512]);
      assert.deepEqual(
        outputs.map((o) => o.name),
        ['output', '566', '576'],
      );
      for (const output of outputs) {
        assert.equal(output.shape.length, 4);
        output.shape.forEach((d, i) => dimension(d, [1, 19, 512, 512][i]));
      }
    } else if (model.id === 'landmarker') {
      assert.equal(inputs[0].name, 'input');
      assert.deepEqual(inputs[0].shape.slice(1), [3, 256, 256]);
      assert.deepEqual(
        outputs.map((o) => o.name),
        ['landmarks', 'score'],
      );
      assert.equal(outputs[0].shape.length, 3);
      outputs[0].shape.forEach((d, i) => dimension(d, [1, 478, 3][i]));
      assert.equal(outputs[1].shape.length, 2);
      outputs[1].shape.forEach((d) => dimension(d, 1));
    } else if (model.id === 'skin') {
      assert.equal(inputs[0].name, 'input_29');
      assert.deepEqual(inputs[0].shape, [1, 256, 256, 3]);
      assert.deepEqual(
        outputs.map((o) => o.name),
        ['Identity'],
      );
      assert.deepEqual(outputs[0].shape, [1, 256, 256, 6]);
      const nodes = graphFields
        .filter(([n]) => n === 1)
        .map(([, v]) => ({
          op: text(field(v, 4)),
          inputs: [...fields(v)].filter(([n]) => n === 1).map(([, s]) => text(s)),
          outputs: [...fields(v)].filter(([n]) => n === 2).map(([, s]) => text(s)),
        }));
      const tail = [];
      let name = outputs[0].name;
      for (let i = 0; i < 6; i++) {
        const node = nodes.find((n) => n.outputs.includes(name));
        if (!node) break;
        tail.push(node.op);
        name = node.inputs[0];
      }
      console.log('皮肤输出反向运算链', tail);
      assert.deepEqual(tail.slice(0, 2), ['Transpose', 'Conv']);
    } else if (model.id === 'detector_blazeface') {
      assert.equal(inputs[0].name, 'input');
      inputs[0].shape.forEach((d, axis) => dimension(d, [1, 192, 192, 3][axis]));
      assert.deepEqual(
        outputs.map((o) => o.name),
        ['reshaped_regressor_face_4', 'reshaped_classifier_face_4'],
      );
      assert.deepEqual(outputs[0].shape, [1, 2304, 16]);
      assert.deepEqual(outputs[1].shape, [1, 2304, 1]);
    } else if (model.id === 'body_pose_rtmpose_s') {
      assert.equal(inputs[0].name, 'input');
      assert.deepEqual(inputs[0].shape, [1, 3, 256, 192]);
      assert.deepEqual(
        outputs.map((o) => o.name),
        ['simcc_x', 'simcc_y'],
      );
      assert.deepEqual(outputs[0].shape, [1, 17, 384]);
      assert.deepEqual(outputs[1].shape, [1, 17, 512]);
    } else if (model.id === 'arm_parsing_schp_pascal7') {
      assert.equal(inputs[0].name, 'pixel_values');
      assert.equal(inputs[0].shape.length, 4);
      inputs[0].shape.forEach((d, axis) => dimension(d, [1, 3, 512, 512][axis]));
      assert.deepEqual(
        outputs.map((o) => o.name),
        ['logits'],
      );
      outputs[0].shape.forEach((d, axis) => dimension(d, [1, 7, 512, 512][axis]));
    } else {
      assert.equal(inputs[0].shape[1], 3);
      assert.equal(inputs[0].shape.length, 4);
      assert.deepEqual(
        outputs.map((o) => o.name),
        ['loc', 'conf', 'landmarks'],
      );
      outputs.forEach((output, i) => {
        assert.equal(output.shape.length, 3);
        dimension(output.shape.at(-1), [4, 2, 10][i]);
      });
    }
  });
}
