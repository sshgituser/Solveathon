// Strict-GPU Qolda engine: RTMPose (SimCC) and the E02 forearm classifier.
//
// Two compiled graphs, GPU accelerator only, both proven fully delegated to a non-CPU
// accelerator before any Run. A session owns the environment, both models, the source
// byte buffers that LiteRT requires to outlive the models, and reusable tensor buffers.
// No CPU/NPU fallback, no file access, no video decoding: the caller supplies assets and
// float arrays.
#include <jni.h>

#include <android/log.h>

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "QOLDA_ENGINE", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "QOLDA_ENGINE", __VA_ARGS__)

namespace {

constexpr int kPoseInput = 1 * 256 * 192 * 3;
constexpr int kSimccX = 1 * 17 * 384;
constexpr int kSimccY = 1 * 17 * 512;
constexpr int kFeatures = 1 * 64 * 5;
// Reference node counts for this exact payload and LiteRT 2.4.0 (T-017, CPH2399).
constexpr int kPoseNodes = 278;
constexpr int kClassifierNodes = 60;

struct Graph {
  std::vector<uint8_t> bytes;  // LiteRT keeps a pointer into this buffer for the model's lifetime.
  LiteRtModel model = nullptr;
  LiteRtCompiledModel compiled = nullptr;
};

struct Session {
  LiteRtEnvironment env = nullptr;
  Graph pose;
  Graph classifier;
  bool ready = false;
  std::mutex mutex;
  // Reused buffers, sized from the verified shapes.
  LiteRtTensorBuffer pose_in = nullptr;
  std::vector<LiteRtTensorBuffer> pose_out;
  LiteRtTensorBuffer classifier_in = nullptr;
  LiteRtTensorBuffer classifier_out = nullptr;
};

void throw_engine(JNIEnv* env, const std::string& message) {
  jclass cls = env->FindClass("net/qolda/engine/EngineException");
  if (cls != nullptr) {
    env->ThrowNew(cls, message.c_str());
  }
}

std::string status_text(LiteRtStatus status) {
  return std::to_string(static_cast<int>(status));
}

LiteRtRankedTensorType make_type(const std::vector<int32_t>& dims) {
  LiteRtRankedTensorType type;
  std::memset(&type, 0, sizeof(type));
  type.element_type = kLiteRtElementTypeFloat32;
  type.layout.rank = static_cast<unsigned int>(dims.size());
  type.layout.has_strides = 0;
  for (size_t i = 0; i < dims.size(); ++i) type.layout.dimensions[i] = dims[i];
  return type;
}

bool check_signature(Graph* graph, const char* expected_name, const std::vector<int32_t>& shape,
                     const std::vector<std::pair<const char*, std::vector<int32_t>>>& outputs,
                     std::string* error) {
  LiteRtParamIndex signatures = 0;
  if (LiteRtGetNumModelSignatures(graph->model, &signatures) != kLiteRtStatusOk || signatures != 1) {
    *error = "expected exactly one signature";
    return false;
  }
  LiteRtSignature signature = nullptr;
  if (LiteRtGetModelSignature(graph->model, 0, &signature) != kLiteRtStatusOk || signature == nullptr) {
    *error = "signature is not readable";
    return false;
  }
  const char* key = nullptr;
  if (LiteRtGetSignatureKey(signature, &key) != kLiteRtStatusOk || key == nullptr ||
      std::strcmp(key, "serving_default") != 0) {
    *error = "unexpected signature key";
    return false;
  }
  LiteRtParamIndex inputs = 0, outs = 0;
  if (LiteRtGetNumSignatureInputs(signature, &inputs) != kLiteRtStatusOk || inputs != 1 ||
      LiteRtGetNumSignatureOutputs(signature, &outs) != kLiteRtStatusOk ||
      outs != static_cast<LiteRtParamIndex>(outputs.size())) {
    *error = "unexpected signature arity";
    return false;
  }
  const char* input_name = nullptr;
  if (LiteRtGetSignatureInputName(signature, 0, &input_name) != kLiteRtStatusOk || input_name == nullptr ||
      std::strcmp(input_name, expected_name) != 0) {
    *error = "unexpected input name";
    return false;
  }
  for (LiteRtParamIndex i = 0; i < outs; ++i) {
    const char* name = nullptr;
    if (LiteRtGetSignatureOutputName(signature, i, &name) != kLiteRtStatusOk || name == nullptr ||
        std::strcmp(name, outputs[i].first) != 0) {
      *error = std::string("unexpected output name at index ") + std::to_string(i);
      return false;
    }
  }
  // The shapes below are checked through the compiled model's buffer requirements, which is the
  // only public path for the tensor type in this LiteRT build.
  (void)shape;
  return true;
}

bool check_shapes(LiteRtCompiledModel compiled, const std::vector<int32_t>& input_shape,
                  const std::vector<std::vector<int32_t>>& output_shapes, std::string* error) {
  LiteRtTensorBufferRequirements requirements = nullptr;
  if (LiteRtGetCompiledModelInputBufferRequirements(compiled, 0, 0, &requirements) != kLiteRtStatusOk) {
    *error = "input requirements are missing";
    return false;
  }
  size_t expected = 1;
  for (int32_t dim : input_shape) expected *= static_cast<size_t>(dim);
  size_t size = 0;
  if (LiteRtGetTensorBufferRequirementsBufferSize(requirements, &size) != kLiteRtStatusOk ||
      size < expected * sizeof(float)) {
    *error = "input buffer size does not match the verified shape";
    return false;
  }
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    if (LiteRtGetCompiledModelOutputBufferRequirements(compiled, 0, i, &requirements) != kLiteRtStatusOk) {
      *error = "output requirements are missing";
      return false;
    }
    expected = 1;
    for (int32_t dim : output_shapes[i]) expected *= static_cast<size_t>(dim);
    if (LiteRtGetTensorBufferRequirementsBufferSize(requirements, &size) != kLiteRtStatusOk ||
        size < expected * sizeof(float)) {
      *error = "output buffer size does not match the verified shape";
      return false;
    }
  }
  return true;
}

bool verify_delegation(LiteRtCompiledModel compiled, int expected_nodes, const char* label,
                       std::string* error) {
  bool full = false;
  if (LiteRtCompiledModelIsNonCpuFullyAccelerated(compiled, &full) != kLiteRtStatusOk) {
    *error = std::string(label) + ": delegation status is unavailable";
    return false;
  }
  if (!full) {
    *error = std::string(label) + ": graph is not fully delegated to a non-CPU accelerator";
    return false;
  }
  LiteRtDelegationMetricsConst metrics = nullptr;
  if (LiteRtCompiledModelGetDelegationMetrics(compiled, &metrics) != kLiteRtStatusOk || metrics == nullptr) {
    *error = std::string(label) + ": delegation metrics are unavailable";
    return false;
  }
  int total = 0;
  LiteRtGetDelegationMetricsTotalNodeCount(metrics, &total);
  if (total != expected_nodes) {
    *error = std::string(label) + ": unexpected node count " + std::to_string(total);
    return false;
  }
  LiteRtParamIndex accelerators = 0;
  LiteRtGetNumDelegationMetricsAccelerators(metrics, &accelerators);
  int gpu_nodes = 0;
  for (LiteRtParamIndex i = 0; i < accelerators; ++i) {
    LiteRtAcceleratorDelegationMetricsConst acc = nullptr;
    if (LiteRtGetDelegationMetricsAccelerator(metrics, i, &acc) != kLiteRtStatusOk) continue;
    const char* name = "";
    int delegated = 0;
    LiteRtGetAcceleratorDelegationMetricsName(acc, &name);
    LiteRtGetAcceleratorDelegationMetricsDelegatedNodeCount(acc, &delegated);
    if (name != nullptr && std::strstr(name, "GPU") != nullptr) gpu_nodes += delegated;
  }
  if (gpu_nodes != total) {
    *error = std::string(label) + ": " + std::to_string(total - gpu_nodes) + " node(s) outside the GPU";
    return false;
  }
  LOGI("%s: %d/%d nodes on LiteRT GPU", label, gpu_nodes, total);
  return true;
}

bool create_options(LiteRtOptions* options, std::string* error) {
  if (LiteRtCreateOptions(options) != kLiteRtStatusOk) {
    *error = "LiteRtCreateOptions failed";
    return false;
  }
  if (LiteRtSetOptionsHardwareAccelerators(*options, kLiteRtHwAcceleratorGpu) != kLiteRtStatusOk) {
    *error = "GPU accelerator is not accepted";
    return false;
  }
  // FP32 through the opaque gpu_options TOML; this LiteRT build does not export the Lrt* setters.
  const char* toml = "precision = 2\n";
  char* payload = static_cast<char*>(std::malloc(std::strlen(toml) + 1));
  if (payload == nullptr) {
    *error = "GPU options payload allocation failed";
    return false;
  }
  std::memcpy(payload, toml, std::strlen(toml) + 1);
  LiteRtOpaqueOptions opaque = nullptr;
  const LiteRtStatus created = LiteRtCreateOpaqueOptions(
      "gpu_options", payload, [](void* data) { std::free(data); }, &opaque);
  if (created != kLiteRtStatusOk) {
    std::free(payload);
    *error = "gpu_options creation failed (" + status_text(created) + ")";
    return false;
  }
  const LiteRtStatus added = LiteRtAddOpaqueOptions(*options, opaque);
  if (added != kLiteRtStatusOk) {
    LiteRtDestroyOpaqueOptions(opaque);
    *error = "gpu_options were not accepted (" + status_text(added) + ")";
    return false;
  }
  return true;
}

bool compile(Graph* graph, LiteRtEnvironment env, jbyteArray source, JNIEnv* jni,
             const char* label, std::string* error) {
  const jsize size = jni->GetArrayLength(source);
  graph->bytes.resize(static_cast<size_t>(size));
  jni->GetByteArrayRegion(source, 0, size, reinterpret_cast<jbyte*>(graph->bytes.data()));
  if (LiteRtCreateModelFromBuffer(env, graph->bytes.data(), graph->bytes.size(), &graph->model) !=
      kLiteRtStatusOk || graph->model == nullptr) {
    *error = std::string(label) + ": model cannot be created from buffer";
    return false;
  }
  (void)label;
  return true;
}

void destroy_graph(Graph* graph) {
  if (graph->compiled != nullptr) {
    LiteRtDestroyCompiledModel(graph->compiled);
    graph->compiled = nullptr;
  }
  if (graph->model != nullptr) {
    LiteRtDestroyModel(graph->model);
    graph->model = nullptr;
  }
  graph->bytes.clear();
  graph->bytes.shrink_to_fit();
}

void destroy_session(Session* session) {
  for (auto* buffer : session->pose_out) {
    if (buffer != nullptr) LiteRtDestroyTensorBuffer(buffer);
  }
  session->pose_out.clear();
  if (session->pose_in != nullptr) LiteRtDestroyTensorBuffer(session->pose_in);
  if (session->classifier_out != nullptr) LiteRtDestroyTensorBuffer(session->classifier_out);
  if (session->classifier_in != nullptr) LiteRtDestroyTensorBuffer(session->classifier_in);
  session->pose_in = nullptr;
  session->classifier_in = nullptr;
  session->classifier_out = nullptr;
  destroy_graph(&session->pose);
  destroy_graph(&session->classifier);
  if (session->env != nullptr) LiteRtDestroyEnvironment(session->env);
  session->env = nullptr;
  session->ready = false;
  delete session;
}

LiteRtTensorBuffer create_buffer(LiteRtEnvironment env, LiteRtCompiledModel compiled, bool input,
                                 size_t index, const std::vector<int32_t>& shape, std::string* error) {
  LiteRtTensorBufferRequirements requirements = nullptr;
  const LiteRtStatus got = input
                               ? LiteRtGetCompiledModelInputBufferRequirements(compiled, 0, index, &requirements)
                               : LiteRtGetCompiledModelOutputBufferRequirements(compiled, 0, index, &requirements);
  if (got != kLiteRtStatusOk) {
    *error = "buffer requirements are missing";
    return nullptr;
  }
  LiteRtRankedTensorType type = make_type(shape);
  LiteRtTensorBuffer buffer = nullptr;
  if (LiteRtCreateManagedTensorBufferFromRequirements(env, &type, requirements, &buffer) !=
      kLiteRtStatusOk) {
    *error = "managed tensor buffer allocation failed";
    return nullptr;
  }
  return buffer;
}

Session* create_session(JNIEnv* jni, jbyteArray pose_bytes, jbyteArray classifier_bytes,
                        std::string* error) {
  auto* session = new Session();
  if (LiteRtCreateEnvironment(0, nullptr, &session->env) != kLiteRtStatusOk || session->env == nullptr) {
    *error = "LiteRT environment is unavailable";
    destroy_session(session);
    return nullptr;
  }
  if (!compile(&session->pose, session->env, pose_bytes, jni, "pose", error) ||
      !compile(&session->classifier, session->env, classifier_bytes, jni, "classifier", error)) {
    destroy_session(session);
    return nullptr;
  }
  if (!check_signature(&session->pose, "input", {1, 256, 192, 3}, {{"simcc_x", {1, 17, 384}},
                                                                 {"simcc_y", {1, 17, 512}}}, error) ||
      !check_signature(&session->classifier, "features", {1, 64, 5}, {{"logit", {1, 1}}}, error)) {
    destroy_session(session);
    return nullptr;
  }
  LiteRtOptions options = nullptr;
  if (!create_options(&options, error)) {
    destroy_session(session);
    return nullptr;
  }
  const LiteRtStatus pose_status =
      LiteRtCreateCompiledModel(session->env, session->pose.model, options, &session->pose.compiled);
  const LiteRtStatus classifier_status = LiteRtCreateCompiledModel(
      session->env, session->classifier.model, options, &session->classifier.compiled);
  LiteRtDestroyOptions(options);
  if (pose_status != kLiteRtStatusOk || session->pose.compiled == nullptr) {
    *error = "pose GPU compilation failed (" + status_text(pose_status) + ")";
    destroy_session(session);
    return nullptr;
  }
  if (classifier_status != kLiteRtStatusOk || session->classifier.compiled == nullptr) {
    *error = "classifier GPU compilation failed (" + status_text(classifier_status) + ")";
    destroy_session(session);
    return nullptr;
  }
  if (!verify_delegation(session->pose.compiled, kPoseNodes, "pose", error) ||
      !verify_delegation(session->classifier.compiled, kClassifierNodes, "classifier", error)) {
    destroy_session(session);
    return nullptr;
  }
  if (!check_shapes(session->pose.compiled, {1, 256, 192, 3}, {{1, 17, 384}, {1, 17, 512}}, error) ||
      !check_shapes(session->classifier.compiled, {1, 64, 5}, {{1, 1}}, error)) {
    destroy_session(session);
    return nullptr;
  }
  session->pose_in = create_buffer(session->env, session->pose.compiled, true, 0, {1, 256, 192, 3}, error);
  session->pose_out.push_back(
      create_buffer(session->env, session->pose.compiled, false, 0, {1, 17, 384}, error));
  session->pose_out.push_back(
      create_buffer(session->env, session->pose.compiled, false, 1, {1, 17, 512}, error));
  session->classifier_in =
      create_buffer(session->env, session->classifier.compiled, true, 0, {1, 64, 5}, error);
  session->classifier_out =
      create_buffer(session->env, session->classifier.compiled, false, 0, {1, 1}, error);
  if (session->pose_in == nullptr || session->pose_out[0] == nullptr || session->pose_out[1] == nullptr ||
      session->classifier_in == nullptr || session->classifier_out == nullptr) {
    destroy_session(session);
    return nullptr;
  }
  session->ready = true;
  return session;
}

bool copy_into(LiteRtTensorBuffer buffer, const float* data, size_t count, std::string* error) {
  void* locked = nullptr;
  if (LiteRtLockTensorBuffer(buffer, &locked, kLiteRtTensorBufferLockModeWrite) != kLiteRtStatusOk ||
      locked == nullptr) {
    *error = "input buffer lock failed";
    return false;
  }
  std::memcpy(locked, data, count * sizeof(float));
  if (LiteRtUnlockTensorBuffer(buffer) != kLiteRtStatusOk) {
    *error = "input buffer unlock failed";
    return false;
  }
  return true;
}

bool copy_out(LiteRtTensorBuffer buffer, float* data, size_t count, std::string* error) {
  void* locked = nullptr;
  if (LiteRtLockTensorBuffer(buffer, &locked, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk ||
      locked == nullptr) {
    *error = "output buffer lock failed";
    return false;
  }
  std::memcpy(data, locked, count * sizeof(float));
  if (LiteRtUnlockTensorBuffer(buffer) != kLiteRtStatusOk) {
    *error = "output buffer unlock failed";
    return false;
  }
  return true;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_net_qolda_engine_GpuEngine_nativeOpen(JNIEnv* jni, jobject /*thiz*/, jbyteArray pose_bytes,
                                           jbyteArray classifier_bytes) {
  if (pose_bytes == nullptr || classifier_bytes == nullptr) return 0;
  std::string error;
  Session* session = create_session(jni, pose_bytes, classifier_bytes, &error);
  if (session == nullptr) {
    LOGE("open failed: %s", error.c_str());
    throw_engine(jni, error.empty() ? "model_unavailable" : error);
    return 0;
  }
  return reinterpret_cast<jlong>(session);
}

extern "C" JNIEXPORT void JNICALL
Java_net_qolda_engine_GpuEngine_nativeInferPose(JNIEnv* jni, jobject /*thiz*/, jlong handle,
                                                jfloatArray input, jfloatArray simcc_x,
                                                jfloatArray simcc_y) {
  auto* session = reinterpret_cast<Session*>(handle);
  if (session == nullptr || !session->ready) {
    throw_engine(jni, "model_unavailable");
    return;
  }
  std::lock_guard<std::mutex> guard(session->mutex);
  jfloat* in = jni->GetFloatArrayElements(input, nullptr);
  jfloat* out_x = jni->GetFloatArrayElements(simcc_x, nullptr);
  jfloat* out_y = jni->GetFloatArrayElements(simcc_y, nullptr);
  std::string error;
  bool ok = in != nullptr && out_x != nullptr && out_y != nullptr;
  if (ok) ok = copy_into(session->pose_in, in, kPoseInput, &error);
  if (ok) {
    LiteRtTensorBuffer ins[] = {session->pose_in};
    LiteRtTensorBuffer outs[] = {session->pose_out[0], session->pose_out[1]};
    if (LiteRtRunCompiledModel(session->pose.compiled, 0, 1, ins, 2, outs) != kLiteRtStatusOk) {
      error = "pose Run failed";
      ok = false;
    }
  }
  if (ok) ok = copy_out(session->pose_out[0], out_x, kSimccX, &error);
  if (ok) ok = copy_out(session->pose_out[1], out_y, kSimccY, &error);
  if (in != nullptr) jni->ReleaseFloatArrayElements(input, in, JNI_ABORT);
  if (out_x != nullptr) jni->ReleaseFloatArrayElements(simcc_x, out_x, ok ? 0 : JNI_ABORT);
  if (out_y != nullptr) jni->ReleaseFloatArrayElements(simcc_y, out_y, ok ? 0 : JNI_ABORT);
  if (!ok) {
    // A GPU failure retires both graphs; the caller cannot silently continue on CPU.
    session->ready = false;
    LOGE("pose inference failed: %s", error.c_str());
    throw_engine(jni, error.empty() ? "model_unavailable" : error);
  }
}

extern "C" JNIEXPORT jfloat JNICALL
Java_net_qolda_engine_GpuEngine_nativeInferLogit(JNIEnv* jni, jobject /*thiz*/, jlong handle,
                                                 jfloatArray features) {
  auto* session = reinterpret_cast<Session*>(handle);
  if (session == nullptr || !session->ready) {
    throw_engine(jni, "model_unavailable");
    return 0.0f;
  }
  std::lock_guard<std::mutex> guard(session->mutex);
  jfloat* in = jni->GetFloatArrayElements(features, nullptr);
  float logit = 0.0f;
  std::string error;
  bool ok = in != nullptr;
  if (ok) ok = copy_into(session->classifier_in, in, kFeatures, &error);
  if (ok) {
    LiteRtTensorBuffer ins[] = {session->classifier_in};
    LiteRtTensorBuffer outs[] = {session->classifier_out};
    if (LiteRtRunCompiledModel(session->classifier.compiled, 0, 1, ins, 1, outs) != kLiteRtStatusOk) {
      error = "classifier Run failed";
      ok = false;
    }
  }
  if (ok) ok = copy_out(session->classifier_out, &logit, 1, &error);
  if (in != nullptr) jni->ReleaseFloatArrayElements(features, in, JNI_ABORT);
  if (!ok) {
    session->ready = false;
    LOGE("classifier inference failed: %s", error.c_str());
    throw_engine(jni, error.empty() ? "model_unavailable" : error);
    return 0.0f;
  }
  return logit;
}

extern "C" JNIEXPORT void JNICALL
Java_net_qolda_engine_GpuEngine_nativeClose(JNIEnv* /*jni*/, jobject /*thiz*/, jlong handle) {
  auto* session = reinterpret_cast<Session*>(handle);
  if (session == nullptr) return;
  std::lock_guard<std::mutex> guard(session->mutex);
  destroy_session(session);
}
