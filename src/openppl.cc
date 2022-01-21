// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "openppl_utils.h"
#include "ppl/nn/utils/array.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

#include <vector>
#include <string>
#include <mutex>
#include <map>

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

using namespace ppl::nn;
using namespace ppl::common;
using namespace std;

//
// Openppl Backend that implements the TRITONBACKEND API.
//
namespace triton { namespace backend { namespace openppl {

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** model_state)
{
  try {
    *model_state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }
  
  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*model_state)->AutoCompleteConfig());

    triton::common::TritonJson::WriteBuffer json_buffer;
    (*model_state)->ModelConfig().Write(&json_buffer);

    TRITONSERVER_Message* message;
    RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(
        &message, json_buffer.Base(), json_buffer.Size()));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(
        triton_model, 1 /* config_version */, message));
  }
  return nullptr;
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // If the model configuration already specifies inputs and outputs
  // then don't perform any auto-completion.
  size_t input_cnt = 0;
  size_t output_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (ModelConfig().Find("input", &inputs)) {
      input_cnt = inputs.ArraySize();
    }

    triton::common::TritonJson::Value config_batch_inputs;
    if (ModelConfig().Find("batch_input", &config_batch_inputs)) {
      input_cnt += config_batch_inputs.ArraySize();
    }

    triton::common::TritonJson::Value outputs;
    if (ModelConfig().Find("output", &outputs)) {
      output_cnt = outputs.ArraySize();
    }
  }

  if ((input_cnt > 0) && (output_cnt > 0)) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("skipping model configuration auto-complete for '") +
         Name() + "': inputs and outputs already specified")
            .c_str());
    return nullptr;  // success
  }
  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() {};

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

  ModelState* StateForModel() {
    return model_state_;
  }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  TRITONSERVER_Error* RegisterCudaEngine(vector<unique_ptr<Engine>>* engines);
  TRITONSERVER_Error* SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, std::vector<const char*>* input_names,
      bool* cuda_copy);
  TRITONSERVER_Error* OpenPPLRun(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count);
  TRITONSERVER_Error* ReadOutputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  ModelState* model_state_;

  // Store input and output openppl tensers for all requests
  vector<unique_ptr<Engine>> engines_;
  vector<Engine*> engine_ptrs_;
  unique_ptr<OnnxRuntimeBuilder> builder_;
  unique_ptr<Runtime> runtime_;

};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state)
{
  GetCurrentLogger()->SetLogLevel(2); // Only prrint Error message for pplnn

  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    try {
      RegisterCudaEngine(&engines_);
    }
    catch (const BackendModelInstanceException& ex) {
      throw BackendModelException(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG, ("Register cuda engine fail.")));
    }
  } else { // Only support GPU right now
    throw BackendModelException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED, ("Only support GPU. Unsupport engine type input.")));
  }


  string version = std::to_string(model_state_->Version());
  string g_flag_onnx_model = model_state_->RepositoryPath() + "/" + version + "/model.onnx";
  LOG(INFO) << "begin to read onnx-model: " << g_flag_onnx_model;
  engine_ptrs_.resize(engines_.size());
  for (uint32_t i = 0; i < engines_.size(); ++i) {
      engine_ptrs_[i] = engines_[i].get();
  }

  builder_.reset(OnnxRuntimeBuilderFactory::Create());
  if (!builder_) {
      LOG(ERROR) << "create RuntimeBuilder failed.";
      throw BackendModelException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, ("create RuntimeBuilder failed.")));
  }

  auto status = builder_->Init(g_flag_onnx_model.c_str(), engine_ptrs_.data(), engine_ptrs_.size());
  if (status != RC_SUCCESS) {
      LOG(ERROR) << "create OnnxRuntimeBuilder failed: " << GetRetCodeStr(status);
      throw BackendModelException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, ("create OnnxRuntimeBuilder failed.")));
  }

  status = builder_->Preprocess();
  if (status != RC_SUCCESS) {
      LOG(ERROR) << "onnx preprocess failed: " << GetRetCodeStr(status);
      throw BackendModelException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, ("onnx preprocess failed: ")));
  }

  runtime_.reset(builder_->CreateRuntime());
  
  if (!runtime_) {
    LOG(ERROR) << "Init runtime fail.";
    throw BackendModelException(TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL, ("Init runtime fail.")));
  }
  LOG(INFO) << "***** create runtime *****";
}

TRITONSERVER_Error* 
ModelInstanceState::RegisterCudaEngine(vector<unique_ptr<Engine>> *engines)
{
  CudaEngineOptions options;
  options.device_id = DeviceId();
  options.mm_policy = CUDA_MM_BEST_FIT;
  auto cuda_engine = CudaEngineFactory::Create(options);

  // TODO: Use quick select for test
  cuda_engine->Configure(ppl::nn::CUDA_CONF_USE_DEFAULT_ALGORITHMS, true);

  // pass input shapes to cuda engine for further optimizations
  string g_flag_input_shapes = "1_3_224_224"; // TODO: use input dims
  if (!g_flag_input_shapes.empty()) {
      vector<vector<int64_t>> input_shapes;
      if (!ParseInputShapes(g_flag_input_shapes, &input_shapes)) {
        string message = "invalid input shapes[" + g_flag_input_shapes + "].";
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, message.data());
      }

      vector<utils::Array<int64_t>> dims(input_shapes.size());
      for (uint32_t i = 0; i < input_shapes.size(); ++i) {
          auto& arr = dims[i];
          arr.base = input_shapes[i].data();
          arr.size = input_shapes[i].size();
      }
      cuda_engine->Configure(ppl::nn::CUDA_CONF_SET_INPUT_DIMS, dims.data(), dims.size());
  }

  engines->emplace_back(unique_ptr<Engine>(cuda_engine));
  LOG(INFO) << "***** register CudaEngine *****";
  return nullptr;
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG(INFO) << "Process requests with count: " << request_count;
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = model_state_->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to Openppl backend for '" + Name() +
                  "'")
                  .c_str()));
      return;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                Name() + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
    return;
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  bool all_response_failed = false;

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }


  std::vector<const char*> input_names;
  bool cuda_copy = false;
  BackendInputCollector collector(
      requests, request_count, &responses, model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedInput(), CudaStream(), nullptr, nullptr, 0,
      HostPolicyName().c_str());

  RESPOND_ALL_AND_RETURN_IF_ERROR(
      requests, request_count, &responses,
      SetInputTensors(
          total_batch_size, requests, request_count, &responses, &collector,
          &input_names, &cuda_copy));

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(CudaStream());
  }
#endif

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  RESPOND_ALL_AND_RETURN_IF_ERROR(
    requests, request_count, &responses, OpenPPLRun(&responses, request_count));

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  RESPOND_ALL_AND_RETURN_IF_ERROR(
      requests, request_count, &responses,
      ReadOutputTensors(total_batch_size, requests, request_count, &responses));


  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send OpenPPL backend response");
    }
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  if (!all_response_failed) {
    // Report the entire batch statistics.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportBatchStatistics(
            TritonModelInstance(), total_batch_size, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting batch request statistics");
  }
}

TRITONSERVER_Error*
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    bool* cuda_copy)
{
  const int max_batch_size = model_state_->MaxBatchSize();

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));
  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));
    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape, &input_dims_count,
        nullptr, nullptr));

    input_names->emplace_back(input_name);
    std::vector<int64_t> batchn_shape;
    // For a ragged input tensor, the tensor shape should be
    // the flatten shape of the whole batch
    if (model_state_->IsInputRagged(input_name)) {
      batchn_shape = std::vector<int64_t>{0};
      for (size_t idx = 0; idx < request_count; idx++) {
        TRITONBACKEND_Input* input;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_RequestInput(requests[idx], input_name, &input));
        const int64_t* input_shape;
        uint32_t input_dims_count;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]), TRITONBACKEND_InputProperties(
                                      input, nullptr, nullptr, &input_shape,
                                      &input_dims_count, nullptr, nullptr));

        batchn_shape[0] += GetElementCount(input_shape, input_dims_count);
      }
    }
    // The shape for the entire input batch, [total_batch_size, ...]
    else {
      batchn_shape =
          std::vector<int64_t>(input_shape, input_shape + input_dims_count);
      if (max_batch_size != 0) {
        batchn_shape[0] = total_batch_size;
      }
    }

    std::vector<int64_t> input_dims = batchn_shape;

    if (input_dims.size() == 1) {
      for (size_t i = 1; i < input_dims_count; i++) {
        input_dims.push_back(input_shape[i]);
      }
    }

    // The input must be in contiguous CPU memory. Use appropriate
    // allocator info to bind inputs to the right device. .i.e bind inputs
    // to GPU if they are being provided on GPU.
    const char* input_buffer;
    size_t batchn_byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>>
        allowed_input_types;
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
      allowed_input_types = {{TRITONSERVER_MEMORY_GPU, DeviceId()},
                              {TRITONSERVER_MEMORY_CPU_PINNED, 0},
                              {TRITONSERVER_MEMORY_CPU, 0}};
    } else {
      allowed_input_types = {{TRITONSERVER_MEMORY_CPU_PINNED, 0},
                              {TRITONSERVER_MEMORY_CPU, 0}};
    }

    RETURN_IF_ERROR(collector->ProcessTensor(
        input_name, nullptr, 0, allowed_input_types, &input_buffer,
        &batchn_byte_size, &memory_type, &memory_type_id));

    // Alloc OpenPPL Tensor
    auto ppl_tensor = runtime_->GetInputTensor(input_idx);
    ppl_tensor->GetShape()->Reshape(input_dims);
    ppl_tensor->GetShape()->SetDataType(ConvertToOpenPPLDataType(input_datatype));
    ppl_tensor->GetShape()->SetDataFormat(DATAFORMAT_NDARRAY);
    ppl_tensor->ReallocBuffer();
    ppl_tensor->ConvertFromHost(input_buffer, *ppl_tensor->GetShape());
  }

  // Finalize...
  *cuda_copy |= collector->Finalize();
  return nullptr;
}


/////////////////////////////////////////////////////////////////////////////

TRITONSERVER_Error*
ModelInstanceState::OpenPPLRun(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count)
{
  auto status = runtime_->Run();
  if (status != ppl::common::RC_SUCCESS) {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Run failed.");
  } else {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Run OK.");
  }
  return nullptr;
}


TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->MaxBatchSize(),
      model_state_->TritonMemoryManager(), model_state_->EnablePinnedInput(),
      CudaStream());

  // Use to hold string output contents
  bool cuda_copy = false;
  std::pair<TRITONSERVER_MemoryType, int64_t> alloc_perference = {
      TRITONSERVER_MEMORY_GPU, 2};

  for (uint32_t i = 0; i < runtime_->GetOutputCount(); i++) {
    auto ppl_tensor = runtime_->GetOutputTensor(i);
    auto ppl_shape = ppl_tensor->GetShape();
    auto name = ppl_tensor->GetName();
    // const BatchOutput* batch_output = model_state_->FindBatchOutput(name);

    TRITONSERVER_DataType dtype = ConvertFromOpenPPLDataType(ppl_shape->GetDataType());

    std::vector<int64_t> batchn_shape;
    for (uint32_t j = 0; j < ppl_shape->GetDimCount(); j++) {
      batchn_shape.push_back(ppl_shape->GetDim(j));
    }

    responder.ProcessTensor(
              name, dtype, batchn_shape, reinterpret_cast<char*>(ppl_tensor->GetBufferPtr()),
              alloc_perference.first, alloc_perference.second);
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRITON_ENABLE_GPU
  return nullptr;
}

///////////////////////////////////////////////////////////////////////////

extern "C" {

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Triton TRITONBACKEND API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor) + " does not support '" + name +
         "' TRITONBACKEND API version: " +
         std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
         std::to_string(TRITONBACKEND_API_VERSION_MINOR))
            .c_str());
  }

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::openppl
