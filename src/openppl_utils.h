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

#pragma once

#include "ppl/nn/models/onnx/onnx_runtime_builder_factory.h"
#include "ppl/nn/engines/cuda/cuda_engine_options.h"
#include "ppl/nn/engines/cuda/cuda_options.h"
#include "ppl/nn/engines/cuda/engine_factory.h"
#include "ppl/nn/common/logger.h"
#include <set>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

using namespace std;

namespace triton { namespace backend { namespace openppl {

#define RESPOND_ALL_AND_RETURN_IF_ERROR(                          \
    REQUESTS, REQUEST_COUNT, RESPONSES, S)                        \
  do {                                                            \
    TRITONSERVER_Error* raarie_err__ = (S);                       \
    if (raarie_err__ != nullptr) {                                \
      for (uint32_t r = 0; r < REQUEST_COUNT; ++r) {              \
        TRITONBACKEND_Response* response = (*RESPONSES)[r];       \
        if (response != nullptr) {                                \
          LOG_IF_ERROR(                                           \
              TRITONBACKEND_ResponseSend(                         \
                  response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                  raarie_err__),                                  \
              "failed to send OpenPPL backend response");     \
          response = nullptr;                                     \
        }                                                         \
        LOG_IF_ERROR(                                             \
            TRITONBACKEND_RequestRelease(                         \
                REQUESTS[r], TRITONSERVER_REQUEST_RELEASE_ALL),   \
            "failed releasing request");                          \
        REQUESTS[r] = nullptr;                                    \
      }                                                           \
      TRITONSERVER_ErrorDelete(raarie_err__);                     \
      return;                                                     \
    }                                                             \
  } while (false) 

extern const unique_ptr<ppl::nn::Runtime> runtime;

ppl::common::RetCode ReadFileContent(const char* fname, string* buf);

const char* MemMem(const char* haystack, unsigned int haystack_len, const char* needle, unsigned int needle_len);

void SplitString(const char* str, unsigned int len, const char* delim, unsigned int delim_len,
                        const function<bool(const char* s, unsigned int l)>& f);

bool ParseInputShapes(const string& shape_str, vector<vector<int64_t>>* input_shapes);

TRITONSERVER_DataType ConvertFromOpenPPLDataType(ppl::common::datatype_t data_type);

ppl::common::datatype_t ConvertToOpenPPLDataType(TRITONSERVER_DataType data_type);

}}}  // namespace triton::backend::openppl
