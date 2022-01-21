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

namespace triton { namespace backend { namespace openppl {

ppl::common::RetCode ReadFileContent(const char* fname, string* buf) {
    ifstream ifile;

    ifile.open(fname, ios_base::in);
    if (!ifile.is_open()) {
        LOG(ERROR) << "open file[" << fname << "] failed.";
        return ppl::common::RC_NOT_FOUND;
    }

    stringstream ss;
    ss << ifile.rdbuf();
    *buf = ss.str();

    ifile.close();
    return ppl::common::RC_SUCCESS;
}

const char* MemMem(const char* haystack, unsigned int haystack_len, const char* needle,
                          unsigned int needle_len) {
    if (!haystack || haystack_len == 0 || !needle || needle_len == 0) {
        return nullptr;
    }

    for (auto h = haystack; haystack_len >= needle_len; ++h, --haystack_len) {
        if (memcmp(h, needle, needle_len) == 0) {
            return h;
        }
    }
    return nullptr;
}

void SplitString(const char* str, unsigned int len, const char* delim, unsigned int delim_len,
                        const function<bool(const char* s, unsigned int l)>& f) {
    const char* end = str + len;

    while (str < end) {
        auto cursor = MemMem(str, len, delim, delim_len);
        if (!cursor) {
            f(str, end - str);
            return;
        }

        if (!f(str, cursor - str)) {
            return;
        }

        cursor += delim_len;
        str = cursor;
        len = end - cursor;
    }

    f("", 0); // the last empty field
}

bool ParseInputShapes(const string& shape_str, vector<vector<int64_t>>* input_shapes) {
    bool ok = true;

    vector<string> input_shape_list;
    SplitString(shape_str.data(), shape_str.size(), ",", 1,
                [&ok, &input_shape_list](const char* s, unsigned int l) -> bool {
                    if (l > 0) {
                        input_shape_list.emplace_back(s, l);
                        return true;
                    }
                    LOG(ERROR) << "empty shape in option '--input-shapes'";
                    ok = false;
                    return false;
                });
    if (!ok) {
        return false;
    }

    for (auto x = input_shape_list.begin(); x != input_shape_list.end(); ++x) {
        ok = true;
        vector<int64_t> shape;
        SplitString(x->data(), x->size(), "_", 1, [&ok, &shape](const char* s, unsigned int l) -> bool {
            if (l > 0) {
                int64_t dim = atol(string(s, l).c_str());
                shape.push_back(dim);
                return true;
            }
            LOG(ERROR) << "illegal dim format.";
            ok = false;
            return false;
        });
        if (!ok) {
            return false;
        }

        input_shapes->push_back(shape);
    }

    return true;
}


TRITONSERVER_DataType ConvertFromOpenPPLDataType(
    ppl::common::datatype_t data_type) {
  switch (data_type) {
    case ppl::common::DATATYPE_UINT8:
		return TRITONSERVER_TYPE_UINT8;
    case ppl::common::DATATYPE_UINT16:
		return TRITONSERVER_TYPE_UINT16;
    case ppl::common::DATATYPE_UINT32:
		return TRITONSERVER_TYPE_UINT32;
    case ppl::common::DATATYPE_UINT64:
		return TRITONSERVER_TYPE_UINT64;
    case ppl::common::DATATYPE_INT8:
		return TRITONSERVER_TYPE_INT8;
    case ppl::common::DATATYPE_INT16:
		return TRITONSERVER_TYPE_INT16;
    case ppl::common::DATATYPE_INT32:
		return TRITONSERVER_TYPE_INT32;
    case ppl::common::DATATYPE_INT64:
		return TRITONSERVER_TYPE_INT64;
    case ppl::common::DATATYPE_FLOAT16:
		return TRITONSERVER_TYPE_FP16;
    case ppl::common::DATATYPE_FLOAT32:
		return TRITONSERVER_TYPE_FP32;
    case ppl::common::DATATYPE_FLOAT64:
		return TRITONSERVER_TYPE_FP64;
    case ppl::common::DATATYPE_BOOL:
		return TRITONSERVER_TYPE_BOOL;
    default:
      break;
  }

  return TRITONSERVER_TYPE_INVALID;
}

ppl::common::datatype_t ConvertToOpenPPLDataType(
    TRITONSERVER_DataType data_type) {
  switch (data_type) {
    case TRITONSERVER_TYPE_UINT8:
      return ppl::common::DATATYPE_UINT8;
    case TRITONSERVER_TYPE_UINT16:
      return ppl::common::DATATYPE_UINT16;
    case TRITONSERVER_TYPE_UINT32:
      return ppl::common::DATATYPE_UINT32;
    case TRITONSERVER_TYPE_UINT64:
      return ppl::common::DATATYPE_UINT64;
    case TRITONSERVER_TYPE_INT8:
      return ppl::common::DATATYPE_INT8;
    case TRITONSERVER_TYPE_INT16:
      return ppl::common::DATATYPE_INT16;
    case TRITONSERVER_TYPE_INT32:
      return ppl::common::DATATYPE_INT32;
    case TRITONSERVER_TYPE_INT64:
      return ppl::common::DATATYPE_INT64;
    case TRITONSERVER_TYPE_FP16:
      return ppl::common::DATATYPE_FLOAT16;
    case TRITONSERVER_TYPE_FP32:
      return ppl::common::DATATYPE_FLOAT32;
    case TRITONSERVER_TYPE_FP64:
      return ppl::common::DATATYPE_FLOAT64;
    case TRITONSERVER_TYPE_BOOL:
      return ppl::common::DATATYPE_BOOL;
    default:
      break;
  }

  return ppl::common::DATATYPE_UNKNOWN;
}

}}}  // namespace triton::backend::openppl
