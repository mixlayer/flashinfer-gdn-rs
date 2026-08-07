// Native end-to-end smoke harness for the first GDN TVM FFI artifact.

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <tvm/ffi/c_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class DynamicLibrary {
 public:
  explicit DynamicLibrary(const char* path) : path_(path) {
    handle_ = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (handle_ == nullptr) {
      throw std::runtime_error("dlopen(" + path_ + ") failed: " + dlerror());
    }
  }

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  ~DynamicLibrary() {
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
  }

  template <typename T>
  T symbol(const char* name) const {
    dlerror();
    void* value = dlsym(handle_, name);
    if (const char* error = dlerror(); error != nullptr) {
      throw std::runtime_error("dlsym(" + path_ + ", " + name + ") failed: " +
                               error);
    }
    return reinterpret_cast<T>(value);
  }

 private:
  std::string path_;
  void* handle_ = nullptr;
};

struct Tensor {
  Tensor(DLDataType dtype, std::vector<int64_t> dimensions)
      : shape(std::move(dimensions)), strides(shape.size()) {
    int64_t stride = 1;
    for (std::size_t index = shape.size(); index-- > 0;) {
      strides[index] = stride;
      stride *= shape[index];
    }
    const std::size_t bytes = static_cast<std::size_t>(stride) * dtype.bits / 8;
    check_cuda(cudaMalloc(&data, bytes), "cudaMalloc");
    check_cuda(cudaMemset(data, 0, bytes), "cudaMemset");
    tensor = DLTensor{
        data,
        DLDevice{kDLCUDA, 0},
        static_cast<int32_t>(shape.size()),
        dtype,
        shape.data(),
        strides.data(),
        0,
    };
  }

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&&) = delete;
  Tensor& operator=(Tensor&&) = delete;

  ~Tensor() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }

  void* data = nullptr;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLTensor tensor{};
};

TVMFFIAny tensor_argument(Tensor& tensor) {
  TVMFFIAny argument{};
  argument.type_index = kTVMFFIDLTensorPtr;
  argument.v_ptr = &tensor.tensor;
  return argument;
}

TVMFFIAny stream_argument(cudaStream_t stream) {
  TVMFFIAny argument{};
  argument.type_index = kTVMFFIOpaquePtr;
  argument.v_ptr = stream;
  return argument;
}

std::string byte_array_string(const TVMFFIByteArray& value) {
  if (value.data == nullptr || value.size == 0) {
    return {};
  }
  return std::string(value.data, value.size);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 5) {
      std::cerr << "usage: pretranspose_decode_smoke <module.so> <entry-symbol> "
                   "<libcute_dsl_runtime.so> <libtvm_ffi.so>\n";
      return 2;
    }

    DynamicLibrary cute_runtime(argv[3]);
    DynamicLibrary tvm_runtime(argv[4]);
    DynamicLibrary module(argv[1]);
    auto entry = module.symbol<TVMFFISafeCallType>(argv[2]);
    auto error_move =
        tvm_runtime.symbol<decltype(&TVMFFIErrorMoveFromRaised)>(
            "TVMFFIErrorMoveFromRaised");
    auto object_dec_ref = tvm_runtime.symbol<decltype(&TVMFFIObjectDecRef)>(
        "TVMFFIObjectDecRef");

    constexpr int64_t batch = 1;
    constexpr int64_t t = 1;
    constexpr int64_t h = 16;
    constexpr int64_t hv = 16;
    constexpr int64_t k = 128;
    constexpr int64_t v = 128;
    const DLDataType f32{kDLFloat, 32, 1};
    const DLDataType bf16{kDLBfloat, 16, 1};
    const DLDataType i32{kDLInt, 32, 1};

    Tensor h0(f32, {batch * hv, v, k});
    Tensor a_log(f32, {hv});
    Tensor a(bf16, {batch, t, hv});
    Tensor dt_bias(f32, {hv});
    Tensor q(bf16, {batch, t, h, k});
    Tensor key(bf16, {batch, t, h, k});
    Tensor value(bf16, {batch, t, hv, v});
    Tensor beta(bf16, {batch, t, hv});
    Tensor output(bf16, {batch, t, hv, v});
    Tensor h0_indices(i32, {batch});
    Tensor h0_out_indices(i32, {batch});
    Tensor cu_seqlens(i32, {batch + 1});

    std::vector<TVMFFIAny> arguments{
        tensor_argument(h0),
        tensor_argument(a_log),
        tensor_argument(a),
        tensor_argument(dt_bias),
        tensor_argument(q),
        tensor_argument(key),
        tensor_argument(value),
        tensor_argument(beta),
        tensor_argument(output),
        tensor_argument(h0_indices),
        tensor_argument(h0_out_indices),
        tensor_argument(cu_seqlens),
        stream_argument(nullptr),
    };
    TVMFFIAny result{};
    result.type_index = kTVMFFINone;
    const int status =
        entry(nullptr, arguments.data(), static_cast<int32_t>(arguments.size()), &result);
    if (status != 0) {
      TVMFFIObjectHandle error = nullptr;
      error_move(&error);
      std::string message = "TVM FFI call failed without an error object";
      if (error != nullptr) {
        const TVMFFIErrorCell* cell = TVMFFIErrorGetCellPtr(error);
        message = byte_array_string(cell->kind) + ": " +
                  byte_array_string(cell->message);
        object_dec_ref(error);
      }
      throw std::runtime_error(message);
    }

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    std::vector<std::uint8_t> host_output(batch * t * hv * v * 2);
    check_cuda(cudaMemcpy(host_output.data(), output.data, host_output.size(),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy output");
    if (!std::all_of(host_output.begin(), host_output.end(),
                     [](std::uint8_t value) { return value == 0; })) {
      throw std::runtime_error("zero-input smoke produced nonzero output");
    }

    std::cout << "launched " << argv[2]
              << " through TVM FFI; zero-input output verified\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

