set no_proxy=localhost,127.0.0.0/8,::1
set ftp_proxy=http://child-prc.intel.com:913/

set https_proxy=http://proxy-dmz.intel.com:912
set http_proxy=http://proxy-dmz.intel.com:911

git submodule update --init --recursive

cd build
cmake -G "Visual Studio 17 2022" -Wno-dev -DENABLE_CPPLINT=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF -DCMAKE_BUILD_TYPE=Release -DENABLE_PROFILING_ITT=FULL -DENABLE_INTEL_GPU=ON -DENABLE_INTEL_CPU=ON -DENABLE_INTEL_GNA=OFF -DENABLE_MULTI=OFF -DENABLE_AUTO=ON -DENABLE_AUTO_BATCH=OFF -DENABLE_HETERO=OFF  -DENABLE_INTEL_NPU=OFF -DENABLE_TEMPLATE=ON -DENABLE_TEMPLATE_REGISTRATION=ON -DENABLE_OV_PADDLE_FRONTEND=OFF -DENABLE_OV_PYTORCH_FRONTEND=ON -DENABLE_PYTHON=ON  -DPython3_EXECUTABLE="C:\Users\Local_Admin\river\py310\Scripts\python.exe"  -DENABLE_WHEEL=OFF -DENABLE_TESTS=OFF -DENABLE_FUNCTIONAL_TESTS=OFF -DTHREADING=TBB -DENABLE_SYSTEM_TBB=ON -DENABLE_JS=OFF -DENABLE_OV_ONNX_FRONTEND=ON -DENABLE_DEBUG_CAPS=ON -DCMAKE_INSTALL_PREFIX=C:\Users\Local_Admin\river\openvino\release_install  ..
cmake --build . --config Release --parallel 16
cmake --install . --prefix C:\Users\Local_Admin\river\openvino\release_install --config Release

cd ..