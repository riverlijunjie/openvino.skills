OPENVINO_HOME=`pwd`

export no_proxy=localhost,127.0.0.0/8,::1
export ftp_proxy=http://child-prc.intel.com:913/

export https_proxy=http://proxy-dmz.intel.com:912
export http_proxy=http://proxy-dmz.intel.com:911

git submodule init
git submodule update

mkdir -p $OPENVINO_HOME/build-x86_64-release
cd $OPENVINO_HOME/build-x86_64-release

#-GNinja

cmake \
            -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_SYSTEM_OPENCL=OFF \
            -DENABLE_LTO=OFF \
            -DENABLE_CPPLINT=OFF \
            -DTHREADING=TBB \
            -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
            -DENABLE_TESTS=ON \
            -DENABLE_PROFILING_ITT=FULL \
            -DENABLE_HETERO=OFF \
            -DENABLE_HETERO=OFF \
            -DENABLE_INTEL_NPU=OFF \
            -DENABLE_INTEL_CPU=ON \
            -DENABLE_SANITIZER=OFF \
            -DENABLE_FUNCTIONAL_TESTS=ON \
            -DNGRAPH_UNIT_TEST_ENABLE=ON \
            -DENABLE_OV_ONNX_FRONTEND=OFF \
            -DENABLE_DEBUG_CAPS:BOOL=ON \
            -DENABLE_PYTHON=ON \
            -DPYTHON_EXECUTABLE=`which python3.12` \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.12.so \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.12 \
            -DENABLE_WHEEL=OFF \
            -DENABLE_FASTER_BUILD=OFF \
            -DENABLE_STRICT_DEPENDENCIES=OFF \
            -DCMAKE_INSTALL_PREFIX="$OPENVINO_HOME/install_release" \
            ..

#DCMAKE_CXX_COMPILER_LAUNCHER=ccache 
#DCMAKE_C_COMPILER_LAUNCHER=ccache 

#make -j${nproc}
make -j 16
make install
cd $OPENVINO_HOME