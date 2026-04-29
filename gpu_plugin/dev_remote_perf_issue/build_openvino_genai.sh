export OV_INSTALL_DIR=~/openvino/install_release

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OV_INSTALL_DIR}/runtime/3rdparty/tbb/lib
git submodule update --init



OPENVINO_INSTALL_DIR=${OV_INSTALL_DIR}
export OpenVINO_DIR=${OPENVINO_INSTALL_DIR}/runtime

source ${OPENVINO_INSTALL_DIR}/setupvars.sh

mkdir -p build

cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
cmake --build ./build/ --config Release -j10

cmake --install ./build/ --config Release --prefix ${OPENVINO_INSTALL_DIR}