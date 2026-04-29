
set TBB_DIR=C:\Users\Local_Admin\river\openvino\temp\tbb\bin

set OPENVINO_INSTALL=C:\Users\Local_Admin\river\openvino\release_install
set BUILD_DIR=C:\Users\Local_Admin\river\openvino.genai\build


set no_proxy=localhost,127.0.0.0/8,::1
set ftp_proxy=http://child-prc.intel.com:913/

set https_proxy=http://proxy-dmz.intel.com:912
set http_proxy=http://proxy-dmz.intel.com:911

set OpenVINO_DIR=%OPENVINO_INSTALL%\runtime\cmake


git submodule update --init --recursive

cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B %BUILD_DIR%
cmake --build ./build/ --config Release -j10

cmake --install ./build/ --config Release --prefix %OPENVINO_INSTALL%