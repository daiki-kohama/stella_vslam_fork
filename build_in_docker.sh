#!/bin/sh

CMAKE_INSTALL_PREFIX=/usr/local

mkdir -p /stella_vslam_ws/build && \
cd /stella_vslam_ws/build && \
cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DDETRMINISTIC=ON \
    -DUSE_ONNX_RUNTIME=ON \
    .. && \
make -j15 && \
make install

if [ $? -ne 0 ]; then
  echo "Error: some_command failed in stella_vslam_ws"
  exit 1
fi

mkdir -p /pangolin_viewer/build && \
cd /pangolin_viewer/build && \
cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
    .. && \
make -j15 && \
make install

if [ $? -ne 0 ]; then
  echo "Error: some_command failed in pangolin_viewer"
  exit 1
fi

cd /stella_vslam_examples/build && \
cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
    -DUSE_STACK_TRACE_LOGGER=ON \
    .. && \
make -j15
