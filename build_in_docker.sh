#!/bin/sh

cd /stella_vslam_ws/build && \
cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DDETRMINISTIC=ON \
    .. && \
make -j15 && \
make install

if [ $? -ne 0 ]; then
  echo "Error: some_command failed"
  exit 1
fi

cd /stella_vslam_examples/build && \
cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
    -DUSE_STACK_TRACE_LOGGER=ON \
    .. && \
make -j15
