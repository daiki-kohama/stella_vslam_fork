docker build -t openvslam_kohama -f Dockerfile.desktop . --build-arg NUM_THREADS=`expr $(nproc) - 1`
docker run -it --gpus all -e DISPLAY=$DISPLAY --name openvslam_kohama -v ~/webService/django/app/media:/stella_vslam/media -v ~/webService/cmm_process/openvslam/vocab:/stella_vslam/vocab:ro -v ~/webService/cmm_process/openvslam/configurations:/stella_vslam/configurations openvslam_kohama
