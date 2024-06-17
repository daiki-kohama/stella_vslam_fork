parent=`dirname $0`

docker build -t openvslam_kohama -f Dockerfile.desktop . --build-arg NUM_THREADS=`expr $(nproc) - 1`
docker run -it -d --gpus all -e DISPLAY=$DISPLAY --name openvslam_kohama -v $1:/stella_vslam/media -v ${parent}/vocab:/stella_vslam/vocab:ro -v ${parent}/configurations:/stella_vslam/configurations -v ${parent}/python:/stella_vslam/python openvslam_kohama
