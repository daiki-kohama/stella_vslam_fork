docker build -t openvslam_kohama -f Dockerfile.desktop . --build-arg NUM_THREADS=`expr $(nproc) - 1`
docker run -it --gpus all -e DISPLAY=$DISPLAY --name openvslam_kohama -v /Users/kohamadaiki/Documents/UCLab/kyudenko/repos_webservice/webservice-django/app/media:/stella_vslam/media -v /Users/kohamadaiki/Documents/UCLab/kyudenko/repos_webservice/webservice-cmm_process/openvslam/vocab:/stella_vslam/vocab:ro -v /Users/kohamadaiki/Documents/UCLab/kyudenko/repos_webservice/webservice-cmm_process/openvslam/configurations:/stella_vslam/configurations -v /Users/kohamadaiki/Documents/UCLab/kyudenko/repos_webservice/webservice-cmm_process/openvslam/python:/stella_vslam/python openvslam_kohama
