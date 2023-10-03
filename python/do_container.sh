apt update
apt-get install -y supervisor

cp /stella_vslam/python/check_openvslam.conf /etc/supervisor/conf.d/

/usr/bin/supervisord -c /etc/supervisor/supervisord.conf