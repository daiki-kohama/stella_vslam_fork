# -*- coding: utf-8 -*-
import threading
import os
import yaml
import subprocess
import shutil
import time
import datetime
import sys
import cv2

def get_splitedFilePath(filePath):
    filePath_splited = filePath.split(".")
    if len(filePath_splited) == 2:
        return filePath_splited[0], "."+filePath_splited[1]
    else:
        fileName = ""
        for i in range(len(filePath_splited)-1):
            fileName = fileName + filePath_splited[i] + "."
        return fileName[:-1], "."+filePath_splited[-1]

def exec_openvslam(media_root, mapDir):
    mapDirReal = os.path.join(media_root, mapDir)
    print("PATH", os.path.join(mapDirReal, "order.yaml")) 
    with open(os.path.join(mapDirReal, "order.yaml"), "r") as file:
        orderData = yaml.safe_load(file)
    videoPath = orderData["video"]["path"]
    videoPathReal = os.path.join(media_root, videoPath)
    cameraModel = orderData["video"]["cameraModel"]
    model = orderData["video"]["model"]
    fps = orderData["video"]["fps"]
    cols = orderData["video"]["cols"]
    rows = orderData["video"]["rows"]
    frame_skip = orderData["openvslam"]["params"]["frame_skip"]
    keyframe_maxinterval = orderData["openvslam"]["params"]["keyframe_maxinterval"]
    start_time = orderData["openvslam"]["params"]["start_time"]
    mapName = orderData["openvslam"]["mapName"]
    logPath = orderData["openvslam"]["logPath"]
    logPathReal = os.path.join(media_root, logPath)
    use_sharpened_video = orderData["openvslam"]["use_sharpened_video"]

    if os.path.islink(videoPathReal):
        realpath = os.path.realpath(videoPathReal)
        videoPathReal = os.path.join("..", "local", *(realpath.split(os.sep)[3:]))

    if cameraModel == "equirectangular":
        configPath = os.path.join("..", "configurations", "config.yaml")
    elif cameraModel == "perspective":
        if "iPhone" in model:
            if "15" in model and "Pro" in model:
                if rows == 1920:
                    configPath = os.path.join("..", "configurations", "iPhone15ProHD.yaml")
            elif "12" in model and "Pro" in model:
                if rows == 1920:
                    configPath = os.path.join("..", "configurations", "iPhone12ProHD.yaml")
                elif rows == 3840:
                    configPath = os.path.join("..", "configurations", "iPhone12Pro4k.yaml")
            else:
                configPath = os.path.join("..", "configurations", "iPhone15ProHD.yaml")
    #if not os.path.exists(os.path.join("..", "configurations", configyaml)):
    with open(configPath) as file:
        configData = yaml.safe_load(file)
    configData["Camera"]["name"] = model
    configData["Camera"]["cols"] = cols
    configData["Camera"]["rows"] = rows
    configData["Camera"]["fps"] = fps
    if keyframe_maxinterval == 0:
        configData.pop("KeyframeInserter")
    else:
        configData["KeyframeInserter"]["max_interval"] = keyframe_maxinterval
    with open(os.path.join(mapDirReal, "config.yaml"), "w") as file:
        yaml.dump(configData, file, allow_unicode=True)
    
    if use_sharpened_video:
        wait_time = 0
        while True:
            try:
                cap = cv2.VideoCapture(videoPathReal)
                if cap.isOpened():
                    break
            except:
                pass
            time.sleep(1)
            if wait_time % 60 == 0:
                with open(logPathReal, "a") as file:
                    file.write(f"waiting about {wait_time//60} minutes for sharpened video {videoPathReal}\n")
            wait_time += 1

    command = ["./run_video_slam -v " + os.path.join("..", "vocab" ,"orb_vocab.fbow") + " -m " + videoPathReal + " -c " + os.path.join(mapDirReal, "config.yaml") + " --frame-skip " + str(frame_skip) + " -s " + str(start_time) + " --log-level=debug -o " + mapName]
    exec = subprocess.Popen(command, shell=True, encoding='UTF-8', stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    with open(logPathReal, "a") as file:
        file.write("-------------------------------\n")
        file.write(" ".join(command))
        file.write("\n")
    while True:
        line = exec.stdout.readline()
        with open(logPathReal, "a") as file:
            file.write(line)
        if exec.poll() is not None:
            break
    with open(logPathReal, "a") as file:
        file.write("-------------------------------\n")
    #exec.wait()

    if exec.returncode == 0:
        while True:
            if os.path.exists(mapName):
                break
        shutil.move(mapName, os.path.join(mapDirReal, mapName))

    with open(os.path.join(mapDirReal, "order.yaml"), "r+") as file:
        orderData = yaml.safe_load(file)
        orderData["openvslam"]["did"] = True
        file.seek(0)
        yaml.dump(orderData, file, allow_unicode=True)

    with open(os.path.join("..", "media", "check", "openvslam.txt"), "r") as file:
        mapDirs = file.readlines()
        mapDirs.remove(mapDir+"\n")
    with open(os.path.join("..", "media", "check", "openvslam.txt"), "w") as file:
        file.writelines(mapDirs)


if __name__ == "__main__":
    
    checkDir = os.path.join("..", "media", "check")     #要変更
    if not os.path.exists(checkDir):
        os.makedirs(checkDir)
        os.chmod(checkDir, 0o777)
    if not os.path.exists(os.path.join(checkDir, "openvslam.txt")):
        with open(os.path.join(checkDir, "openvslam.txt"), "w") as file:
            file.write("")
        os.chmod(os.path.join(checkDir, "openvslam.txt"), 0o777)
    exec_mapDirs = []
    print(datetime.datetime.now() + datetime.timedelta(hours=9))
    print("start check_openvslam.py")
    while True:
        start_exec_mapDirs = []
        with open(os.path.join(checkDir, "openvslam.txt"), "r") as file:
            mapDirs = file.readlines()
        for i in range(len(mapDirs)):
            mapDirs[i] = mapDirs[i][:-1]
        for mapDir in mapDirs:
            if mapDir not in exec_mapDirs and len(mapDir)!=0:
                start_exec_mapDirs.append(mapDir)
        
        for start_exec_mapDir in start_exec_mapDirs:
            exec_mapDirs.append(start_exec_mapDir)
            t = threading.Thread(target=exec_openvslam, args=(os.path.join("..", "media"), start_exec_mapDir,))
            t.start()
            print(datetime.datetime.now() + datetime.timedelta(hours=9))
            print("start VSLAM mapDir :", start_exec_mapDir)

        for exec_mapDir in exec_mapDirs:
            if exec_mapDir not in mapDirs:
                exec_mapDirs.remove(exec_mapDir)
                print(datetime.datetime.now() + datetime.timedelta(hours=9))
                print("end VSLAM mapDir:", exec_mapDir)
        
        #print(datetime.datetime.now() + datetime.timedelta(hours=9))
        #print("start_exec_mapDirs:", start_exec_mapDirs)
        #print("exec_mapDirs:", exec_mapDirs)
        #print("\n")

        time.sleep(3)
