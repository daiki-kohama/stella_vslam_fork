# -*- coding: utf-8 -*-
import threading
import os
import yaml
import subprocess
import shutil
import time
import datetime
import sys

def get_splitedFilePath(filePath):
    filePath_splited = filePath.split(".")
    if len(filePath_splited) == 2:
        return filePath_splited[0], "."+filePath_splited[1]
    else:
        fileName = ""
        for i in range(len(filePath_splited)-1):
            fileName = fileName + filePath_splited[i] + "."
        return fileName[:-1], "."+filePath_splited[-1]

def exec_openvslam(mapDir):
    print("PATH", os.path.join(mapDir, "order.yaml")) 
    with open(os.path.join(mapDir, "order.yaml"), "r") as file:
        orderData = yaml.safe_load(file)
    videoPath = orderData["video"]["path"]
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

    configPath = os.path.join("..", "configurations", "iPhone15Pro.yaml" if cameraModel == "perspective" else "config.yaml" )
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
    with open(os.path.join(mapDir, "config.yaml"), "w") as file:
        yaml.dump(configData, file, allow_unicode=True)
    
    #print(configData)
    #print("\n",configData.items())
    #print("./run_video_slam",  "-v", os.path.join("..", "vocab" ,"orb_vocab.fbow"), "-m", os.path.join("..", "media", "video", videoPath), "-c", os.path.join("..", "configurations", configyaml), "--frame-skip", str(frame_skip), "--log-level=debug", "-o", mapName)
    command = ["./run_video_slam -v " + os.path.join("..", "vocab" ,"orb_vocab.fbow") + " -m " + videoPath + " -c " + os.path.join(mapDir, "config.yaml") + " --frame-skip " + str(frame_skip) + " -s " + str(start_time) + " --log-level=debug -o " + mapName]
    #command = ["./run_video_slam", "-v", os.path.join("..", "vocab" ,"orb_vocab.fbow"), "-m", videoPath, "-c", os.path.join(mapDir, "config.yaml"), "--frame-skip", str(frame_skip), "-s", str(start_time), "--log-level=debug", "-o", mapName]
    exec = subprocess.Popen(command, shell=True, encoding='UTF-8', stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    with open(logPath, "a") as file:
        file.write("-------------------------------\n")
        file.write("./run_video_slam -v " + os.path.join("..", "vocab" ,"orb_vocab.fbow") + " -m " + videoPath + " -c " + os.path.join(mapDir, "config.yaml") + " --frame-skip " + str(frame_skip) + " -s " + str(start_time) + " --log-level=debug -o " + "map.msg")
        file.write("\n")
    while True:
        line = exec.stdout.readline()
        with open(logPath, "a") as file:
            file.write(line)
        if exec.poll() is not None:
            break
    with open(logPath, "a") as file:
        file.write("-------------------------------\n")
    #exec.wait()

    if exec.returncode == 0:
        while True:
            if os.path.exists(mapName):
                break
        shutil.move(mapName, os.path.join(mapDir, mapName))

    with open(os.path.join(mapDir, "order.yaml"), "r+") as file:
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
            pass
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
            t = threading.Thread(target=exec_openvslam, args=(start_exec_mapDir,))
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
