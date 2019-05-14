import os
import sys
import hashlib

if sys.version[0] == '2':
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

# Inspired by https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py

MB = 1024*1024
BUFSIZE = 10*MB

def verify(filePath, targetSHA):
    sha = hashlib.sha1()
    with open(filePath, 'rb') as f:
        while True:
            buf = f.read(BUFSIZE)
            if not buf:
                break
            sha.update(buf)
    return targetSHA == sha.hexdigest()


def checkOrDownload(filePath, url, sha):
    if not os.path.exists(filePath) or not verify(filePath, sha):
        print("%s doesn't exist. Downloading..." % filePath)
        print("URL: " + url)

        baseDir = os.path.dirname(filePath)
        if not os.path.exists(baseDir):
            os.makedirs(baseDir)

        r = urlopen(url)
        with open(filePath, 'wb') as f:
            sys.stdout.write('  progress ')
            sys.stdout.flush()
            while True:
                buf = r.read(BUFSIZE)
                if not buf:
                    break
                f.write(buf)
                sys.stdout.write('>')
                sys.stdout.flush()
        print('')

        if not verify(filePath, sha):
            print('Check sum failed!')
            exit(0)


def checkOrDownload_2019R1(filePath, sha):
    fileName = os.path.basename(filePath)
    modelName = fileName[:fileName.rfind('.')]
    baseUrl = 'https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/' + modelName + '/FP16/' + fileName
    checkOrDownload(filePath, baseUrl, sha)
