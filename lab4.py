
import urllib.request 
import zipfile
import hashlib
import os.path
import os
import sys
import shutil
import tempfile
import itertools
import functools
import types
from pprint import pprint
import attr
import random 
import pandas as pd
import numpy as np
from shutil import copyfile


@attr.s(slots=True)
class Checksum(object):
    '''
    The checksum consists of the actual hash value (value)
    as well as a stringrepresenting the hashing algorithm.
    The validator enforces that the algorithm can only be one of the listed acceptable methods
    '''
    value = attr.ib() 
    kind = attr.ib(validator=lambda o, a, v: v in 'md5 sha1 sha224 sha256 sha384 sha512'.split())

@attr.s(slots=True)
class Path(object):
    checksum = attr.ib()  
    filepath = attr.ib()
    
@attr.s(slots=True)
class image(object):
    id= attr.ib()    
    path = attr.ib()

def fetch_url(url, sha256, prefix='.', checksum_blocksize=2**20, dryRun=False):
    """
    Download a url.    
    :param url: the url to the file on the web    
    :param sha256: the SHA-256 checksum. Used to determine if the file was previously downloaded.    
    :param prefix: directory to save the file    
    :param checksum_blocksize: blocksize to used when computing the checksum    
    :param dryRun: boolean indicating that calling this function should do nothing    
    :returns: the local path to the downloaded file    
    """
    if not os.path.exists(prefix):
        os.makedirs(prefix)
        
    local = os.path.join(prefix, os.path.basename(url))
    if dryRun: return local
    if os.path.exists(local):
        print ('Verifying checksum')        
        chk = hashlib.sha256()
        with open(local, 'rb') as fd:
            while True:                
                bits = fd.read(checksum_blocksize)
                if not bits: break
                chk.update(bits)
        if sha256 == chk.hexdigest():
            return local
        
    print ('Downloading', url)
    
    def report(sofar, blocksize, totalsize):
        msg ='{}%\r'.format(100* sofar * blocksize / totalsize, 100)
        sys.stderr.write(msg)    

    urllib.request.urlretrieve(url, local, report)
    return local

def prepare_dataset(url=None, sha256=None, prefix='.', skip=False):
    url = url 
    sha256 = sha256
    # local = fetch_url(url, sha256=sha256, prefix=prefix, dryRun=skip)
    dir = os.path.dirname(__file__)
    file = "NISTSpecialDatabase4GrayScaleImagesofFIGS.zip"
    local = os.path.join(dir, file)

    if not skip:
        print ('Extracting', local, 'to', prefix)
        with zipfile.ZipFile(local, 'r') as zip:
            zip.extractall(prefix)
    
    name, _ = os.path.splitext(local)
    return name

def locate_paths(path_md5list, prefix):
    with open(path_md5list) as fd:
        for line in itertools.imap(str.strip, fd):            
            parts = line.split()
            if not len(parts) ==2: continue
            md5sum, path = parts
            chksum = Checksum(value=md5sum, kind='md5')            
            filepath = os.path.join(prefix, path)
            yield Path(checksum=chksum, filepath=filepath)

def locate_images(paths):
    def predicate(path):
        _, ext = os.path.splitext(path.filepath)
        return ext in ['.png']
    
    for path in itertools.ifilter(predicate, paths):
        yield image(id=path.checksum.value, path=path)
        

def make_copy(prefix, num):
    path = f'{prefix}NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/'
    
def split_images_train_and_test(prefix):
    path = f'{prefix}NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/'
    train_dir = f'{prefix}train/'
    test_dir  = f'{prefix}test/'
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir ):
        os.makedirs(test_dir )
        
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file[-3:] == "png":
                img = f'{subdir}/{file}'
                num = int(file[1:4])
                
                if num <= 1500:
                    print(img)
                    copyfile(img, f'{train_dir}{file}')
                elif num > 1500:
                    copyfile(img, f'{train_dir}{file}')
                    



def main():
    prefix ='/tmp/fingerprint_example/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    dataprefix = prepare_dataset(prefix=prefix, skip=True)
    # md5listpath = os.path.join(prefix, 'NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/sd04_md5.lst')
    split_images_train_and_test(prefix)

    
    
if __name__ == "__main__":
    main()