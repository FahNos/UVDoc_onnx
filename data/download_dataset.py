import os
import ssl
import urllib.request
from tqdm import tqdm
import yaml
from argparse import ArgumentParser
import shutil
from pathlib import Path
import json
from typing import List, Dict, Tuple
# import magic
import numpy as np

import zipfile
import tarfile
import os
import io

def get_dataset_info(cfg):
    download_urls = cfg.get("download_links", [])
    filenames = cfg.get("filenames", [])   
    check_validity = cfg.get("check_validity", True)
    return download_urls, filenames, check_validity

def urlretrieve(url, filename, chunk_size=1024 * 32, check_validity=True):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    ctx = ssl.create_default_context()
    if not check_validity:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, context=ctx) as response:
        with open(filename, "wb") as fh, tqdm(total=response.length, unit="B", unit_scale=True) as pbar:
            while chunk := response.read(chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))

def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    


def check_archive_file(file_path):  

    if not os.path.exists(file_path):
        return False
   
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:            
            pass
        return True
    except zipfile.BadZipFile:     
        pass
    except Exception as e:       
        return False 

  
    try:
        with tarfile.open(file_path, 'r:*') as tf:            
             pass
        return True
    except tarfile.ReadError:       
        pass
    except Exception as e:       
        return False
    
    rar_magic_v1_v4 = b'\x52\x61\x72\x21\x1a\x07\x00'
    rar_magic_v5 = b'\x52\x61\x72\x21\x1a\x07\x01\x00'
    max_magic_len = max(len(rar_magic_v1_v4), len(rar_magic_v5))

    try:
        with open(file_path, 'rb') as f:       
            header = f.read(max_magic_len)
        
            if header.startswith(rar_magic_v1_v4) or header.startswith(rar_magic_v5):
                return True
    except Exception as e:      
        return False    
    return False

def extract_and_remove(file_path, extract_to_dir):   
    if not os.path.exists(file_path):
        print(f"Tệp nén không tồn tại: {file_path}")
        return   
    os.makedirs(extract_to_dir, exist_ok=True)
  
    if file_path.endswith('.zip'):
        print(f"Đang giải nén ZIP {file_path} vào {extract_to_dir}...")
        shutil.unpack_archive(file_path, extract_to_dir)
    elif file_path.endswith('.tar') or file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        print(f"Đang giải nén TAR {file_path} vào {extract_to_dir}...")
        import tarfile
        with tarfile.open(file_path) as tar:
            tar.extractall(path=extract_to_dir)
    else:
        print(f"Không hỗ trợ định dạng tệp nén: {file_path}")
        return

    print("Giải nén hoàn tất.")
   
    print(f"Đang xóa tệp nén {file_path}...")
    os.remove(file_path)
    print("Đã xóa tệp nén.")


def change_image_paths(input_file, output_file, old_path, new_path):
    """
    Thay đổi đường dẫn trong file
    """
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    image_path, text = parts
                    new_image_path = image_path.replace(old_path, new_path)
                    outfile.write(f"{new_image_path}\t{text}\n")
                else:  # Xử lý các dòng không có tab
                    continue
    except FileNotFoundError:
        print(f"File not found: {input_file}") 
   

def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', help='configuration file to use', default='download_config.yaml')
    args = parser.parse_args()
    
    config_path = args.config    

    
    # Load the YAML config
    cfg = load_yaml_config(config_path)
    
 
    if "root" in cfg:
        os.makedirs(cfg["root"], exist_ok=True)
        
    extract_to_dir = None
    if "extraction_paths" in cfg:
        os.makedirs(cfg["extraction_paths"], exist_ok=True)
        extract_to_dir = cfg["extraction_paths"]
    
    # Get dataset info and download files
    urls, filename_paths, check_validity = get_dataset_info(cfg)
    
    if len(urls) != len(filename_paths):
        print(f"Error: Number of URLs, filenames and extraction paths don't match")
        return    
   
    for i, (url, filename_path) in enumerate(zip(urls, filename_paths)):
        print(f"\n[{i+1}/{len(urls)}] Downloading {filename_path} from {url} ...")
        try:
            urlretrieve(url=url, filename=filename_path, check_validity=check_validity)
            print(f"Successfully downloaded {filename_path}") 
          
            is_archive = check_archive_file(filename_path)
            if is_archive:
                extract_and_remove(filename_path, extract_to_dir)
            
        except Exception as e:
            print(f"Error downloading {filename_path}: {e}") 
 

if __name__ == "__main__":
    main()