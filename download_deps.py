#!/usr/bin/env python3

# PEP 723 metadata
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface-hub",
#   "nltk",
# ]
# ///

from huggingface_hub import snapshot_download
import nltk
import os
import urllib.request
import shutil
import sys

urls = [
    "http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb",
    "http://ports.ubuntu.com/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_arm64.deb",
    "https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/3.0.0/tika-server-standard-3.0.0.jar",
    "https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/3.0.0/tika-server-standard-3.0.0.jar.md5",
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    "https://bit.ly/chrome-linux64-121-0-6167-85",
    "https://bit.ly/chromedriver-linux64-121-0-6167-85",
]

repos = [
    "InfiniFlow/text_concat_xgb_v1.0",
    "InfiniFlow/deepdoc",
    "InfiniFlow/huqie",
    "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-reranker-v2-m3",
    "maidalun1020/bce-embedding-base_v1",
    "maidalun1020/bce-reranker-base_v1",
]

def download_model(repo_id):
    local_dir = os.path.abspath(os.path.join("huggingface.co", repo_id))
    os.makedirs(local_dir, exist_ok=True)
    # 使用镜像
    mirror = "https://hf-mirror.com"
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False, endpoint=mirror)

def download_nltk_resources():
    # 设置多个可能的下载目录，确保系统能找到
    nltk_dirs = [
        os.path.abspath('nltk_data'),
        os.path.join(os.path.expanduser('~'), 'nltk_data'),
        '/usr/share/nltk_data',
        '/usr/local/share/nltk_data'
    ]
    
    # 创建所有目录
    for directory in nltk_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # 设置 NLTK_DATA 环境变量
    os.environ['NLTK_DATA'] = nltk_dirs[0]
    
    # 下载标准资源
    resources = ['wordnet', 'punkt', 'omw-1.4']
    
    for resource in resources:
        print(f"下载 NLTK 资源: {resource}...")
        try:
            nltk.download(resource, download_dir=nltk_dirs[0], quiet=False)
        except Exception as e:
            print(f"下载 {resource} 失败: {str(e)}")
            sys.exit(1)
    
    # 确保 punkt_tab 路径正确
    punkt_dir = os.path.join(nltk_dirs[0], 'tokenizers', 'punkt')
    punkt_tab_dir = os.path.join(nltk_dirs[0], 'tokenizers', 'punkt_tab', 'english')
    
    # 创建 punkt_tab 目录结构
    os.makedirs(punkt_tab_dir, exist_ok=True)
    
    # 复制 punkt 文件到 punkt_tab 目录
    if os.path.exists(punkt_dir):
        punkt_files = os.listdir(punkt_dir)
        for file in punkt_files:
            if file.endswith('.pickle'):
                src = os.path.join(punkt_dir, file)
                dst = os.path.join(punkt_tab_dir, file)
                shutil.copy2(src, dst)
                print(f"复制 {src} 到 {dst}")
                
    # 添加确认步骤，验证 punkt_tab 资源是否可用
    try:
        from nltk.data import find
        find('tokenizers/punkt_tab/english/')
        print("punkt_tab 资源已成功配置！")
    except LookupError:
        print("警告: punkt_tab 资源仍然无法找到，尝试手动创建...")
        # 尝试直接创建英语模型文件
        english_pickle = os.path.join(punkt_dir, 'english.pickle')
        if os.path.exists(english_pickle):
            dst = os.path.join(punkt_tab_dir, 'english.pickle')
            shutil.copy2(english_pickle, dst)
            print(f"已复制 {english_pickle} 到 {dst}")
        
    print("NLTK 资源下载和配置完成！")

if __name__ == "__main__":
    for url in urls:
        filename = url.split("/")[-1]
        print(f"下载 {url}...")
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)

    # 使用新的 NLTK 资源下载函数
    download_nltk_resources()

    for repo_id in repos:
        print(f"下载 huggingface 仓库 {repo_id}...")
        download_model(repo_id)
