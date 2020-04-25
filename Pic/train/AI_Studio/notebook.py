# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
!ls /home/aistudio/data


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. All changes under this directory will be kept even after reset. Please clean unnecessary files in time to speed up environment loading.
!ls /home/aistudio/work


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, you need to use the persistence path as the following:
!conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
!conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
!conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
!conda config --set show_channel_urls yes
!mkdir /home/aistudio/external-libraries
!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.1.0 torchvision==0.2.1 pillow==6.1 scikit-image -t /home/aistudio/external-libraries
# !pip install beautifulsoup4 -t /home/aistudio/external-libraries


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
# Also add the following code, so that every time the environment (kernel) starts, just run the following code:
import sys
sys.path.append('/home/aistudio/external-libraries')


# 解压数据集
import os
import zipfile
os.chdir('/home/aistudio/data/posters')
extracting = zipfile.ZipFile('posters.zip')
extracting.extractall()


!nvidia-smi
#Thu Apr 23 23:17:06 2020       
#+-----------------------------------------------------------------------------+
#| NVIDIA-SMI 396.82                 Driver Version: 396.82                    |
#|-------------------------------+----------------------+----------------------+
#| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
#| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
#|===============================+======================+======================|
#|   0  Tesla V100-SXM2...  On   | 00000000:00:08.0 Off |                    0 |
#| N/A   35C    P0    51W / 300W |   7886MiB / 16160MiB |      0%      Default |
#+-------------------------------+----------------------+----------------------+
#                                                                               
#+-----------------------------------------------------------------------------+
#| Processes:                                                       GPU Memory |
#|  GPU       PID   Type   Process name                             Usage      |
#|=============================================================================|
#+-----------------------------------------------------------------------------+