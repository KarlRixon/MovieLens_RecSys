#-*- coding:utf-8 -*-
import os,sys
if not os.path.exists('/home/aistudio/pypi'):
	os.mkdir('/home/aistudio/pypi')

# os.system('conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/')
# os.system('conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge')
# os.system('conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/')
# os.system('conda config --set show_channel_urls yes')

# os.system('pip download torch==1.1.0 torchvision==0.2.0 pillow==6.1 scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple -d /home/aistudio/pypi')
os.system('pip install torch torchvision pillow scikit-image --no-index -f /home/aistudio/pypi')