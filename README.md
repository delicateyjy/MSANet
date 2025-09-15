# 环境准备

conda create --name MSANet python=3.9

conda activate MSANet

conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch

conda install numpy opencv matplotlib scipy=1.7.3 tensorboard

pip install tensorboardX>=2.5 timm==0.5.4 einops

pip install Flask flask-cors

# 下载模型数据

在[百度网盘](https://pan.baidu.com/s/1PYfDi7koL8k6wdvB85K-zw?pwd=53ar)下载后放在 **/checkpoint** 文件夹内

# 运行代码

在命令行输入 `python app.py`，打开生成的网址，例如：`Running on http://127.0.0.1:5000`

在打开的网页选择一张图片（可使用images文件夹里面的图片），运行等待后即可得到结果。
