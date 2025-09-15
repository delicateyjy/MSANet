from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import io
import os
import cv2
import logging
import sys
from model.MSAnet import MSANet

# 初始化Flask应用
app = Flask(__name__, template_folder='templates')  # 显式指定模板目录
CORS(app)
print("[启动检查] 模板目录绝对路径:", os.path.abspath(app.template_folder))

@app.route('/')
def home():
    return render_template('index.html')  # 正确渲染模板

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CamoDetect')

# 配置类，仅用于模型加载
class Config(object):
    def __init__(self, snapshot=None, mode='test'):
        self.snapshot = snapshot
        self.mode = mode
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[ 56.77,  55.97,  57.50]]])

# 全局模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型权重路径验证
model_paths = [
    "MSANet.pth",  # 当前目录
    os.path.join("checkpoint", "MSANet.pth"),  # checkpoint目录
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        break

if not model_path:
    raise FileNotFoundError(f"未找到模型权重文件，已尝试路径: {model_paths}")

print(f"使用模型权重: {model_path}")

# 加载模型权重
try:
    # 使用与infer.py相同的方式初始化模型
    cfg = Config(snapshot=model_path)
    model = MSANet(cfg).to(device)
    model.eval()
    
    logger.info("=== 模型权重加载成功 ===")
    logger.info(f"模型设备: {next(model.parameters()).device}")

except Exception as e:
    logger.error(f"权重加载失败: {str(e)}")
    exit(1)


@app.route('/process', methods=['POST'])
def process():
    debug_data = {}
    try:
        # 验证文件上传
        if 'image' not in request.files:
            return jsonify({"error": "未上传文件"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "空文件名"}), 400

        # 记录处理开始
        logger.info(f"开始处理文件: {file.filename}")

        # 图像预处理
        img = Image.open(file.stream).convert('RGB')
        orig_w, orig_h = img.size
        debug_data['original_size'] = f"{orig_w}x{orig_h}"
        
        # 保存原图用于后续处理
        img_np = np.array(img)

        # 调整尺寸并归一化
        img_resized = cv2.resize(img_np, (384, 384))
        img_normalized = (img_resized - cfg.mean) / cfg.std
        
        debug_data['resized_shape'] = str(img_normalized.shape)

        # 转换为Tensor
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # 执行推理
        with torch.no_grad():
            # 修改为与infer.py一致的调用方式
            shape = img_np.shape[:2]
            P5, P4, P3, P2, P1 = model(img_tensor, shape)
            p1 = torch.sigmoid(P1)
            debug_data['output_range'] = f"{p1.min().item():.4f} ~ {p1.max().item():.4f}"

        # 后处理与可视化
        mask = (p1.squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        # 调整回原始尺寸
        mask_resized = cv2.resize(mask, (orig_w, orig_h))
        
        # 二值化处理
        _, mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
        
        # 直接返回二值掩码图
        result_img = Image.fromarray(mask_binary)
        byte_io = io.BytesIO()
        result_img.save(byte_io, 'PNG')
        byte_io.seek(0)

        logger.info(f"处理成功: {file.filename}")
        return send_file(byte_io, mimetype='image/png')

    except Exception as e:
        logger.error(f"处理异常: {str(e)}", exc_info=True)
        debug_data['error'] = str(e)
        return jsonify({"error": "处理失败", "debug": debug_data}), 500


@app.route('/process_mask', methods=['POST'])
def process_mask():
    """返回叠加效果图"""
    debug_data = {}
    try:
        # 验证文件上传
        if 'image' not in request.files:
            return jsonify({"error": "未上传文件"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "空文件名"}), 400

        # 记录处理开始
        logger.info(f"开始处理掩码: {file.filename}")

        # 图像预处理
        img = Image.open(file.stream).convert('RGB')
        orig_w, orig_h = img.size
        
        # 保存原图用于后续处理
        img_np = np.array(img)
        
        # 调整尺寸并归一化
        img_resized = cv2.resize(img_np, (384, 384))
        img_normalized = (img_resized - cfg.mean) / cfg.std

        # 转换为Tensor
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # 执行推理
        with torch.no_grad():
            shape = img_np.shape[:2]
            P5, P4, P3, P2, P1 = model(img_tensor, shape)
            p1 = torch.sigmoid(P1)

        # 后处理与可视化
        mask = (p1.squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        # 调整回原始尺寸
        mask_resized = cv2.resize(mask, (orig_w, orig_h))
        
        # 二值化处理
        _, mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
        
        # 创建彩色掩码用于可视化
        mask_colored = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)
        mask_colored[..., 0] = 0  # 将B通道置为0
        mask_colored[..., 2] = 0  # 将R通道置为0
        # 只保留G通道，生成绿色掩码
        
        # 叠加到原图
        alpha = 0.5
        overlay = img_np.copy()
        overlay_mask = np.where(mask_colored > 0, True, False)
        overlay[overlay_mask] = cv2.addWeighted(img_np, 1-alpha, mask_colored, alpha, 0)[overlay_mask]
        
        # 生成响应图像
        result_img = Image.fromarray(overlay)
        byte_io = io.BytesIO()
        result_img.save(byte_io, 'PNG')
        byte_io.seek(0)

        logger.info(f"叠加效果处理成功: {file.filename}")
        return send_file(byte_io, mimetype='image/png')

    except Exception as e:
        logger.error(f"叠加效果处理异常: {str(e)}", exc_info=True)
        debug_data['error'] = str(e)
        return jsonify({"error": "叠加效果处理失败", "debug": debug_data}), 500


@app.route('/exit', methods=['POST'])
def exit_app():
    """关闭服务器"""
    try:
        # 记录关闭请求
        logger.info("接收到关闭请求，服务器将在5秒后关闭...")
        
        # 返回成功消息
        response = jsonify({"status": "success", "message": "服务器将在5秒后关闭"})
        
        # 启动一个线程在返回响应后关闭服务器
        from threading import Timer
        def shutdown():
            sys.exit(0)
        
        Timer(5, shutdown).start()
        
        return response
    except Exception as e:
        logger.error(f"关闭请求处理异常: {str(e)}")
        return jsonify({"error": "关闭请求失败", "message": str(e)}), 500


if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('static', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # 启动应用
    print("伪装目标检测系统已启动，访问 http://localhost:5000 开始使用")
    app.run(host='0.0.0.0', port=5000, debug=False)