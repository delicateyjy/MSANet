import os
import sys
import cv2
import torch
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import time
from model.MSAnet import MSANet
from torchvision import transforms

# 配置类，仅用于模型加载
class Config(object):
    def __init__(self, snapshot=None, mode='test'):
        self.snapshot = snapshot
        self.mode = mode
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[ 56.77,  55.97,  57.50]]])

class Inference:
    def __init__(self, ckpt_path, max_retries=3):
        # 初始化配置
        self.cfg = Config(snapshot=ckpt_path)
        # 提取模型名称
        self.model_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        
        # 检查是否可用CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = None
        
        # 添加重试逻辑
        retry_count = 0
        while retry_count < max_retries:
            try:
                # 初始化模型
                self.model = MSANet(self.cfg)
                
                # 载入模型参数
                if os.path.exists(ckpt_path):
                    print(f"正在加载模型 {ckpt_path}...")
                    try:
                        checkpoint = torch.load(ckpt_path, map_location=self.device)
                        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            # 如果检查点包含state_dict键
                            self.model.load_state_dict(checkpoint['state_dict'])
                        else:
                            # 直接加载模型参数
                            self.model.load_state_dict(checkpoint)
                        print(f"模型已从 {ckpt_path} 加载成功")
                        break  # 成功加载，跳出循环
                    except Exception as e:
                        print(f"模型加载失败: {str(e)}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise ValueError(f"无法加载模型，已尝试 {max_retries} 次")
                        print(f"尝试重新加载... ({retry_count}/{max_retries})")
                        time.sleep(1)  # 等待一秒后重试
                else:
                    raise FileNotFoundError(f"模型文件 {ckpt_path} 不存在")
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise ValueError(f"模型初始化失败: {str(e)}")
                print(f"模型初始化失败: {str(e)}，尝试重新初始化... ({retry_count}/{max_retries})")
                time.sleep(1)  # 等待一秒后重试
                
        # 将模型转移到设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        print(f"模型名称: {self.model_name}")
        print(f"使用设备: {self.device}")

    def preprocess_image(self, img_path):
        # 读取图像
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")
        except Exception as e:
            raise ValueError(f"图像读取失败: {str(e)}")
        
        # 保存原始尺寸用于后处理
        shape = img.shape[:2]  # (H, W)
        
        try:
            # 归一化处理
            img = (img - self.cfg.mean) / self.cfg.std
            
            # 调整尺寸为模型输入大小 (384x384)
            img_resized = cv2.resize(img, (384, 384))
            
            # 转换为张量
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            return img_tensor, shape
        except Exception as e:
            raise ValueError(f"图像预处理失败: {str(e)}")

    def predict(self, img_path):
        # 预处理图像
        img_tensor, orig_shape = self.preprocess_image(img_path)
        
        # 推理
        try:
            with torch.no_grad():
                print("执行模型推理...")
                P5, P4, P3, P2, P1 = self.model(img_tensor, orig_shape)
                pred = torch.sigmoid(P1[0, 0]).cpu().numpy() * 255
                print("推理完成")
                return pred, orig_shape
        except Exception as e:
            raise ValueError(f"模型推理失败: {str(e)}")

    def find_gt_image(self, img_path):
        """
        尝试找到与输入图像对应的GT图像
        搜索上一级目录中的GT文件夹
        """
        # 获取输入图像的目录和文件名
        img_dir = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_name)[0]
        
        # 尝试在上一级目录的GT文件夹中寻找GT图像
        parent_dir = os.path.dirname(img_dir)
        gt_dir = os.path.join(parent_dir, "GT")
        
        # 可能的GT图像路径
        gt_path = os.path.join(gt_dir, f"{img_name_no_ext}.png")
        
        if os.path.exists(gt_path):
            return gt_path
        
        # 如果找不到.png格式，尝试.jpg格式
        gt_path_jpg = os.path.join(gt_dir, f"{img_name_no_ext}.jpg")
        if os.path.exists(gt_path_jpg):
            return gt_path_jpg
        
        return None

    def visualize(self, img_path, pred, save_path=None):
        try:
            # 读取原始图像
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 将预测结果调整为原始图像大小
            pred_resized = cv2.resize(pred, (img.shape[1], img.shape[0]))
            
            # 创建可视化
            plt.figure(figsize=(15, 5))
            
            # 第一张图：原图
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.axis('off')
            
            # 第二张图：预测的二值图
            plt.subplot(1, 3, 2)
            plt.imshow(pred_resized, cmap='gray')
            plt.axis('off')
            
            # 第三张图：原图叠加预测结果
            plt.subplot(1, 3, 3)
            # 将预测结果转换为二值图（0或255）
            binary_mask = (pred_resized > 127).astype(np.uint8) * 255
            # 创建绿色掩码
            green_mask = np.zeros_like(img)
            green_mask[:, :, 1] = binary_mask  # 绿色通道
            # 叠加原图和掩码
            overlay = cv2.addWeighted(img, 0.7, green_mask, 0.3, 0)
            plt.imshow(overlay)
            plt.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                # 确保保存目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                print(f"可视化结果已保存至: {save_path}")
                
            plt.show()
        except Exception as e:
            print(f"可视化过程中出错: {str(e)}")

    def save_mask(self, pred, save_path):
        try:
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存预测的掩码图像
            cv2.imwrite(save_path, np.round(pred))
            print(f"掩码已保存至: {save_path}")
            return True
        except Exception as e:
            print(f"掩码保存失败: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description='伪装物体检测')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--ckpt', type=str, default='MSANet.pth', help='模型检查点路径')
    parser.add_argument('--save_vis', type=str, default=None, help='可视化结果保存路径')
    parser.add_argument('--save_mask', type=str, default=None, help='掩码结果保存路径 (默认: output/Prediction/{model_name}/)')
    parser.add_argument('--no_vis', action='store_true', help='不显示可视化结果')
    args = parser.parse_args()

    try:
        # 检查输入图像是否存在
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"输入图像不存在: {args.image}")
            
        # 检查模型文件是否存在
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"模型文件不存在: {args.ckpt}")
            
        # 创建推理器
        print(f"初始化推理器...")
        inferencer = Inference(args.ckpt)
        
        # 根据模型名称创建输出目录
        model_name = inferencer.model_name
        DEFAULT_OUTPUT_DIR = os.path.join('output', 'Prediction', model_name)
        
        # 进行预测
        print(f"开始处理图像: {args.image}")
        pred, orig_shape = inferencer.predict(args.image)
        
        # 可视化结果
        if not args.no_vis:
            inferencer.visualize(args.image, pred, args.save_vis)
        
        # 保存掩码结果
        if args.save_mask:
            mask_save_path = args.save_mask
        else:
            # 确保默认输出目录存在
            os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
            # 从原图路径提取文件名
            img_name = os.path.basename(args.image)
            img_name_no_ext = os.path.splitext(img_name)[0]
            # 构建默认输出路径
            mask_save_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{img_name_no_ext}_pred.png")
        
        # 保存掩码
        inferencer.save_mask(pred, mask_save_path)
        
        print(f"预测结果已保存到文件夹: {DEFAULT_OUTPUT_DIR}")
        print(f"处理完成!")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
