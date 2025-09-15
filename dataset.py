# 导入所需包
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from PIL import Image

########################### Data Augmentation ###########################
class Normalize:
    """标准化处理(兼容Windows内存布局)"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None, edge=None, body=None, detail=None, tail=None):
        # 使用原地操作减少内存占用
        image = (image - self.mean) / self.std
        if mask is None:
            return image
        # 返回三个值：图像、掩码和边缘
        if edge is not None:
            return image, mask/255, edge/255
        return image, mask/255

class RandomCrop(object):
    """Windows下整数除法修正"""
    def __call__(self, image, mask=None, edge=None, body=None, detail=None):
        H,W,_   = image.shape # 获取实际高度宽度
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw

        if mask is None:
            return image[p0:p1, p2:p3, :]
        if edge is not None:
            return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], edge[p0:p1, p2:p3]
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]

class RandomFlip(object):
    """水平翻转（兼容OpenCV Windows实现）"""
    def __call__(self, image, mask=None, edge=None, body=None, detail=None):
        if np.random.randint(2) == 0:
            # 使用copy()避免负步长问题
            if mask is None:
                return image[:, ::-1, :].copy()
            if edge is not None:
                return image[:, ::-1, :].copy(), mask[:, ::-1].copy(), edge[:, ::-1].copy()
            return image[:, ::-1, :].copy(), mask[:, ::-1].copy()
        else:
            if mask is None:
                return image
            if edge is not None:
                return image, mask, edge
            return image, mask

class Resize(object):
    """调整尺寸（保持宽高比一致）"""
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, edge=None, body=None, detail=None):
        # 使用线性插值调整图像大小
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        # 调整掩码和边缘图大小
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if edge is not None:
            edge = cv2.resize(edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image, mask, edge
        return image, mask

class RandomRotate(object):
    """随机旋转（修复角度裁剪逻辑）"""
    def rotate(self, img, random_angle, mode='image'):
        if mode == 'image':
            H, W, _ = img.shape
        else:
            H, W = img.shape

        random_angle %= 360
        image_change = cv2.getRotationMatrix2D((W/2, H/2), random_angle, 1)
        image_rotated = cv2.warpAffine(img, image_change, (W, H))
    
        angle_crop = random_angle % 180
        if random_angle > 90:
            angle_crop = 180 - angle_crop
        theta = angle_crop * np.pi / 180
        hw_ratio = float(H) / float(W)
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)
        r = hw_ratio if H > W else 1 / hw_ratio
        denominator = r * tan_theta + 1
        crop_mult = numerator / denominator

        w_crop = int(crop_mult * W)
        h_crop = int(crop_mult * H)
        x0 = int((W - w_crop) / 2)
        y0 = int((H - h_crop) / 2)
        crop_image = lambda x, x0, y0, W, H: x[y0:y0+h_crop, x0:x0+w_crop]
        output = crop_image(image_rotated, x0, y0, w_crop, h_crop)

        return output

    def __call__(self, image, mask=None, edge=None, body=None, detail=None):
        do_seed = np.random.randint(0, 3)
        if do_seed != 2:
            if mask is None:
                return image
            if edge is not None:
                return image, mask, edge
            return image, mask
        
        # 30%的概率旋转
        random_angle = np.random.randint(-10, 10)
        image = self.rotate(image, random_angle, 'image')

        if mask is None:
            return image
        mask = self.rotate(mask, random_angle, 'mask')
        
        if edge is not None:
            edge = self.rotate(edge, random_angle, 'mask')
            return image, mask, edge
        return image, mask

class ColorEnhance(object):
    """颜色增强（每次调用生成随机参数）"""
    def __init__(self):

        #A:0.5~1.5, G: 5-15
        self.A = np.random.randint(7, 13, 1)[0]/10
        self.G = np.random.randint(7, 13, 1)[0]

    def __call__(self, image, mask=None, edge=None, body=None, detail=None):

        do_seed = np.random.randint(0,3)
        if do_seed > 1:#1: # 1/3
            H, W, _   = image.shape
            dark_matrix = np.zeros([H, W, _], image.dtype)
            image = cv2.addWeighted(image, self.A, dark_matrix, 1-self.A, self.G) 
        else:
            pass
            
        if mask is None:
            return image
        if edge is not None:
            return image, mask, edge
        return image, mask

class GaussNoise(object):
    """高斯噪声"""
    def __init__(self):
        self.Mean = 0
        self.Var = 0.001

    def __call__(self, image, mask=None, edge=None, body=None, detail=None):
        H, W, _ = image.shape
        do_seed = np.random.randint(0, 3)


        if do_seed == 0: #1: # 1/3
            factor = np.random.randint(0,10)
            noise = np.random.normal(self.Mean, self.Var ** 0.5, image.shape) * factor
            noise = noise.astype(image.dtype)
            image = cv2.add(image, noise)
        else:
            pass

        if mask is None:
            return image
        if edge is not None:
            return image, mask, edge
        return image, mask

class ToTensor(object):
    """转换为Tensor"""
    def __call__(self, image, mask=None, edge=None, body=None, detail=None):
        # 显式转换为float32类型
        image = torch.from_numpy(image).float()  # 确保是float32类型
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask = torch.from_numpy(mask).float()  # 确保是float32类型
        if edge is not None:
            edge = torch.from_numpy(edge).float()  # 确保是float32类型
            return image, mask, edge
        return image, mask 

########################### Config File ###########################
class Config(object):
    """配置类（简化版本）"""
    def __init__(self, **kwargs):
        # 使用标准ImageNet均值方差
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[ 56.77,  55.97,  57.50]]])
        
        # 设置默认数据集划分比例
        self.train_ratio = 0.8  # 默认80%用于训练
        self.seed = 42         # 随机种子，保证可复现性

        # 将所有kwargs直接设置为属性
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # 参数验证
        print('\n配置参数:')
        for k, v in kwargs.items():
            print(f'{k:15}: {v}')

########################### Dataset Class ###########################
def safe_imread(path, flag=cv2.IMREAD_COLOR):
    """安全读取图像（解决中文路径问题）"""
    try:
        stream = open(path, 'rb').read()
        bytes = np.frombuffer(stream, dtype=np.uint8)
        return cv2.imdecode(bytes, flag)
    except Exception as e:
        print(f"读取失败: {path} - {str(e)}")
        return None

class CamoDataset(Dataset):
    """伪装检测数据集（支持动态划分训练/测试集）"""
    def __init__(self, cfg, model_name, split='train'):
        self.cfg = cfg
        self.model_name = model_name
        self.split = split  # 'train'或'test'
        
        # 检查数据路径
        self.data_root = cfg.datapath
        self.image_dir = os.path.join(self.data_root, 'Image')
        self.gt_dir = os.path.join(self.data_root, 'GT')
        self.edge_dir = os.path.join(self.data_root, 'Edge')
        
        # 获取所有图像文件名（不含扩展名）
        all_image_files = os.listdir(self.image_dir)
        self.all_samples = [os.path.splitext(f)[0] for f in all_image_files 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 设置随机种子以确保可重现性
        random.seed(cfg.seed)
        # 随机打乱样本顺序
        random.shuffle(self.all_samples)
        
        # 按比例划分训练集和测试集
        split_idx = int(len(self.all_samples) * cfg.train_ratio)
        
        # 根据split参数选择训练集或测试集样本
        if split == 'train':
            self.samples = self.all_samples[:split_idx]
            print(f"加载训练集: {len(self.samples)} 个样本 ({cfg.train_ratio*100:.1f}%)")
        else:
            self.samples = self.all_samples[split_idx:]
            print(f"加载测试集: {len(self.samples)} 个样本 ({(1-cfg.train_ratio)*100:.1f}%)")
        
        # 初始化变换
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(384, 384)
        self.randomrotate = RandomRotate()
        self.colorenhance = ColorEnhance()
        self.gaussnoise = GaussNoise()
        self.totensor   = ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        
        # 读取图像文件
        try:
            image_path = os.path.join(self.image_dir, name + '.jpg')
            # 如果找不到jpg格式，尝试其他格式
            if not os.path.exists(image_path):
                for ext in ['.jpeg', '.png']:
                    test_path = os.path.join(self.image_dir, name + ext)
                    if os.path.exists(test_path):
                        image_path = test_path
                        break
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
        except Exception as e:
            print(f"加载图像失败 ({name}): {e}")
            # 如果发生错误，尝试返回另一个样本
            return self.__getitem__((idx + 1) % len(self.samples))
        
        # 始终读取mask和边缘图
        try:
            mask_path = os.path.join(self.gt_dir, name + '.png')
            mask = cv2.imread(mask_path, 0).astype(np.float32)
            
            # 读取边缘图
            edge_path = os.path.join(self.edge_dir, name + '.png')
            edge = cv2.imread(edge_path, 0).astype(np.float32)
        except Exception as e:
            print(f"加载掩码/边缘图失败 ({name}): {e}")
            # 如果无边缘图，可以使用掩码生成简单边缘
            if 'edge' not in locals() and 'mask' in locals():
                edge = cv2.Canny(mask.astype(np.uint8), 10, 100)
            elif 'mask' not in locals():
                # 两者都加载失败则跳过此样本
                return self.__getitem__((idx + 1) % len(self.samples))
        
        # 数据增强处理
        image, mask, edge = self.normalize(image, mask, edge)
        
        # 训练时应用更多数据增强
        if self.split == 'train':
            image, mask, edge = self.randomcrop(image, mask, edge)
            image, mask, edge = self.randomflip(image, mask, edge)
            # 可选：添加更多增强
            # image, mask, edge = self.randomrotate(image, mask, edge)
        else:
            # 测试时只调整大小，保持一致性
            shape = image.shape[:2]  # 记录原始尺寸用于还原
            image, mask, edge = self.resize(image, mask, edge)
        
        # 转换为张量
        image, mask, edge = self.totensor(image, mask, edge)
        
        # 测试时返回原始大小信息用于还原
        if self.split != 'train':
            return image, mask, edge, shape, name
        
        return image, mask, edge

    def collate_fn(self, batch):
        """处理无效数据样本"""
        size = 384
        
        # 导入函数以处理张量大小调整
        import torch.nn.functional as F
        
        if self.split == 'train':
            # 训练模式：返回image, mask, edge
            image, mask, edge = [list(item) for item in zip(*batch)]
        else:
            # 测试模式：返回image, mask, edge, shape, name
            image, mask, edge, shape, name = [list(item) for item in zip(*batch)]
        
        # 确保所有样本大小一致
        for i in range(len(batch)):
            # 处理图像
            if isinstance(image[i], np.ndarray):  # 如果还不是tensor
                image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
                mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
                edge[i]  = cv2.resize(edge[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
                
                # 转换为张量
                image[i] = torch.from_numpy(image[i]).float().permute(2, 0, 1)
                mask[i]  = torch.from_numpy(mask[i]).float().unsqueeze(0) 
                edge[i]  = torch.from_numpy(edge[i]).float().unsqueeze(0)
            else:  # 已经是tensor但可能尺寸不一致
                # 调整图像大小
                if image[i].shape[1] != size or image[i].shape[2] != size:
                    image[i] = F.interpolate(image[i].unsqueeze(0), size=(size, size), 
                                          mode='bilinear', align_corners=False).squeeze(0)
                
                # 调整掩码大小
                if len(mask[i].shape) == 2:  # 确保mask是3D张量 [1,H,W]
                    mask[i] = mask[i].unsqueeze(0)
                if mask[i].shape[1] != size or mask[i].shape[2] != size:
                    mask[i] = F.interpolate(mask[i].unsqueeze(0), size=(size, size), 
                                          mode='nearest').squeeze(0)
                
                # 调整边缘图大小
                if len(edge[i].shape) == 2:  # 确保edge是3D张量 [1,H,W]
                    edge[i] = edge[i].unsqueeze(0)
                if edge[i].shape[1] != size or edge[i].shape[2] != size:
                    edge[i] = F.interpolate(edge[i].unsqueeze(0), size=(size, size), 
                                          mode='nearest').squeeze(0)

        # 整批次堆叠
        image = torch.stack(image, dim=0)
        mask  = torch.stack(mask, dim=0)
        edge  = torch.stack(edge, dim=0)
        
        if self.split != 'train':
            return image, mask, edge, shape, name
        return image, mask, edge
