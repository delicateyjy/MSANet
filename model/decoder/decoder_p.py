# 导入所需的包
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

# 自定义权重初始化函数（适用于CNN和BN层）
def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()
            
# 张量形状变换工具函数
def to_3d(x):
    """将4D特征图转换为3D序列 (B,C,H,W) -> (B,H*W,C)"""
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    """将3D序列恢复为4D特征图 (B,H*W,C) -> (B,C,H,W)"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# 自定义LayerNorm实现
class BiasFree_LayerNorm(nn.Module):
    """无偏置的层归一化（适用于通道维度）"""
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
            
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    """带偏置的层归一化（兼容标准LayerNorm）"""
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
    def initialize(self):
        weight_init(self)

class LayerNorm(nn.Module):
    """空间维度保持的LayerNorm（输入输出保持4D）"""
    def __init__(self, dim, norm_type='WithBias'):
        super().__init__()
        self.body = WithBias_LayerNorm(dim) if norm_type == 'WithBias' else BiasFree_LayerNorm(dim)
    
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
    def initialize(self):
        weight_init(self)

# Transformer前馈网络
class FeedForward(nn.Module):
    """扩展比为ffn_expansion_factor的前馈网络"""
    def __init__(self, dim, ffn_expansion_factor=4, bias=False):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim*2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_dim*2, hidden_dim*2, 3, padding=1, groups=hidden_dim*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=bias)
    
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 分割特征进行门控
        x = F.gelu(x1) * x2  # Gated机制
        return self.project_out(x)

    def initialize(self):
        weight_init(self)

# 多头自注意力模块
class Attention(nn.Module):
    """基于卷积的多头自注意力机制"""
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 可学习温度系数
        
        # QKV生成分支
        self.qkv_0 = nn.Conv2d(dim, dim, 1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, 1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, 1, bias=bias)
        
        # 深度卷积增强局部特征
        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)

        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        # 生成QKV并应用深度卷积
        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))

        # 应用注意力掩码
        if mask is not None:
            q = q * mask
            k = k * mask
        
        # 重组为多头形式
        q = rearrange(q, 'b (h d) x y -> b h d (x y)', h=self.num_heads)
        k = rearrange(k, 'b (h d) x y -> b h d (x y)', h=self.num_heads)
        v = rearrange(v, 'b (h d) x y -> b h d (x y)', h=self.num_heads)
        
        # 归一化后计算注意力
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        # 特征聚合与重组
        out = rearrange(attn @ v, 'b h d (x y) -> b (h d) x y', h=self.num_heads, x=H, y=W)
        return self.project_out(out)
    
    def initialize(self):
        weight_init(self)

# 多尺度注意力头
class MSA_head(nn.Module):
    """多尺度注意力头（包含层归一化和残差连接）"""
    def __init__(self, dim=128, num_heads=8, ffn_expansion=4, norm_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm(dim, norm_type)
        self.attn = Attention(dim, num_heads)
        self.norm2 = LayerNorm(dim, norm_type)
        self.ffn = FeedForward(dim, ffn_expansion)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)  # 带掩码的注意力残差连接
        x = x + self.ffn(self.norm2(x))         # 前馈残差连接
        return x
    
    def initialize(self):
        weight_init(self)

# 多尺度注意力融合模块
class MSA_module(nn.Module):
    """多尺度注意力融合模块"""
    def __init__(self, dim=128):
        super().__init__()
        # 三个注意力分支
        self.B_TA = MSA_head(dim)   # 背景注意力
        self.F_TA = MSA_head(dim)   # 前景注意力
        self.TA = MSA_head(dim)  # 全局注意力
        
        # 特征融合层
        self.Fuse = nn.Conv2d(3*dim,dim,kernel_size=3,padding=1)
        self.Fuse2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, side_x, mask):
        """参数说明：
        x: 主特征图
        side_x: 侧支特征
        mask: 来自上一层的注意力掩码
        """
        N, C, H, W = x.shape
        # 生成动态掩码
        mask = F.interpolate(mask, x.shape[2:], mode='bilinear')
        mask_d = torch.sigmoid(mask.detach())  # 梯度截断
        
        # 多分支处理
        fg_feat = self.F_TA(x, mask_d)       # 前景增强
        bg_feat = self.B_TA(x, 1 - mask_d)   # 背景抑制
        global_feat = self.TA(x)     # 全局上下文
        
        # 特征融合
        x = torch.cat([fg_feat, bg_feat, global_feat], dim=1)
        x = x.view(N, 3*C, H, W)
        x = self.Fuse(x)
        
        # 与侧支特征交互
        return self.Fuse2(side_x + side_x * x)  # 门控融合
    
    def initialize(self):
        weight_init(self)

# 特征融合卷积块
class Conv_Block(nn.Module):
    """三输入融合卷积块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(3*channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, 2*channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels*2)

        self.conv3 = nn.Conv2d(2*channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
    
    def forward(self, input1, input2, input3):
        fuse = torch.cat((input1, input2, input3), 1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse
    
    def initialize(self):
        weight_init(self)

# 主解码器架构
class Decoder(nn.Module):
    """多尺度渐进式解码器"""
    def __init__(self, channels):
        super(Decoder, self).__init__()

        # 编码器特征适配层（假设输入通道为PVTv2-b4的输出）
        # 对应PVTv4的四个阶段输出通道
        self.side_conv1 = nn.Conv2d(512, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2 = nn.Conv2d(320, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv3 = nn.Conv2d(128, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4 = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)

        # 特征融合模块
        self.conv_block = Conv_Block(channels)
        
        self.fuse1 = nn.Sequential(nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(channels))
        self.fuse2 = nn.Sequential(nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(channels))
        self.fuse3 = nn.Sequential(nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(channels))
       
        # 多尺度注意力模块
        self.MSA5=MSA_module(dim = channels)
        self.MSA4=MSA_module(dim = channels)
        self.MSA3=MSA_module(dim = channels)
        self.MSA2=MSA_module(dim = channels)

        # 预测头
        self.predtrans1  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans2  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans3  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans4  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans5  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

        self.initialize()
    
    def forward(self, E4, E3, E2, E1,shape):
        """处理流程说明：
        E4: 编码器第4阶段特征（最深/最抽象）
        E3: 第3阶段特征
        E2: 第2阶段特征 
        E1: 第1阶段特征（最浅层）
        """
        # 特征通道适配
        E4, E3, E2, E1= self.side_conv1(E4), self.side_conv2(E3), self.side_conv3(E2), self.side_conv4(E1)
        
        # 特征对齐（统一到E3的尺寸）
        if E4.size()[2:] != E3.size()[2:]:
            E4 = F.interpolate(E4, size=E3.size()[2:], mode='bilinear')
        if E2.size()[2:] != E3.size()[2:]:
            E2 = F.interpolate(E2, size=E3.size()[2:], mode='bilinear')

        # 三特征融合生成E5
        E5 = self.conv_block(E4, E3, E2)

        # 逐级融合与上采样
        E4 = torch.cat((E4, E5),1)
        E3 = torch.cat((E3, E5),1)
        E2 = torch.cat((E2, E5),1)

        E4 = F.relu(self.fuse1(E4), inplace=True)
        E3 = F.relu(self.fuse2(E3), inplace=True)
        E2 = F.relu(self.fuse3(E2), inplace=True)

        # 生成多尺度预测图
        P5 = self.predtrans5(E5)

        D4 = self.MSA5(E5, E4, P5)
        D4 = F.interpolate(D4, size=E3.size()[2:], mode='bilinear')
        P4  = self.predtrans4(D4)
        
        D3 = self.MSA4(D4, E3, P4)
        D3 = F.interpolate(D3,   size=E2.size()[2:], mode='bilinear')
        P3  = self.predtrans3(D3)  
        
        D2 = self.MSA3(D3, E2, P3)
        D2 = F.interpolate(D2, size=E1.size()[2:], mode='bilinear')
        P2  = self.predtrans2(D2)
        
        D1 = self.MSA2(D2, E1, P2)
        P1  =self.predtrans1(D1)

        P1 = F.interpolate(P1, size=shape, mode='bilinear')
        P2 = F.interpolate(P2, size=shape, mode='bilinear')
        P3 = F.interpolate(P3, size=shape, mode='bilinear')
        P4 = F.interpolate(P4, size=shape, mode='bilinear')
        P5 = F.interpolate(P5, size=shape, mode='bilinear')
        
        return P5, P4, P3, P2, P1 # 返回P5-P1五个尺度的预测
    
    def initialize(self):
        weight_init(self)