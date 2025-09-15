# 导入所需包
import torch
from torch import nn
from torch.utils import model_zoo
from .encoder.pvtv2_encoder import pvt_v2_b4      # 实际使用的PVTv2编码器
from .decoder.decoder_p import Decoder            # 自定义解码器
from timm.models import create_model

def weight_init_backbone(module):
    """编码器专用权重初始化（与解码器区分）"""
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

def weight_init(module):
    """全模型通用权重初始化"""
    # 初始化逻辑与weight_init_backbone类似，但应用于所有模块
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

class MSANet(torch.nn.Module):  # 使用torch.nn.Module保持一致
    """伪装检测模型主类（集成编码器-解码器）"""
    def __init__(self, cfg, load_path=None):  # 移除默认参数None，与旧版一致
        super(MSANet, self).__init__()  # 与旧版保持一致的初始化方式
        self.cfg = cfg
        
        # 初始化PVTv2编码器
        self.encoder = pvt_v2_b4()  # 输入尺寸需为32的倍数
        
        # 加载预训练编码器权重
        if load_path is not None:
            pretrained_dict = torch.load(load_path)  
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(pretrained_dict)
            print('Pretrained encoder loaded.')
        
        # 初始化自定义解码器
        self.decoder = Decoder(128)
        
        # 模型初始化
        self.initialize()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)


    def forward(self, x, shape=None):
        """
        前向传播流程：
        Args:
            x: 输入图像 (B,3,H,W)
            shape: 目标输出尺寸，默认同输入尺寸
        Returns:
            多尺度预测图元组：(P5, P4, P3, P2, P1)
        """
        # 编码器提取特征
        features = self.encoder(x)  # 返回 [stage4, stage3, stage2, stage1]
        
        # 解包特征（根据PVTv2的输出顺序）
        x1 = features[0]  # stage4: (B,512,24,24)
        x2 = features[1]  # stage3: (B,320,24,24)
        x3 = features[2]  # stage2: (B,128,48,48)
        x4 = features[3]  # stage1: (B,64,96,96)
        
        # 确定输出尺寸
        if shape is None:
            shape = x.size()[2:]
            
        # 解码器生成多尺度预测
        P5, P4, P3, P2, P1 = self.decoder(x1, x2, x3, x4, shape)
        
        return P5, P4, P3, P2, P1

    def initialize(self):
        """全模型初始化（若未加载预训练）"""
        if self.cfg is not None:
            if self.cfg.snapshot:
                # 加载完整模型快照
                self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            # 初始化编码器和解码器
            weight_init(self)
