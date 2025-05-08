import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        if x.ndim == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2)
            x = super().forward(x)
            x = x.transpose(1, 2).view(B, C, H, W)
        else:
            x = super().forward(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.transpose(1, 2).view(B, -1, H, W)
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(LayerNorm, eps=1e-6),
        depths=[2,2,2,2],
        sr_ratios=[8, 4, 2, 1],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.mix1 = MixStyle()

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        # self.patch_embed3 = OverlapPatchEmbed(
        #     img_size=img_size // 8,
        #     patch_size=3,
        #     stride=2,
        #     in_chans=embed_dims[1],
        #     embed_dim=embed_dims[2],
        # )
        # self.patch_embed4 = OverlapPatchEmbed(
        #     img_size=img_size // 16,
        #     patch_size=3,
        #     stride=2,
        #     in_chans=embed_dims[2],
        #     embed_dim=embed_dims[3],
        # )

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.Sequential(
            *[
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.Sequential(
            *[
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        # cur += depths[1]
        # self.block3 = nn.Sequential(
        #     *[
        #         Block(
        #             dim=embed_dims[2],
        #             num_heads=num_heads[2],
        #             mlp_ratio=mlp_ratios[2],
        #             qkv_bias=qkv_bias,
        #             qk_scale=qk_scale,
        #             drop=drop_rate,
        #             attn_drop=attn_drop_rate,
        #             drop_path=dpr[cur + i],
        #             norm_layer=norm_layer,
        #             sr_ratio=sr_ratios[2],
        #         )
        #         for i in range(depths[2])
        #     ]
        # )
        # self.norm3 = norm_layer(embed_dims[2])
        #
        # cur += depths[2]
        # self.block4 = nn.Sequential(
        #     *[
        #         Block(
        #             dim=embed_dims[3],
        #             num_heads=num_heads[3],
        #             mlp_ratio=mlp_ratios[3],
        #             qkv_bias=qkv_bias,
        #             qk_scale=qk_scale,
        #             drop=drop_rate,
        #             attn_drop=attn_drop_rate,
        #             drop_path=dpr[cur + i],
        #             norm_layer=norm_layer,
        #             sr_ratio=sr_ratios[3],
        #         )
        #         for i in range(depths[3])
        #     ]
        # )
        # self.norm4 = norm_layer(embed_dims[3])


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        pass

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed1",
            "pos_embed2",
            "pos_embed3",
            "pos_embed4",
            "cls_token",
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        outs = []

        # stage 1
        x = self.patch_embed1(x)
        x = self.block1(x)
        x = self.norm1(x).contiguous()
        x = self.mix1(x)
        outs.append(x)

        # stage 2

        x = self.patch_embed2(x)
        x = self.block2(x)
        x = self.norm2(x).contiguous()
        outs.append(x)

        # stage 3
        # x = self.patch_embed3(x)
        # x = self.block3(x)
        # x = self.norm3(x).contiguous()
        # outs.append(x)
        #
        # # stage 4
        # x = self.patch_embed4(x)
        # x = self.block4(x)
        # x = self.norm4(x).contiguous()
        # outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        return x

import random


class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(0.1, 0.1)
        self.eps = eps
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x



        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        index = lmda < 0.5
        lmda[index] = 1 - lmda[index]
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix



class ConditionalDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        """
        具有动态控制能力的条件深度可分离卷积（Depthwise + Pointwise）。
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param padding: 填充
        :param bias: 是否使用偏置
        :param dynamic_strength: 控制动态权重影响力的参数（0~1之间）
        """
        super(ConditionalDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 条件特征生成（自适应池化 + 逐点卷积）
        self.condition_generator = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3,stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1),  # 加强通道表征
            nn.ReLU()  # 控制动态特征
        )

        # 生成动态权重（用于深度可分离卷积）
        self.weight_generator = nn.Sequential(
            nn.Conv2d(16, in_channels * kernel_size * kernel_size, kernel_size=1),
            nn.ReLU()
        )

        # 生成 Pointwise 卷积的权重
        self.pointwise_generator = nn.Sequential(
            nn.Conv2d(16, in_channels * out_channels, kernel_size=1),
            nn.ReLU() # 控制动态性
        )

        # 动态偏置生成（可选）
        self.bias_generator = nn.Conv2d(16, out_channels, kernel_size=1) if bias else None

        self.bn_pointwise = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # 2. 计算条件特征（代表输入的高层信息）
        condition = self.condition_generator(x)  # (B, 16, 1, 1)

        # 3. 生成动态深度卷积核
        depthwise_weight = self.weight_generator(condition).view(
            batch_size * self.in_channels, 1, self.kernel_size, self.kernel_size
        )

        # 4. 生成 Pointwise 卷积的动态权重
        pointwise_weight = self.pointwise_generator(condition).view(
            batch_size * self.out_channels, self.in_channels, 1, 1
        )

        # 5. 生成动态偏置（如果有的话）
        bias = self.bias_generator(condition).view(batch_size * self.out_channels) if self.bias_generator else None

        # 6. 计算动态深度卷积
        x = x.view(1, batch_size * self.in_channels, height, width)
        dynamic_depthwise_out = F.conv2d(x, depthwise_weight, stride=self.stride, padding=self.padding,
                                         groups=batch_size * self.in_channels)

        # 7. 计算动态 Pointwise 卷积
        dynamic_pointwise_out = F.conv2d(dynamic_depthwise_out, pointwise_weight, bias, stride=1, padding=0,
                                         groups=batch_size)

        # 8. 计算最终输出
        dynamic_pointwise_out = dynamic_pointwise_out.view(batch_size, self.out_channels, height, width)
        dynamic_pointwise_out = self.bn_pointwise(dynamic_pointwise_out)
        output = dynamic_pointwise_out

        return output

class conv_block_cond(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1):
        super(conv_block_cond, self).__init__()

        self.dynamic = nn.Sequential(
            ConditionalDepthwiseConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=stride)

    def forward(self, input):
        return F.relu(self.conv(input) + self.dynamic(input))


class SCTNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(SCTNet, self).__init__()

        self.CNN_encoder = conv_block_cond(in_channels, 16)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.Trans_encoder = MixVisionTransformer()

        self.decoder3 = conv_block_cond(64, 64)
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.decoder2 = conv_block_cond(64, 32)
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.decoder1 = conv_block_cond(48, 24)
        self.up_final = nn.ConvTranspose2d(24, num_classes, kernel_size=2, stride=2)

        self.init_backbone()

    def init_backbone(self):
        ckpt = '/data/weix/SCTNet/mit_b0.pth'
        model_state_dict = torch.load(ckpt)
        self.Trans_encoder.load_state_dict(model_state_dict, strict=False)

    def forward(self, input):
        CNN_enc1 = self.downsample(self.CNN_encoder(input))

        enc2, enc3 = self.Trans_encoder(input)

        dec3 = self.decoder3(enc3)
        dec2 = self.upconv2(dec3)

        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)

        dec1 = torch.cat((dec1, CNN_enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec = self.up_final(dec1)

        return dec








from thop import profile
from thop import clever_format
if __name__ == '__main__':
    # fake_image = torch.rand(1, 3, 256, 256)
    model = SCTNet(in_channels=3, num_classes=2)
    # output = model(fake_image)
    # print(output.size())

  
    input = torch.randn(1, 3, 256, 256)

    macs, params = profile(model, inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"macs: {macs}")
    print(f"Parameters: {params}")


