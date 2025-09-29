import torch
import torch.nn as nn
from mcd.src.mlp import MLP
from mcd.src.moe_all import MoE
# from mcd.src.moe_wo_template import MoE
from mcd.module.base_module import CrossModule
from usfutils.utils import set_seed_everything

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)  # 全连接层
        return out

class MCDModel(nn.Module):
    def __init__(self, dataNumber, config, is_cold_start=False):
        super().__init__()
        self.config = config
        self.dataNumber = dataNumber
        self.is_cold_start = is_cold_start
        self.decay_rate = config.decay_rate
        # video content
        config1 = config.v_moe
        cls_dim = config1.cls_dim
        feature_dim = config1.feature_dim
        feature_dim2 = config1.feature_dim2
        feature_hidden_dim = config1.feature_hidden_dim
        num_trans_heads = config1.num_trans_heads
        dropout = config1.dropout
        self.linear_video = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_title = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_author = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_asr = nn.Sequential(
            nn.Linear(feature_dim, feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_topics1 = nn.Sequential(
            nn.Linear(feature_dim2, feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_topics2 = nn.Sequential(
            nn.Linear(feature_dim2, feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_comment = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim * 2),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.moe = MoE(
            input_size=config1.moe_in_feature * 3,  # * 3
            hidden_size=config1.moe_hidden_feature,
            output_size=config1.moe_out_feature,
            num_experts=config1.num_experts,
            model=MLP,
            k=config1.k,
        )
        self.moe2 = MoE(
            input_size=config1.moe_in_feature * 2,  # * 2
            hidden_size=config1.moe_hidden_feature,
            output_size=config1.moe_out_feature,
            num_experts=config1.num_experts,
            model=MLP,
            k=config1.k,
        )
        self.attn_weights = None  # 保存权重
        self.attn_weights2 = None  # 保存权重
        self.multihead_attn = nn.MultiheadAttention(feature_hidden_dim * 2, config1.num_feature, batch_first=True)
        self.multihead_attn2 = nn.MultiheadAttention(feature_hidden_dim * 2, config1.num_feature, batch_first=True)

        self.crossmodule = CrossModule(feature_hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(cls_dim, int(cls_dim / 2)),
            nn.LayerNorm(int(cls_dim / 2)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(int(cls_dim / 2), 2),
        )
        # 回归任务头
        self.gru = GRUModel(
            input_size=cls_dim,
            hidden_size=cls_dim,
            num_layers=3,
            output_size=cls_dim
        )
        self.regressor = nn.Sequential(
            nn.Linear(cls_dim, int(cls_dim / 2)),
            nn.LayerNorm(int(cls_dim / 2)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(int(cls_dim / 2), 1),  # 输出一个标量（评论数目）
        )

        self.reduce_dim1 = nn.Linear(cls_dim * 7, cls_dim * 3)  # cls_dim * 4, cls_dim * 2 512, 256
        self.reduce_dim2 = nn.Linear(cls_dim * 2, cls_dim * 2)  # cls_dim * 4, cls_dim * 2 512, 256
        self.proj1 = nn.Linear(feature_hidden_dim * 6, feature_hidden_dim * 2)
        self.proj2 = nn.Linear(feature_hidden_dim * 2, feature_hidden_dim * 2)

    def forward(self, **kwargs):
        # video-features
        video_fea = kwargs["video_feas"]
        video_fea = self.linear_video(video_fea)
        # video_fea = torch.mean(video_fea, -2)
        asr_fea = kwargs["asr_feas"]
        asr_fea = self.linear_asr(asr_fea)
        # topics-features
        topics_fea = kwargs["topics_fea"]
        # publish-features
        title_fea = kwargs["title_feas"]
        title_fea = self.linear_title(title_fea)
        author_fea = kwargs["author_feas"]
        author_fea = self.linear_author(author_fea)
        # commenter-features
        comments_feature = kwargs["comment_feas"][:, :self.dataNumber, :]
        comments_fea = self.linear_comment(comments_feature)
        # comments_fea2 = torch.mean(comments_fea, dim=1)

        # video-features
        dim = video_fea.shape[1]
        asr_fea_expanded = asr_fea.unsqueeze(1).expand(-1, dim, -1)  # → [256, 10, 128]
        v_fea = torch.cat([video_fea, asr_fea_expanded], dim=-1)  # → [256, 10, 256]
        # publish-features
        p_fea = torch.cat((title_fea, author_fea), 1)  # → [256, 256]

        # guide-features
        p_fea2 = p_fea.unsqueeze(1)  # → [256, 1, 256]
        p_v_fea, atw = self.multihead_attn(p_fea2, v_fea, v_fea)  # → [256, 1, 256]  # , average_attn_weights=False
        p_v_fea2 = p_v_fea + p_fea2  # → [256, 1, 256]
        self.attn_weights = atw
        c_v_fea, atw2 = self.multihead_attn2(comments_fea, v_fea, v_fea)  # , average_attn_weights=False
        c_v_fea2 = c_v_fea + comments_fea  # → [256, x, 256]
        c_v_fea2 = torch.mean(c_v_fea2, dim=1, keepdim=True)  # → [256, 1, 256]
        self.attn_weights2 = atw2  # 保存权重
        # 计算差异
        diff = p_v_fea2 - c_v_fea2  # [256, 1, 256]
        w = torch.sigmoid(self.proj1(torch.cat([p_v_fea2, c_v_fea2, diff.abs()], dim=-1)))  # [768, 1, 256]
        conflict_space = self.proj2(diff)  # [256, 1, 256]
        weighted_conflict = w * conflict_space  # [256, 1, 256]

        # concat
        if self.is_cold_start:  # 冷启动 没有评论数据
            # 冷启动，没有主题词
            # fea = torch.mean(p_v_fea2, dim=1)  # → [256, 256]
            # fea = self.reduce_dim2(fea)  # → [256, 256]
            # cls_fea, reg_fea, moe_loss_cls, moe_loss_reg = self.moe2(fea)  # → [512, 128]

            # 冷启动，有主题词
            p_v_fea2 = torch.mean(p_v_fea2, dim=1)  # → [256, 256]
            topics_fea = self.linear_topics2(topics_fea)
            fea = torch.concat((topics_fea, p_v_fea2), -1)  # → [256, 1, 256]
            cls_fea, reg_fea, moe_loss_cls, moe_loss_reg = self.moe(fea)  # → [512, 128]

        else:
            topics_fea = self.linear_topics1(topics_fea)
            topics_fea = topics_fea.unsqueeze(1)  # [256, 1, 256]
            fea = torch.concat((p_v_fea2, c_v_fea2, weighted_conflict, topics_fea), -1)  # → [256, 1, 256]
            fea = torch.mean(fea, dim=1)  # → [256, 256]
            fea = self.reduce_dim1(fea)  # → [256, 256]
            cls_fea, reg_fea, moe_loss_cls, moe_loss_reg = self.moe(fea)  # → [512, 128]

        # 主任务输出（争议检测）
        cls_output = self.classifier(cls_fea)
        # 辅助任务输出（评论数目预测）
        reg_output0 = self.gru(reg_fea)
        reg_output = self.regressor(reg_output0)
        # return cls_output, reg_output
        return cls_output, reg_output, moe_loss_cls, moe_loss_reg


# 消融guideAttn
class MCDModel_wo_guideAttn(nn.Module):
    def __init__(self, dataNumber, config, is_cold_start=False):
        super().__init__()
        self.config = config
        self.dataNumber = dataNumber
        self.is_cold_start = is_cold_start
        self.decay_rate = config.decay_rate
        # video content
        config1 = config.v_moe
        cls_dim = config1.cls_dim
        feature_dim = config1.feature_dim
        feature_dim2 = config1.feature_dim2
        feature_hidden_dim = config1.feature_hidden_dim
        dropout = config1.dropout
        self.linear_video = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_title = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_author = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_asr = nn.Sequential(
            nn.Linear(feature_dim, feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_topics1 = nn.Sequential(
            nn.Linear(feature_dim2, feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_comment = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim * 2),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.moe = MoE(
            input_size=config1.moe_in_feature * 3,  # * 3
            hidden_size=config1.moe_hidden_feature,
            output_size=config1.moe_out_feature,
            num_experts=config1.num_experts,
            model=MLP,
            k=config1.k,
        )

        self.classifier = nn.Sequential(
            nn.Linear(cls_dim, int(cls_dim / 2)),
            nn.LayerNorm(int(cls_dim / 2)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(int(cls_dim / 2), 2),
        )
        # 回归任务头
        self.gru = GRUModel(
            input_size=cls_dim,
            hidden_size=cls_dim,
            num_layers=3,
            output_size=cls_dim
        )
        self.regressor = nn.Sequential(
            nn.Linear(cls_dim, int(cls_dim / 2)),
            nn.LayerNorm(int(cls_dim / 2)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(int(cls_dim / 2), 1),  # 输出一个标量（评论数目）
        )
        self.linear_vp = nn.Sequential(
            nn.Linear(cls_dim * 4, cls_dim * 3),
            nn.LayerNorm(cls_dim * 3),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(cls_dim * 3, cls_dim * 3),
        )
        self.linear_vc = nn.Sequential(
            nn.Linear(cls_dim * 4, cls_dim * 3),
            nn.LayerNorm(cls_dim * 3),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(cls_dim * 3, cls_dim * 3),
        )

        self.reduce_dim1 = nn.Linear(cls_dim * 7, cls_dim * 3)

    def forward(self, **kwargs):
        # video-features
        video_fea = kwargs["video_feas"]
        video_fea = self.linear_video(video_fea)
        # video_fea = torch.mean(video_fea, -2)
        asr_fea = kwargs["asr_feas"]
        asr_fea = self.linear_asr(asr_fea)
        # topics-features
        topics_fea = kwargs["topics_fea"]
        topics_fea = self.linear_topics1(topics_fea)
        # publish-features
        title_fea = kwargs["title_feas"]
        title_fea = self.linear_title(title_fea)
        author_fea = kwargs["author_feas"]
        author_fea = self.linear_author(author_fea)
        # commenter-features
        comments_feature = kwargs["comment_feas"][:, :self.dataNumber, :]
        comments_fea = self.linear_comment(comments_feature)
        comments_fea2 = torch.mean(comments_fea, dim=1, keepdim=True)
        # publish-features
        p_fea = torch.cat((title_fea, author_fea), 1)  # → [B, 256]

        # video-features
        dim = video_fea.shape[1]
        asr_fea_expanded = asr_fea.unsqueeze(1).expand(-1, dim, -1)  # → [B, 10, 128]
        v_fea = torch.cat([video_fea, asr_fea_expanded], dim=-1)  # → [B, 10, 256]
        v_fea2 = torch.mean(v_fea, dim=1, keepdim=True)   # [B, 1, 256]

        p_fea = p_fea.unsqueeze(1)
        f1 = torch.mean(torch.cat([v_fea2, p_fea], dim=-1), dim=1)  # [B, 512]
        v_p = self.linear_vp(f1)  # [B, 256]
        f2 = torch.mean(torch.cat([v_fea2, comments_fea2], dim=-1), dim=1)  # [B, 512]
        v_c = self.linear_vc(f2)  # [B, 256]

        fea = torch.concat((topics_fea, v_p, v_c), -1)  # → [B, 1, 128 * 3]
        fea = self.reduce_dim1(fea)  # → [B, 128 * 3]
        cls_fea, reg_fea, moe_loss_cls, moe_loss_reg = self.moe(fea)  # → [B, 128]

        # 主任务输出（争议检测）
        cls_output = self.classifier(cls_fea)
        # 辅助任务输出（评论数目预测）
        reg_output0 = self.gru(reg_fea)
        reg_output = self.regressor(reg_output0)

        return cls_output, reg_output, moe_loss_cls, moe_loss_reg

# 消融taskMOE
class MCDModel_wo_taskMOE(nn.Module):
    def __init__(self, dataNumber, config, is_cold_start=False):
        super().__init__()
        self.config = config
        self.dataNumber = dataNumber
        self.is_cold_start = is_cold_start
        self.decay_rate = config.decay_rate
        # video content
        config1 = config.v_moe
        cls_dim = config1.cls_dim
        feature_dim = config1.feature_dim
        feature_dim2 = config1.feature_dim2
        feature_hidden_dim = config1.feature_hidden_dim
        dropout = config1.dropout

        self.linear_video = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_title = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_author = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_asr = nn.Sequential(
            nn.Linear(feature_dim, feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_topics1 = nn.Sequential(
            nn.Linear(feature_dim2, feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_topics2 = nn.Sequential(
            nn.Linear(feature_dim2, feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_comment = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim * 2),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.attn_weights = None  # 保存权重
        self.attn_weights2 = None  # 保存权重
        self.multihead_attn = nn.MultiheadAttention(feature_hidden_dim * 2, config1.num_feature, batch_first=True)
        self.multihead_attn2 = nn.MultiheadAttention(feature_hidden_dim * 2, config1.num_feature, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(cls_dim * 3, int(cls_dim / 2)),
            nn.LayerNorm(int(cls_dim / 2)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(int(cls_dim / 2), 2),
        )
        # 回归任务头
        self.gru = GRUModel(
            input_size=cls_dim * 3,
            hidden_size=cls_dim,
            num_layers=3,
            output_size=cls_dim
        )
        self.regressor = nn.Sequential(
            nn.Linear(cls_dim, int(cls_dim / 2)),
            nn.LayerNorm(int(cls_dim / 2)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(int(cls_dim / 2), 1),  # 输出一个标量（评论数目）
        )

        self.reduce_dim1 = nn.Linear(cls_dim * 7, cls_dim * 3)  # cls_dim * 4, cls_dim * 2 512, 256
        self.proj1 = nn.Linear(feature_hidden_dim * 6, feature_hidden_dim * 2)
        self.proj2 = nn.Linear(feature_hidden_dim * 2, feature_hidden_dim * 2)

    def forward(self, **kwargs):
        # video-features
        video_fea = kwargs["video_feas"]
        video_fea = self.linear_video(video_fea)
        asr_fea = kwargs["asr_feas"]
        asr_fea = self.linear_asr(asr_fea)
        # topics-features
        topics_fea = kwargs["topics_fea"]
        # publish-features
        title_fea = kwargs["title_feas"]
        title_fea = self.linear_title(title_fea)
        author_fea = kwargs["author_feas"]
        author_fea = self.linear_author(author_fea)
        # commenter-features
        comments_feature = kwargs["comment_feas"][:, :self.dataNumber, :]
        comments_fea = self.linear_comment(comments_feature)

        # video-features
        dim = video_fea.shape[1]
        asr_fea_expanded = asr_fea.unsqueeze(1).expand(-1, dim, -1)  # → [256, 10, 128]
        v_fea = torch.cat([video_fea, asr_fea_expanded], dim=-1)  # → [256, 10, 256]
        # publish-features
        p_fea = torch.cat((title_fea, author_fea), 1)  # → [256, 256]

        # guide-features
        p_fea2 = p_fea.unsqueeze(1)  # → [256, 1, 256]
        p_v_fea, atw = self.multihead_attn(p_fea2, v_fea, v_fea)  # → [256, 1, 256]  # , average_attn_weights=False
        p_v_fea2 = p_v_fea + p_fea2  # → [256, 1, 256]
        self.attn_weights = atw
        c_v_fea, atw2 = self.multihead_attn2(comments_fea, v_fea, v_fea)  # , average_attn_weights=False
        c_v_fea2 = c_v_fea + comments_fea  # → [256, x, 256]
        c_v_fea2 = torch.mean(c_v_fea2, dim=1, keepdim=True)  # → [256, 1, 256]
        self.attn_weights2 = atw2  # 保存权重
        # 计算差异
        diff = p_v_fea2 - c_v_fea2  # [256, 1, 256]
        w = torch.sigmoid(self.proj1(torch.cat([p_v_fea2, c_v_fea2, diff.abs()], dim=-1)))  # [768, 1, 256]
        conflict_space = self.proj2(diff)  # [256, 1, 256]
        weighted_conflict = w * conflict_space  # [256, 1, 256]

        # 添加topic特征
        topics_fea = self.linear_topics1(topics_fea)
        topics_fea = topics_fea.unsqueeze(1)  # [256, 1, 256]
        fea = torch.concat((p_v_fea2, c_v_fea2, weighted_conflict, topics_fea), -1)  # → [256, 1, 256]
        fea = torch.mean(fea, dim=1)  # → [256, 256]
        fea = self.reduce_dim1(fea)  # → [256, 256]

        # 主任务输出（争议检测）
        cls_output = self.classifier(fea)
        # 辅助任务输出（评论数目预测）
        reg_output0 = self.gru(fea)
        reg_output = self.regressor(reg_output0)
        return cls_output, reg_output
