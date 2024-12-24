import torch
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat
from info_nce import InfoNCE, info_nce


class United_Model(nn.Module):
    def __init__(self, fMRI_input_dim, sMRI_trad_input_dim, hidden_dim=64, num_classes=2, num_heads=1, num_layers=2,
                 sparsity=30, dropout=0.5,
                 cls_token='sum', readout='sero'):
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token == 'sum':
            self.cls_token = lambda x: x.sum(0)
        elif cls_token == 'mean':
            self.cls_token = lambda x: x.mean(0)
        elif cls_token == 'param':
            self.cls_token = lambda x: x[-1]
        else:
            raise
        if readout == 'garo':
            readout_module = ModuleGARO
        elif readout == 'sero':
            readout_module = ModuleSERO
        elif readout == 'mean':
            readout_module = ModuleMeanReadout
        else:
            raise

        self.num_classes = num_classes
        self.sparsity = sparsity

        ############################### init for fMRI features  ###############################################
        self.initial_linear = nn.Linear(fMRI_input_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList()  # 它可以以列表的形式来存储多个子模块
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))
            self.readout_modules.append(readout_module(hidden_dim=hidden_dim, input_dim=fMRI_input_dim, dropout=0.1))
            self.transformer_modules.append(
                ModuleTransformer(hidden_dim, 2 * hidden_dim, num_heads=num_heads, dropout=0.1))
            self.linear_layers.append(nn.Linear(hidden_dim, num_classes))

        ############################### init for sMRI trad features ###############################################
        # MLP with one layer
        self.sMRI_trad_mlp1 = nn.Sequential(nn.Linear(sMRI_trad_input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim))
        self.sMRI_deeptrad_mlp2 = nn.Sequential(nn.Linear(hidden_dim, num_classes), nn.BatchNorm1d(num_classes),
                                                nn.ReLU())
        self.mlp_combsmri = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.BatchNorm1d(hidden_dim))
        self.fuse_mlp1_concat_2feat = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.BatchNorm1d(hidden_dim))
        self.fuse_mlp2 = nn.Sequential(nn.Linear(hidden_dim, num_classes), nn.BatchNorm1d(num_classes), nn.ReLU())
        self.weights = nn.Parameter(torch.ones(2) / 2, requires_grad=True)  # 2 means two modalities: fMRI, sMRI

        ############################### init for sMRI deep features  ###############################################
        self.conv1 = self._conv_layer_set(1, 16)
        self.conv2 = self._conv_layer_set(16, 32)
        self.conv3 = self._conv_layer_set(32, 64)
        self.conv4 = self._conv_layer_set(64, 64)

        self.conv1_bn = nn.BatchNorm3d(16)
        self.conv2_bn = nn.BatchNorm3d(32)
        self.conv3_bn = nn.BatchNorm3d(64)
        self.conv4_bn = nn.BatchNorm3d(64)

        self.mlp1 = nn.Sequential(nn.Linear(28224, 4096), nn.ReLU(), nn.BatchNorm1d(4096), nn.Dropout(p=0.3))
        self.mlp2 = nn.Sequential(nn.Linear(4096, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(p=0.3))
        self.mlp3 = nn.Sequential(nn.Linear(1024, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(p=0.3))
        self.mlp4 = nn.Sequential(nn.Linear(64, 2), nn.BatchNorm1d(2), nn.ReLU())

        ######################## init for cross-attention operation deep features  ####################################
        self.csatt_fmri_smri = CrossAttention_fmri_smri(hidden_dim, 2 * hidden_dim, num_heads=num_heads, dropout=0.1)
        self.csatt_smri_fmri = CrossAttention_smri_fmri(hidden_dim, 2 * hidden_dim, num_heads=num_heads, dropout=0.1)

    def _collate_adjacency(self, a):  # 将矩阵变成邻接矩阵形式
        # a.shape: (batch_size, segment_num, 116, 116). It calculates FC matrix for each sliding window
        i_list = []
        v_list = []
        for sample, _dyn_a in enumerate(a):  # _dyn_a.shape: (segment_num, 116, 116)
            for timepoint, _a in enumerate(_dyn_a):  # _a.shape: (116, 116)
                thresholded_a = (_a > np.percentile(_a.detach().cpu().numpy(), 100 - self.sparsity))  # 稀疏, 不大于70的为0
                _i = thresholded_a.nonzero(as_tuple=False)  # 非零元素坐标
                # _i.shape: (4036, 2), here 4036 is the number of top 30% elements, 2 corresponds to x_axis and y_axis

                _v = torch.ones(len(_i))  # _v.shape:  (4036, ) and all elements are 1
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]  # batchID*54*116 + segment_numID*116
                # 上一步的操作是对坐标的操作，对应到矩阵上，相当于是将这batch_size*segment_num个处理好的稀疏矩阵放在了一条对角线上，形成一个(batch_size * segment_num * 116, batch_size * segment_num * 116)的大矩阵
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)  # 将这些坐标矩阵按列合并成一个大坐标矩阵
        _v = torch.cat(v_list).to(a.device)  # 将所有数值合并成一个大数值矩阵
        # _i和_v分别为稀疏矩阵表示法中的坐标矩阵和坐标对应的值的矩阵

        return torch.sparse.FloatTensor(_i, _v, (a.shape[0] * a.shape[1] * a.shape[2],
                                                 a.shape[0] * a.shape[1] * a.shape[
                                                     3]))  # (batch_size * segment_num * 116, batch_size * segment_num * 116)

    def _conv_layer_set(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(5, 5, 5), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2))
        return conv_layer

    def forward(self, fMRI_v, fMRI_a, sMRI_trad, sMRI_deep, csatt, CA_MOD):
        logit_fMRI = 0.0
        latent_list = []
        minibatch_size, num_timepoints, num_nodes = fMRI_a.shape[:3]

        ####################################### for fMRI features ##################################################
        # fMRI_v/fMRI_a.shape: (batch_size, segment_num, 116, 116)
        attention = {'node-attention': [], 'time-attention': []}
        h1 = fMRI_v  # fMRI_v: one-hot encoding (constant), h1: (batch_size, segment_num, 116, 116)
        h2 = rearrange(h1, 'b t n c -> (b t n) c')  # h2.shape: (batch_size * segment_num * 116, 116+?0/64)
        h3 = self.initial_linear(h2)  # 全连接层, h3.shape: (batch_size * segment_num * 116, 64)

        a1 = self._collate_adjacency(
            fMRI_a)  # a1.shape: (batch_size * segment_num * 116, batch_size * segment_num * 116)

        for layer, (G, R, T, L) in enumerate(
                zip(self.gnn_layers, self.readout_modules, self.transformer_modules, self.linear_layers)):
            h = G(h3, a1)  # h.shape: (batch_size * segment_num * 116, 64)  # 使用图同构网络(GIN)
            h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size,
                                 n=num_nodes)  # h_bridge.shape: (segment_num, batch_size, 116, 64)
            h_readout, node_attn = R(h_bridge, node_axis=2)  # SE机制？
            # h_readout.shape: (segment_num, batch_size, 64)
            # node_attn.shape: (batch_size, segment_num, 116) if sero or garo; otherwise node_attn=0 (i.e., mean)

            h_attend, time_attn = T(
                h_readout)  # h_readout.shape: (segment_num, batch_size, 64), h_attend.shape: (segment_num, batch_size, 64), time_attn.shape: (batch_size, segment_num, segment_num)

            fMRI_latent = self.cls_token(h_attend)  # fMRI_latent.shape: (batch_size, 64)
            logit_fMRI += self.dropout(L(fMRI_latent))  # logit_fMRI.shape: (batch_size, 2)

            attention['node-attention'].append(node_attn)
            attention['time-attention'].append(time_attn)
            latent_list.append(fMRI_latent)

        # attention['node-attention'].shape: (batch_size, 2, segment_num, 116)
        # attention['time-attention'].shape: (batch_size, 2, segment_num, segment_num)
        attention['node-attention'] = torch.stack(attention['node-attention'],
                                                  dim=1).detach().cpu()  # 把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推
        attention['time-attention'] = torch.stack(attention['time-attention'], dim=1).detach().cpu()  # dim是选择生成的维度
        fMRI_latent_1 = torch.stack(latent_list, dim=1)  # (batch_size, 2, 64)
        feat_fMRI = torch.mean(fMRI_latent_1, dim=1)  # (batch_size, 64)

        ####################################### for sMRI trad features ################################################
        feat_sMRI_trad = self.sMRI_trad_mlp1(sMRI_trad)  # handcrafted features feat_sMRI_trad 即是论文中的S_H
        # logit_sMRI_trad = self.sMRI_trad_mlp2(feat_sMRI_trad)

        ####################################### for sMRI deep features ################################################
        sMRI_deep_x1 = self.conv1(sMRI_deep)  # T1 features
        sMRI_deep_x2 = self.conv1_bn(sMRI_deep_x1)
        sMRI_deep_x3 = self.conv2(sMRI_deep_x2)
        sMRI_deep_x4 = self.conv2_bn(sMRI_deep_x3)
        sMRI_deep_x5 = self.conv3(sMRI_deep_x4)
        sMRI_deep_x6 = self.conv3_bn(sMRI_deep_x5)
        sMRI_deep_x7 = self.conv4(sMRI_deep_x6)
        sMRI_deep_x8 = self.conv4_bn(sMRI_deep_x7)  # 4 3D convolution blocks

        sMRI_deep_x9_vec = sMRI_deep_x8.view(sMRI_deep_x8.size(0), -1)
        sMRI_deep_x10 = self.mlp1(sMRI_deep_x9_vec)
        sMRI_deep_x11 = self.mlp2(sMRI_deep_x10)
        feat_sMRI_deep = self.mlp3(sMRI_deep_x11)  # feat_sMRI_deep 即是论文中的S_D
        # logit_sMRI_deep = self.mlp4(feat_sMRI_deep)

        ################################## cross-attention across 3 modalities ######################################
        if csatt == 'pair2':
            feat_combsMRI = torch.cat((feat_sMRI_deep, feat_sMRI_trad), dim=1)  # should be (bs, 64*2)  将S_D和S_H按行拼接
            feat_combsMRI = self.mlp_combsmri(feat_combsMRI)  # 将拼接好的features经过两层感知机处理
            logit_sMRIdeeptrad = self.sMRI_deeptrad_mlp2(feat_combsMRI)
            feat_fMRI = feat_fMRI

            # Q_s=K_s=V_s=s
            fmriatt_smri, _ = self.csatt_fmri_smri(feat_fMRI.unsqueeze(0), feat_combsMRI.unsqueeze(0),
                                                   feat_combsMRI.unsqueeze(0))  # Q_f，K_s，V_s
            smriatt_fmri, _ = self.csatt_smri_fmri(feat_combsMRI.unsqueeze(0), feat_fMRI.unsqueeze(0),
                                                   feat_fMRI.unsqueeze(0))  # Q_s, K_f, V_f
            # 所谓的交叉注意力本质是Q,K,V不来自同一个特征

        ########################################## cross-attention modes ############################################

        # CA2: align mod and modatt:
        # fmri = fmriatt_smri
        # smri = smriatt_fmri

        # CA3: align each pair of different modalities (original features)
        # fmri = smri

        # CA4: align each pair of different modalities (attended features)
        # fmriatt_smri = smriatt_fmri

        loss_crit_ca2 = InfoNCE()
        loss_ca2 = (loss_crit_ca2(feat_fMRI, fmriatt_smri.squeeze(0)) + loss_crit_ca2(fmriatt_smri.squeeze(0),
                                                                                      feat_fMRI)) + (
                           loss_crit_ca2(feat_combsMRI, smriatt_fmri.squeeze(0)) + loss_crit_ca2(
                       smriatt_fmri.squeeze(0), feat_combsMRI))

        loss_crit_ca3 = InfoNCE()
        loss_ca3 = loss_crit_ca3(feat_fMRI, feat_combsMRI) + loss_crit_ca3(feat_combsMRI, feat_fMRI)

        loss_crit_ca4 = InfoNCE()
        loss_ca4 = loss_crit_ca4(fmriatt_smri.squeeze(0), smriatt_fmri.squeeze(0)) + loss_crit_ca3(
            smriatt_fmri.squeeze(0), fmriatt_smri.squeeze(0))

        if CA_MOD == 'CA2':
            loss_ca = loss_ca2
        elif CA_MOD == 'CA3':
            loss_ca = loss_ca3
        elif CA_MOD == 'CA4':
            loss_ca = loss_ca4
        elif CA_MOD == 'CA23':
            loss_ca = loss_ca2 + loss_ca3
        elif CA_MOD == 'CA24':
            loss_ca = loss_ca2 + loss_ca4
        elif CA_MOD == 'CA34':
            loss_ca = loss_ca3 + loss_ca4
        elif CA_MOD == 'CA234':
            loss_ca = loss_ca2 + loss_ca3 + loss_ca4

        ##### concatenate attended features for classification loss
        # dynamically assign weights for each modality, i.e., fmriatt_smri and smriatt_fmri
        concat_sfMRI_initfeat = torch.cat(
            (self.weights[0] * fmriatt_smri.squeeze(0), self.weights[1] * smriatt_fmri.squeeze(0)),
            dim=1)  # should be (bs, 64*2)
        concat_sfMRI_hidfeat = self.fuse_mlp1_concat_2feat(concat_sfMRI_initfeat)  # final prediction MLP
        concat_sfMRI_logit = self.dropout(self.fuse_mlp2(concat_sfMRI_hidfeat))  # add dropout

        return concat_sfMRI_logit, loss_ca, logit_sMRIdeeptrad, logit_fMRI
        # concat_sfMRI_logit.shape: (batch_size, 2)


class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon:
            self.epsilon = nn.Parameter(torch.Tensor([[0.0]]))  # assumes that the adjacency matrix includes self-loop
        else:
            self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())
        # nn.Linear对矩阵进行线性变换，即Kx+b，其中K和b是需要学习的参数，所以这里是一个两层的感知机。

    def forward(self, v, a):
        # v.shape: (batch_size * segment_num * 116, 64)
        # a.shape: (batch_size * segment_num * 116, batch_size * segment_num * 116)
        v_aggregate = torch.sparse.mm(a, v)  # 稀疏矩阵乘法，a为稀疏矩阵，由值和在矩阵中的坐标组成
        v_aggregate += self.epsilon * v  # assumes that the adjacency matrix includes self-loop
        v_combine = self.mlp(v_aggregate)
        return v_combine


class ModuleMeanReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1, 1, 1], dtype=torch.float32)


class ModuleSERO(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale * hidden_dim)),
                                   nn.BatchNorm1d(round(upscale * hidden_dim)), nn.GELU())
        self.attend = nn.Linear(round(upscale * hidden_dim), input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):  # x.shape: (segment_num, batch_size, 116, 64)
        x_readout = x.mean(node_axis)  # x_readout.shape: (segment_num, batch_size, 64) # 均值压缩矩阵，node_axis表示要压缩的维度
        x_shape = x_readout.shape  # x_shape: (segment_num, batch_size, 64)
        x_embed = self.embed(x_readout.reshape(-1, x_shape[-1]))  # x_embed.shape: (segment_num * batch_size, 64)
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1],
                                                                    -1)  # x_graphattention.shape: (segment_num, batch_size, 116)
        permute_idx = list(range(node_axis)) + [len(x_graphattention.shape) - 1] + list(
            range(node_axis, len(x_graphattention.shape) - 1))  # permute_idx: [0, 1, 2]
        x_graphattention = x_graphattention.permute(permute_idx)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1, 0, 2)


class ModuleGARO(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale * hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale * hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(
            torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n')) / np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1, 0, 2)

# transformer
class ModuleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, input_dim))

    def forward(self, x):  # x.shape: (segment_num, batch_size, 64)
        x_attend, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x_attend)  # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix
        # x_attend.shape: (segment_num, batch_size, 64), attn_matrix.shape: (batch_size, segment_num, segment_num)


class CrossAttention_fmri_smri(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)

    def forward(self, query, key, value):
        x_attend, attn_matrix = self.multihead_attn(query, key, value)
        return x_attend, attn_matrix


class CrossAttention_smri_fmri(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)

    def forward(self, query, key, value):
        x_attend, attn_matrix = self.multihead_attn(query, key, value)
        return x_attend, attn_matrix


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
