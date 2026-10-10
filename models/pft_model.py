import torch
import torch.nn.functional as F
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import get_root_logger
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class PFTModel(SRModel):

    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']

        # 正交损失：惩罚 td_init token 之间的余弦相似度，鼓励方向正交
        self.lambda_ortho = train_opt.get('lambda_ortho', 0.0)

        # 激活均匀性损失：惩罚 ATD 注意力分布的不均匀，鼓励所有 token 被均等使用
        self.lambda_spread = train_opt.get('lambda_spread', 0.0)

        # 每隔多少 iter 收集一次注意力（收集有额外开销，不必每步都做）
        self.spread_collect_freq = train_opt.get('spread_collect_freq', 1)

        logger = get_root_logger()
        if self.lambda_ortho > 0:
            logger.info(f'Dict diversity loss — orthogonal: lambda_ortho={self.lambda_ortho}')
        if self.lambda_spread > 0:
            logger.info(f'Dict diversity loss — spread: lambda_spread={self.lambda_spread}, '
                        f'collect_freq={self.spread_collect_freq}')

    def _dict_diversity_loss(self, net, attn_maps):
        """
        计算字典多样性损失，包含两个互补项：

        1. 正交损失 (l_ortho)
           对 td_init 的归一化 Gram 矩阵施加 off-diagonal 惩罚。
           目标：让各 token 的方向尽量正交，避免冗余。
           L_ortho = mean((G - I)^2),  G_ij = cos(td_i, td_j)

        2. 激活均匀性损失 (l_spread)
           收集所有 ATD_CA_lite 层的 softmax 注意力图 (b, n, m)，
           对每个 token j 计算其在所有位置上的平均激活概率 p_j，
           然后用负熵惩罚来鼓励均匀分布：
           L_spread = sum(p_j * log(p_j))  （越均匀熵越大，该值越小）

        返回 (l_ortho, l_spread)，未启用时返回 None。
        """
        l_ortho = None
        l_spread = None

        # ── 正交损失 ──────────────────────────────────────────────────────────
        if self.lambda_ortho > 0 and hasattr(net, 'td_init'):
            td_norm = F.normalize(net.td_init, dim=-1)   # (m, c)
            gram = td_norm @ td_norm.T                    # (m, m)
            m = gram.shape[0]
            identity = torch.eye(m, device=gram.device)
            l_ortho = ((gram - identity) ** 2).mean()

        # ── 激活均匀性损失 ────────────────────────────────────────────────────
        if self.lambda_spread > 0 and attn_maps is not None and len(attn_maps) > 0:
            # attn_maps: list of (b, n, m) tensors，每层一个
            # 拼接所有层：(L*b, n, m)，再对空间维 n 做平均得到每层每样本的 token 使用分布
            stacked = torch.cat([a.float() for a in attn_maps], dim=0)   # (L*b, n, m)
            # 对空间位置 n 平均，得到每个 token 的平均激活概率
            p = stacked.mean(dim=1)                 # (L*b, m)
            # 归一化确保是概率分布（softmax 输出已归一化，mean 后仍归一化）
            p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
            # 负熵惩罚：鼓励均匀分布（均匀时熵最大，负熵最小）
            l_spread = (p * torch.log(p + 1e-8)).sum(dim=-1).mean()

        return l_ortho, l_spread

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad(set_to_none=True)

        # 判断本步是否需要收集注意力图
        need_attn = (self.lambda_spread > 0 and
                     current_iter % self.spread_collect_freq == 0)

        net = self.get_bare_model(self.net_g)
        # 仅当模型支持 collect_attn 参数时才启用（即 PFTv3v2_tdrefresh）
        supports_collect = hasattr(net, 'td_init') and callable(getattr(net, 'forward', None))

        with self.amp_cast():
            if need_attn and supports_collect:
                result = self.net_g(self.lq, collect_attn=True)
                self.output, attn_maps = result
            else:
                self.output = self.net_g(self.lq)
                attn_maps = None

        l_total = 0
        loss_dict = OrderedDict()
        output_fp32 = self.output.float()

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(output_fp32, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(output_fp32, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # 字典多样性损失
        if self.lambda_ortho > 0 or (self.lambda_spread > 0 and attn_maps is not None):
            l_ortho, l_spread = self._dict_diversity_loss(net, attn_maps)
            if l_ortho is not None:
                l_total += self.lambda_ortho * l_ortho
                loss_dict['l_ortho'] = l_ortho
            if l_spread is not None:
                l_total += self.lambda_spread * l_spread
                loss_dict['l_spread'] = l_spread

        self.amp_backward_and_step(l_total)

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # 每 1000 步监控字典健康度
        if current_iter % 1000 == 0 and hasattr(net, 'td_init'):
            try:
                logger = get_root_logger()
                td_norm = F.normalize(net.td_init.detach(), dim=-1)
                gram = td_norm @ td_norm.T
                m = gram.shape[0]
                off_diag = gram - torch.eye(m, device=gram.device)
                avg_sim = off_diag.abs().mean().item()
                logger.info(f'[td_init] avg off-diag cosine sim: {avg_sim:.4f} '
                            f'(lower = more orthogonal)')
            except Exception:
                pass

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad(), self.amp_cast():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad(), self.amp_cast():
                self.output = self.net_g(self.lq)
            self.net_g.train()
        self.output = self.output.float()


@MODEL_REGISTRY.register()
class PFTCovModel(SRModel):
    """
    Training model for TDERN.

    The network's forward() returns (sr, attn_list) during training and sr
    during inference.  This model unpacks that tuple and computes the
    coverage loss via the network's own compute_coverage_loss() method.

    yml keys (under 'train'):
        lambda_cov (float): weight for coverage loss. Default: 0.0 (disabled).
    """

    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']
        self.lambda_cov = train_opt.get('lambda_cov', 0.0)
        logger = get_root_logger()
        if self.lambda_cov > 0:
            logger.info(f'Coverage loss enabled: lambda_cov={self.lambda_cov}')

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad(set_to_none=True)

        # forward() returns (sr, attn_list) in training mode
        with self.amp_cast():
            result = self.net_g(self.lq)
        if isinstance(result, tuple):
            self.output, attn_list = result
        else:
            self.output, attn_list = result, None

        l_total = 0
        loss_dict = OrderedDict()
        output_fp32 = self.output.float()

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(output_fp32, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(output_fp32, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # coverage loss — delegated to the network's own helper
        if self.lambda_cov > 0 and attn_list is not None:
            net = self.get_bare_model(self.net_g)
            attn_fp32 = [a.float() for a in attn_list]
            l_cov = net.compute_coverage_loss(attn_fp32, lam=self.lambda_cov)
            l_total += l_cov
            loss_dict['l_cov'] = l_cov

        self.amp_backward_and_step(l_total)

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        # inference: forward() returns plain sr tensor (no attn_list)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad(), self.amp_cast():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad(), self.amp_cast():
                self.output = self.net_g(self.lq)
            self.net_g.train()
        self.output = self.output.float()
