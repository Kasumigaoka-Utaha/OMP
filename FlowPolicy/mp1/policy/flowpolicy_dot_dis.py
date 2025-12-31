import torch

from mp1.common.pytorch_util import dict_apply
from mp1.policy.flowpolicy import FlowPolicy


class FlowPolicyDotDis(FlowPolicy):
    """FlowPolicy variant that combines dispersive and dot-vector alignment losses."""

    def __init__(
        self,
        dis_loss_weight: float = 0.5,
        dot_loss_weight: float = 0.5,
        **kwargs,
    ):
        self.dis_loss_weight = dis_loss_weight
        self.dot_loss_weight = dot_loss_weight
        super().__init__(**kwargs)

    def compute_loss(self, batch):
        eps = self.eps
        num_segments = self.num_segments
        boundary = self.boundary
        delta = self.delta
        alpha = self.alpha
        reduce_op = torch.mean

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        target = nactions

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)

            this_n_point_cloud = this_nobs['point_cloud'].reshape(
                batch_size, -1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # gt & noise
        a0 = torch.randn(trajectory.shape, device=trajectory.device)

        t = torch.rand(target.shape[0], device=target.device) * (1 - eps) + eps  # 1=sde.T
        r = torch.clamp(t + delta, max=1.0)
        t_expand = t.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        r_expand = r.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        xt = t_expand * target + (1. - t_expand) * a0
        xr = r_expand * target + (1. - r_expand) * a0

        # apply mask
        xt[condition_mask] = cond_data[condition_mask]
        xr[condition_mask] = cond_data[condition_mask]

        segments = torch.linspace(0, 1, num_segments + 1, device=target.device)
        seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
        segment_ends = segments[seg_indices]
        segment_ends_expand = segment_ends.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        x_at_segment_ends = segment_ends_expand * target + (1. - segment_ends_expand) * a0

        def f_euler(t_expand, segment_ends_expand, xt, vt):
            return xt + (segment_ends_expand - t_expand) * vt

        def threshold_based_f_euler(t_expand, segment_ends_expand, xt, vt, threshold, x_at_segment_ends):
            if (threshold, int) and threshold == 0:
                return x_at_segment_ends

            less_than_threshold = t_expand < threshold

            res = (
                less_than_threshold * f_euler(t_expand, segment_ends_expand, xt, vt)
                + (~less_than_threshold) * x_at_segment_ends
            )
            return res

        vt_raw = self.model(xt, t * 99, cond=local_cond, global_cond=global_cond)
        vr_raw = self.model(xr, r * 99, local_cond=local_cond, global_cond=global_cond)

        vt = vt_raw.clone()
        vr = vr_raw.clone()

        # mask
        vt[condition_mask] = cond_data[condition_mask]
        vr[condition_mask] = cond_data[condition_mask]

        vr = torch.nan_to_num(vr)

        ft = f_euler(t_expand, segment_ends_expand, xt, vt)
        fr = threshold_based_f_euler(r_expand, segment_ends_expand, xr, vr, boundary, x_at_segment_ends)

        ##### base flow loss #####
        losses_f = torch.square(ft - fr)
        losses_f = reduce_op(losses_f.reshape(losses_f.shape[0], -1), dim=-1)

        def masked_losses_v(vt, vr, threshold, segment_ends, t):
            if (threshold, int) and threshold == 0:
                return 0

            less_than_threshold = t_expand < threshold

            far_from_segment_ends = (segment_ends - t) > 1.01 * delta
            far_from_segment_ends = far_from_segment_ends.view(-1, 1, 1).repeat(1, trajectory.shape[1], trajectory.shape[2])

            losses_v = torch.square(vt - vr)
            losses_v = less_than_threshold * far_from_segment_ends * losses_v
            losses_v = reduce_op(losses_v.reshape(losses_v.shape[0], -1), dim=-1)

            return losses_v

        losses_v = masked_losses_v(vt, vr, boundary, segment_ends, t)

        flow_loss = torch.mean(losses_f + alpha * losses_v)

        # Dispersive regularization encourages diverse predictions.
        dis_loss = self.dispersive_loss(vt_raw)

        # Encourage directional alignment with the data-to-noise flow.
        vector_target = target - a0
        dot_loss = self.dot_vec_loss(vector_target, vt_raw)

        loss = flow_loss + self.dis_loss_weight * dis_loss + self.dot_loss_weight * dot_loss
        loss_dict = {
            'bc_loss': loss.item(),
            'flow_loss': flow_loss.item(),
            'dis_loss': dis_loss.item(),
            'dot_loss': dot_loss.item(),
        }

        return loss, loss_dict

    @staticmethod
    def dispersive_loss(z, tau: float = 1.0, eps: float = 1e-8):
        # z: (B, T, D)
        dist_matrix = torch.cdist(z, z, p=2) ** 2  # (B, T, T)
        max_val = torch.max(dist_matrix).clamp(min=eps)
        dist_matrix = dist_matrix / max_val
        exp_term = torch.exp(-dist_matrix / tau)
        mean_exp = torch.mean(exp_term)
        loss = torch.log(mean_exp)
        return loss

    @staticmethod
    def dot_vec_loss(v0, v, eps: float = 1e-8):
        v0_norm = torch.norm(v0, dim=(1, 2)).clamp(min=eps)
        v_norm = torch.norm(v, dim=(1, 2)).clamp(min=eps)

        dot_product = torch.sum(v0 * v, dim=(1, 2))
        cosine_vec = dot_product / (v_norm * v0_norm)

        safe_cosine = ((cosine_vec + 1) * 0.5).clamp(min=eps)
        loss = torch.mean(-torch.log(safe_cosine))
        return loss
