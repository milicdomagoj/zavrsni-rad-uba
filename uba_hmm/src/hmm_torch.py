import numpy as np
import torch

NEG_INF = -1e30

def _log(x, eps=1e-8):
    return torch.log(x + eps)

class BatchedGaussianHMM:

    def __init__(self, n_users, n_states, n_features, device, var_floor=1e-3, seed=42):
        self.U, self.K, self.F = n_users, n_states, n_features
        self.device = device
        self.var_floor = var_floor
        g = torch.Generator(device="cpu").manual_seed(seed)

        self.startprob = torch.full((self.U, self.K), 1.0 / self.K, device=device)
        tm = torch.full((self.U, self.K, self.K), 1.0 / self.K, device=device)
        self.transmat = tm
        self.means = torch.randn(self.U, self.K, self.F, generator=g).to(device) * 0.5
        self.vars = torch.ones(self.U, self.K, self.F, device=device)

    def _log_emission(self, obs):
        U, T, F = obs.shape
        x = obs.unsqueeze(2)
        mean = self.means.unsqueeze(1)
        var = self.vars.unsqueeze(1).clamp_min(self.var_floor)
        log_norm = -0.5 * (torch.log(2 * torch.pi * var) + (x - mean) ** 2 / var)
        return log_norm.sum(-1)

    def _forward(self, log_b, mask):
        U, T, K = log_b.shape
        log_start = _log(self.startprob)
        log_trans = _log(self.transmat)
        log_alpha = torch.empty(U, T, K, device=self.device)
        log_alpha[:, 0] = log_start + log_b[:, 0]
        for t in range(1, T):
            prev = log_alpha[:, t - 1].unsqueeze(2)
            trans_term = torch.logsumexp(prev + log_trans, dim=1)
            step = log_b[:, t] + trans_term
            m = mask[:, t].unsqueeze(1)
            log_alpha[:, t] = torch.where(m > 0, step, log_alpha[:, t - 1])
        return log_alpha, log_trans

    def _backward(self, log_b, log_trans, mask):
        U, T, K = log_b.shape
        log_beta = torch.zeros(U, T, K, device=self.device)
        for t in range(T - 2, -1, -1):
            nxt = (log_beta[:, t + 1] + log_b[:, t + 1]).unsqueeze(1)
            step = torch.logsumexp(log_trans + nxt, dim=2)
            m = mask[:, t + 1].unsqueeze(1)
            log_beta[:, t] = torch.where(m > 0, step, log_beta[:, t + 1])
        return log_beta

    @torch.no_grad()
    def kmeans_init(self, obs, mask, iters=10):
        obs_np = obs.cpu().numpy(); mask_np = mask.cpu().numpy()
        means = self.means.cpu().numpy()
        rng = np.random.default_rng(123)
        for u in range(self.U):
            m = mask_np[u] > 0
            X = obs_np[u][m]
            if len(X) < self.K:
                continue
            c = X[rng.choice(len(X), self.K, replace=False)].copy()
            for _ in range(iters):
                d = ((X[:, None, :] - c[None]) ** 2).sum(2)
                a = d.argmin(1)
                for k in range(self.K):
                    if (a == k).any():
                        c[k] = X[a == k].mean(0)
            means[u] = c
        self.means = torch.tensor(means, device=self.device)

    def fit(self, obs, mask, n_iter=25, kmeans=False, verbose=False):
        obs = obs.to(self.device)
        mask = mask.to(self.device).float()
        if kmeans:
            self.kmeans_init(obs, mask)
        history = []
        for it in range(n_iter):
            log_b = self._log_emission(obs)
            log_b = log_b * mask.unsqueeze(2)
            log_alpha, log_trans = self._forward(log_b, mask)
            log_beta = self._backward(log_b, log_trans, mask)

            lengths = mask.sum(1).long().clamp_min(1)
            idx = (lengths - 1)
            logZ = torch.logsumexp(log_alpha[torch.arange(self.U), idx], dim=1)
            history.append(float(logZ.mean().item()))

            log_gamma = log_alpha + log_beta - logZ.view(-1, 1, 1)
            gamma = log_gamma.exp() * mask.unsqueeze(2)

            la = log_alpha[:, :-1].unsqueeze(3)
            lb = (log_beta[:, 1:] + log_b[:, 1:]).unsqueeze(2)
            log_xi = la + log_trans.unsqueeze(1) + lb - logZ.view(-1, 1, 1, 1)
            pair_mask = (mask[:, :-1] * mask[:, 1:]).view(self.U, -1, 1, 1)
            xi = log_xi.exp() * pair_mask

            self.startprob = (gamma[:, 0] + 1e-8)
            self.startprob = self.startprob / self.startprob.sum(1, keepdim=True)

            trans_num = xi.sum(1)
            trans_den = gamma[:, :-1].sum(1).unsqueeze(2)
            self.transmat = (trans_num + 1e-8) / (trans_den + 1e-8 * self.K)
            self.transmat = self.transmat / self.transmat.sum(2, keepdim=True)

            gk = gamma.sum(1).clamp_min(1e-8)
            num_mean = torch.einsum("utk,utf->ukf", gamma, obs)
            self.means = num_mean / gk.unsqueeze(2)
            diff2 = (obs.unsqueeze(2) - self.means.unsqueeze(1)) ** 2
            num_var = torch.einsum("utk,utkf->ukf", gamma, diff2)
            self.vars = (num_var / gk.unsqueeze(2)).clamp_min(self.var_floor)

            if verbose and (it % 5 == 0 or it == n_iter - 1):
                print(f"      EM iter {it+1}/{n_iter}  prosj. log-izglednost={history[-1]:.3f}")
        return history

    @torch.no_grad()
    def dominant_state_loglik(self, obs, mask, occ_thresh=0.10):
        obs = obs.to(self.device); mask = mask.to(self.device).float()
        log_b = self._log_emission(obs)
        pi = self.stationary()
        dom = (pi >= occ_thresh)
        none_dom = dom.sum(1) == 0
        if none_dom.any():
            top = pi.argmax(1)
            dom[none_dom, top[none_dom]] = True
        masked_b = log_b.clone()
        masked_b[~dom.unsqueeze(1).expand_as(masked_b)] = NEG_INF
        best = masked_b.max(dim=2).values
        best[mask == 0] = float("nan")
        return best

    @torch.no_grad()
    def stationary(self, n_iter=50):
        pi = torch.full((self.U, self.K), 1.0 / self.K, device=self.device)
        tm = self.transmat
        for _ in range(n_iter):
            pi = torch.bmm(pi.unsqueeze(1), tm).squeeze(1)
            pi = pi / pi.sum(1, keepdim=True).clamp_min(1e-12)
        return pi

    @torch.no_grad()
    def marginal_loglik(self, obs, mask):
        obs = obs.to(self.device); mask = mask.to(self.device).float()
        log_b = self._log_emission(obs)
        log_pi = _log(self.stationary())
        ll = torch.logsumexp(log_b + log_pi.unsqueeze(1), dim=2)
        ll[mask == 0] = float("nan")
        return ll

    @torch.no_grad()
    def conditional_loglik(self, obs, mask):
        obs = obs.to(self.device); mask = mask.to(self.device).float()
        log_b = self._log_emission(obs) * mask.unsqueeze(2)
        log_alpha, _ = self._forward(log_b, mask)
        c = torch.logsumexp(log_alpha, dim=2)
        cond = torch.empty_like(c)
        cond[:, 0] = c[:, 0]
        cond[:, 1:] = c[:, 1:] - c[:, :-1]
        cond[mask == 0] = float("nan")
        return cond
