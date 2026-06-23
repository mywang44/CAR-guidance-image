但问题你看第一作者写的这个2D的代码中的_compute_online_loss_gradient，我是迁移错了吗class GCovAGMOnlineGuidance(GCovAGMGuidance):
    def _prepare_models_for_subclass(self):
        # Check if we need vector field output (for gradient regression)
        online_loss_type = getattr(self.cfg, "online_loss_type", "ground_truth")
        
        # For gradient mode, output vector field
        if online_loss_type == "gradient":
            batch = self.sample_batch_fn(2, self.device)
            inputs, _, _ = self._extract_batch(batch)
            space_dim = inputs.shape[-1] - 1  # Subtract 1 for time dimension
            self._prepare_models(self.sample_batch_fn, output_dim_override=space_dim)
            print(f"[GCovAGMOnlineGuidance] Using gradient mode: output_dim={space_dim}")
        else:
            # For other loss types (ground_truth, mse_simple), output scalar energy
            self._prepare_models(self.sample_batch_fn, output_dim_override=1)
            print(f"[GCovAGMOnlineGuidance] Using scalar mode: output_dim=1")
        
        # Initialize last layer to zero for warm start (residual = 0 initially)
        if self.learned_guidance_model and isinstance(self.learned_guidance_model, nn.Sequential):
            nn.init.zeros_(self.learned_guidance_model[-1].weight)
            nn.init.zeros_(self.learned_guidance_model[-1].bias)
    
    def _compute_trajectory_conflict_mask(self, xs_stacked, num_steps, batch_size):
        """Compute conflict mask for trajectory points."""
        if self.distribution is None:
            return None, 1.0
        
        with torch.no_grad():
            # CRITICAL FIX: Use dynamic dimension instead of hardcoded 2
            space_dim = xs_stacked.shape[-1]
            conflict = self._compute_conflict_score(xs_stacked.reshape(-1, space_dim), self.targets)
        
        if conflict is None:
            return None, 1.0
        
        conflict = conflict.view(num_steps, batch_size)
        threshold, _ = self._get_conflict_threshold_and_temperature()
        conflict_mask, active_ratio = self._compute_conflict_mask(conflict, threshold)
        return conflict_mask, active_ratio

    def _compute_trajectory_weights_ground_truth(self, x1, batch_size):
        """Compute target label distribution p*(i) from terminal reward."""
        with torch.no_grad():
            # 1) Construct energy from distribution's J, then convert to reward
            total_J = torch.zeros(batch_size, device=self.device)
            for clf, target, scale in zip(self.classifiers, self.targets, self.scales):
                # get_J returns "energy" J, smaller is better
                J_i = self.distribution.get_J(x1, classifier=clf, label=target)  # [B]
                total_J = total_J + float(scale) * J_i
            # reward = -energy: larger is better
            r1 = -total_J  # [B]

            # 3) Softmax to get label distribution p*(i) within batch
            beta = getattr(self.cfg, "energy_temperature", 1.0)
            logits = beta * r1                      # p*(i) ∝ exp(β r1(i))
            logits = logits - logits.max()          # Numerical stability
            w_eff = torch.softmax(logits, dim=0)    # [B]
            w_eff = w_eff.unsqueeze(-1)             # [B, 1] for convenient broadcasting

        # Return (label distribution, reward), reward used for logging
        return w_eff, r1

    def _compute_online_loss_ground_truth(self, xs_stacked, ts_stacked, w_eff, conflict_mask, num_steps, batch_size):
        """Compute loss for online training."""
        # CRITICAL FIX: Use dynamic dimension instead of hardcoded 2
        space_dim = xs_stacked.shape[-1]
        inp = torch.cat([xs_stacked.reshape(-1, space_dim), ts_stacked.reshape(-1, 1)], dim=-1)
        
        # Assume output is Energy E(x,t)
        energy_pred = self.learned_guidance_model(inp).view(num_steps, batch_size) 
        
        tau = getattr(self.cfg, "energy_temperature", 1.0)
        
        # Model outputs reward-like values (larger is better, consistent with 方案A: g = +∇r)
        # p_phi ∝ exp(reward / tau), so no negative sign needed
        log_prob = torch.log_softmax(energy_pred / tau, dim=1) 
        
        # Build Target
        w_mat = w_eff.view(1, batch_size).expand(num_steps, batch_size)
        
        # Ensure Target does not participate in gradient computation
        # Ensure mask exists, otherwise use w_mat directly
        mask = conflict_mask.to(w_mat.device) if conflict_mask is not None else torch.ones_like(w_mat)
        loss_weights = (w_mat * mask).detach() 
        
        # Normalize Target, handle zero denominator case
        # Note: w_eff must be non-negative
        weight_sum = loss_weights.sum(dim=1, keepdim=True)
        target_dist = loss_weights / (weight_sum + 1e-8)

        # Compute Loss
        # If sum is 0 at some timestep (all samples masked or invalid), loss should be 0
        # This implementation automatically handles 0, since target_dist is 0
        loss = -torch.sum(target_dist * log_prob, dim=1)
        
        # Average only over valid timesteps (weight_sum > 0), or directly global average
        # If there are many all-zero steps, direct mean() will lower loss value, suggest adding mask or keep as-is
        return loss.mean()

    def _compute_online_loss_mse_simple(
        self,
        xs_stacked,      # (T, B, space_dim)
        ts_stacked,      # (T, B, 1)
        r1,              # (B,) or (B,1), terminal reward/label for each trajectory
        conflict_mask,   # (T, B)
        num_steps,
        batch_size,
    ):
        """
        Simplified MSE version of online loss:
            For each (t, i), regress pred(t, i) to r1[i],
            weighted by conflict_mask only (no trajectory-level weights).
        """
        # CRITICAL FIX: Use dynamic dimension instead of hardcoded 2
        space_dim = xs_stacked.shape[-1]
        inp = torch.cat([xs_stacked.reshape(-1, space_dim), ts_stacked.reshape(-1, 1)], dim=-1)
        pred = self.learned_guidance_model(inp).view(num_steps, batch_size)

        # 1. Handle target shape
        if r1.dim() == 1:
            r1_ = r1.view(1, batch_size)
        else:
            r1_ = r1.view(1, batch_size)
            
        # 2. [Key] Broadcast and Detach to prevent gradient leakage to Reward Model
        target = r1_.expand(num_steps, batch_size).detach()

        # 3. Handle Mask
        if conflict_mask is not None:
            weight = conflict_mask.to(pred.device).float()
        else:
            weight = torch.ones_like(pred)

        # 4. Compute MSE
        # (pred - target)^2
        loss_unreduced = F.mse_loss(pred, target, reduction='none') 
        
        # 5. Weighted average
        loss = (loss_unreduced * weight).sum() / (weight.sum() + 1e-8)
        
        return loss
    
    def _compute_online_loss_gradient(
        self,
        xs_stacked,      # (T, B, space_dim)
        ts_stacked,      # (T, B, 1)
        r1,              # (B,) - 梯度匹配模式下其实不需要这个终端值 r1
        conflict_mask,   # (T, B)
        num_steps,
        batch_size,
        x1=None,         # (B, space_dim) - terminal point x1, if None will try to extract from xs_stacked
    ):
        """
        Gradient regression version of online loss.
        Target: Negative Gradient of the ENERGY function sum_i scale_i * J_i(x_t).
            
        We train the model to predict -∇J (direction of decreasing energy).
        """
        space_dim = xs_stacked.shape[-1]
        
        # Flatten: (T*B, space_dim)
        xs_flat = xs_stacked.reshape(-1, space_dim)
        ts_flat = ts_stacked.reshape(-1, 1)
        
        # 1. Compute the TRUE Target Gradient (∇J - energy gradient)
        # We compute the gradient of the energy J (higher is worse).
        xs_flat_req = xs_flat.detach().requires_grad_(True)
        
        x1 = xs_stacked[-1]  # (B, space_dim) - WARNING: This may not be the true x1
        
        with torch.enable_grad():
            # Robustness check
            if self.distribution is None:
                 raise ValueError("Gradient regression requires self.distribution to be set.")  
            
            # Compute gradient based on r1: r1 = -total_J(x1), so grad(r1) = -grad(total_J(x1))
            # Since r1 is passed in (computed with no_grad), we recompute the same quantity with gradient tracking
            x1_req = x1.detach().requires_grad_(True)
            total_J_x1 = torch.zeros(batch_size, device=self.device)
            for clf, target, scale in zip(self.classifiers, self.targets, self.scales):
                J_i = self.distribution.get_J(x1_req, classifier=clf, label=target)
                total_J_x1 = total_J_x1 + float(scale) * J_i
            # This should match r1: r1_computed = -total_J_x1 ≈ r1 (for validation)
            
            # Compute gradient of r1 w.r.t. x1: grad(r1) = grad(-total_J(x1)) = -grad(total_J(x1))
            # This is the gradient direction (pointing towards higher reward/lower energy)
            grad_r1 = torch.autograd.grad(
                (-total_J_x1).sum(), x1_req,  # grad of r1 = grad(-total_J)
                create_graph=False, retain_graph=False
            )[0]  # (B, space_dim)
        
        # The learned gradient should point towards higher reward (lower energy)
        # target_grad = grad(r1) = -grad(total_J)
        # We need to broadcast to all time steps: (T*B, space_dim)
        target_grad_terminal = grad_r1.detach()  # (B, space_dim)
        # For intermediate points, we use the same gradient (or can be extended later)
        # Expand to (T, B, space_dim) then reshape to (T*B, space_dim)
        target_grad = target_grad_terminal.unsqueeze(0).expand(num_steps, -1, -1).reshape(-1, space_dim)  # (T*B, space_dim)
        
        # 2. Model Prediction
        # Flatten input: (x, t)
        inp = torch.cat([xs_flat.detach(), ts_flat], dim=-1)
        model_output = self.learned_guidance_model(inp) 
        
        print(f"[DEBUG _compute_online_loss_gradient] model_output.shape: {model_output.shape}")
        print(f"[DEBUG _compute_online_loss_gradient] space_dim: {space_dim}")
        print(f"[DEBUG _compute_online_loss_gradient] target_grad.shape: {target_grad.shape}")
        
        # 3. Handle Model Output Type
        if model_output.shape[-1] == space_dim:
            # Case A: Model outputs vector field directly (∇J)
            print(f"[DEBUG] Using Case A: Direct vector field output")
            pred_grad = model_output
        else:
            # Case B: Scalar potential (learning Reward function ≈ -J)
            print(f"[DEBUG] Using Case B: Scalar potential -> gradient (FALLBACK!)")
            print(f"[WARNING] This means model is NOT directly learning gradients!")
            xs_for_grad = xs_flat.detach().requires_grad_(True)
            inp_for_grad = torch.cat([xs_for_grad, ts_flat], dim=-1)
            energy_output = self.learned_guidance_model(inp_for_grad).squeeze(-1)
            pred_grad = torch.autograd.grad(
                energy_output.sum(), xs_for_grad,
                create_graph=True, retain_graph=True
            )[0]
        
        # 4. Compute Loss (MSE on gradients)
        # Reshape to (T, B, D)
        pred_grad_reshaped = pred_grad.view(num_steps, batch_size, space_dim)
        target_grad_reshaped = target_grad.view(num_steps, batch_size, space_dim)
        
        # Loss per point
        loss_per_point = ((pred_grad_reshaped - target_grad_reshaped) ** 2).mean(dim=-1)
        
        # Apply Mask
        if conflict_mask is not None:
            weight = conflict_mask.to(pred_grad.device).float()
        else:
            weight = torch.ones(num_steps, batch_size, device=pred_grad.device)
        
        loss = (loss_per_point * weight).sum() / (weight.sum() + 1e-8)
        
        return loss
    
    def _compute_online_loss(self, xs_stacked, ts_stacked, w_eff, conflict_mask, num_steps, batch_size):
        """
        xs_stacked: (T, B, 2)
        ts_stacked: (T, B, 1)
        w_eff:      (B, 1)   # Terminal reward soft label
        conflict_mask: (T, B) or None
        """
        # 1) Flatten inputs to get energy e_phi(x_t, t)
        # CRITICAL FIX: Use dynamic dimension instead of hardcoded 2
        space_dim = xs_stacked.shape[-1]
        inp = torch.cat([xs_stacked.reshape(-1, space_dim), ts_stacked.reshape(-1, 1)], dim=-1)
        pred_energy = self.learned_guidance_model(inp).view(num_steps, batch_size)  # e_phi(t, i)

        # 2) Model distribution: p_phi(i | t) ∝ exp( reward_phi(i,t) / tau )
        # Model outputs reward-like values (larger is better, consistent with 方案A: g = +∇r)
        tau = getattr(self.cfg, "energy_temperature", 1.0)
        logits = pred_energy / tau                        # No negative sign (reward体系)
        log_prob = torch.log_softmax(logits, dim=1)       # For each timestep t, normalize along batch dimension

        # 3) Target distribution: construct p*(i | t) from w_eff + conflict_mask
        w_mat = w_eff.view(1, batch_size).expand(num_steps, batch_size)  # [T,B]
        if conflict_mask is not None:
            loss_weights = w_mat * conflict_mask.to(w_mat.device)
        else:
            loss_weights = w_mat

        target_dist = loss_weights / (loss_weights.sum(dim=1, keepdim=True) + 1e-8)  # Normalize separately for each t

        # 4) Cross-Entropy: CE(p* || p_phi)
        loss = -(target_dist * log_prob).sum(dim=1).mean()
        return loss
    
    def _log_online_training_progress_ground_truth(self, step, total_steps, loss, active_ratio, w_eff, r1):
        with torch.no_grad():
            w_eff_flat = w_eff.flatten()
            # Expect corr(w_eff, r1) to be positive: higher reward => higher weight
            if len(w_eff_flat) > 1:
                w_mean, r_mean = w_eff_flat.mean(), r1.mean()
                w_centered = w_eff_flat - w_mean
                r_centered = r1 - r_mean
                numerator = (w_centered * r_centered).sum()
                denominator = torch.sqrt(
                    (w_centered.pow(2).sum() * r_centered.pow(2).sum()) + 1e-8
                )
                corr = (numerator / denominator).item() if denominator > 1e-8 else 0.0
            else:
                corr = 0.0

        print(
            f"Online-Step {step}/{total_steps} Loss: {loss.item():.6f} | "
            f"Active Conflict: {active_ratio:.1%} | "
            f"corr(w_eff, reward): {corr:.3f}"
        )
    
    def _log_online_training_progress(self, step, total_steps, loss, active_ratio, w_eff, base_log_prob):
        """Log training progress for online guidance."""
        with torch.no_grad():
            w_eff_flat = w_eff.flatten()
            neg_r1 = -base_log_prob
            
            if len(w_eff_flat) > 1:
                w_mean, r_mean = w_eff_flat.mean(), neg_r1.mean()
                w_centered = w_eff_flat - w_mean
                r_centered = neg_r1 - r_mean
                numerator = (w_centered * r_centered).sum()
                denominator = torch.sqrt((w_centered.pow(2).sum() * r_centered.pow(2).sum()) + 1e-8)
                corr = (numerator / denominator).item() if denominator > 1e-8 else 0.0
            else:
                corr = 0.0
        
        print(f"Online-Step {step}/{total_steps} Loss: {loss.item():.6f} | "
              f"Active Conflict: {active_ratio:.1%} | "
              f"corr(w_eff, -r1): {corr:.3f}")

    def train_model(self):
        steps = getattr(self.cfg, "guidance_train_steps", 1000)
        batch_size = getattr(self.cfg, "guidance_batch_size", 512)
        lr = getattr(self.cfg, "guidance_lr", 1e-3)
        log_interval = 100
        
        # ODE solver params
        num_steps = 20
        dt = 1.0 / num_steps
        
        optimizer = torch.optim.Adam(self.learned_guidance_model.parameters(), lr=lr)
        self.learned_guidance_model.train()
        
        print(f"[GCovAGMOnlineGuidance] Training Online Residual model ({steps} steps)")
        
        for i in range(steps):
            # 1. Generate Online Trajectories using CURRENT Guidance
            # Initially residual is ~0, so it's Base Guidance.
            # As we train, it becomes Base + Learned.
            
            # CRITICAL FIX: Use dynamic dimension instead of hardcoded 2
            # Get space dimension from sample_batch_fn
            sample_batch = self.sample_batch_fn(1, self.device)
            sample_inputs, _, _ = self._extract_batch(sample_batch)
            space_dim = sample_inputs.shape[-1] - 1  # Subtract 1 for time dimension
            x = torch.randn(batch_size, space_dim, device=self.device)
            
            traj_xs = []
            traj_ts = []
            
            curr_x = x
            for step in range(num_steps):
                t_val = step * dt
                t_tensor = torch.full((batch_size, 1), t_val, device=self.device)
                
                # Store current state
                traj_xs.append(curr_x.clone())
                traj_ts.append(t_tensor)
                
                # Compute velocity = v_uncond + g_total
                # CRITICAL FIX: Use pure inference guidance function for rollout
                # No_grad avoids building autograd graphs during trajectory generation
                with torch.no_grad():
                    v_uncond = self.flow_model(curr_x, t_tensor)
                    # Use rollout-specific guidance function (no autograd graph)
                    g_total = self._compute_guidance_for_rollout(curr_x, t_tensor, v_uncond)
                    d_x = v_uncond + g_total
                
                # Euler step
                curr_x = curr_x + d_x * dt
            
            # Final x is x1
            x1 = curr_x
            
            # 2. Compute Conflict Mask Early (before expensive computations)
            xs_stacked = torch.stack(traj_xs, dim=0)  # (T, B, 2)
            ts_stacked = torch.stack(traj_ts, dim=0)  # (T, B, 1)
            # 注意：
            # xs_stacked[-1] = traj_xs 的最后一个元素 = x(t=0.95)（循环中最后存储的点）
            # x1 = 循环结束后的 curr_x = x(t=1.00)（终端点）
            # 如果需要在 xs_stacked 中包含 x1，有两种方案：
            # （1）在循环结束后将 x1 追加到 traj_xs
            # （2）将 x1 作为单独参数传递（已实现）
            
            conflict_mask, active_ratio = self._compute_trajectory_conflict_mask(
                xs_stacked, num_steps, batch_size
            )
            if conflict_mask is None or active_ratio < 1e-6:
                continue
            

            # # 3. Compute weights and regression target
            w_eff, r1 = self._compute_trajectory_weights_ground_truth(x1, batch_size)

            # Select loss function based on hyperparameter
            online_loss_type = getattr(self.cfg, "online_loss_type", "ground_truth")
            if online_loss_type == "mse_simple":  # MSE loss for terminal points
                loss = self._compute_online_loss_mse_simple(
                    xs_stacked, ts_stacked, r1, conflict_mask,
                    num_steps, batch_size
                )
            elif online_loss_type == "gradient":  # Gradient regression: pred ≈ ∇r
                loss = self._compute_online_loss_gradient(
                    xs_stacked, ts_stacked, r1, conflict_mask,
                    num_steps, batch_size, x1=x1
                )
            else:  # default: "ground_truth" cross entropy loss
                loss = self._compute_online_loss_ground_truth(
                    xs_stacked, ts_stacked, w_eff, conflict_mask,
                    num_steps, batch_size
                )

            if log_interval and (i+1) % log_interval == 0:
                self._log_online_training_progress_ground_truth(i+1, steps, loss, active_ratio, w_eff, r1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.learned_guidance_model.eval()
        self.save_learned_guidance()