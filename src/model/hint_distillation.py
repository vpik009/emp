import torch.nn as nn
import torch.nn.functional as F
import torch
from torchmetrics import MetricCollection
import pytorch_lightning as pl
from pathlib import Path
import numpy as np

from src.utils.submission_av2 import SubmissionAv2
from src.metrics import MR, brierMinFDE, minADE, minFDE
from src.utils.optim import WarmupCosLR
from src.model.emp import EMP
from src.model.trainer_forecast import Trainer


class EMPStage0(nn.Module):
    def __init__(self, emp_model):
        super().__init__()
        self.h_proj = emp_model.h_proj
        self.h_embed = emp_model.h_embed
        self.actor_type_embed = emp_model.actor_type_embed
        self.history_steps = emp_model.history_steps
        self.embed_dim = emp_model.embed_dim
        
    def forward(self, data):
        hist_padding_mask = data["x_padding_mask"][:, :, :self.history_steps]
        
        hist_key_padding_mask = data["x_key_padding_mask"]

        hist_feat = torch.cat(
            [
                data["x"],
                data["x_velocity_diff"][..., None],
                (~hist_padding_mask[..., None]).float(),
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(B * N)

        actor_feat = hist_feat[~hist_feat_key_padding]
        
        ts = torch.arange(self.history_steps).view(1, -1, 1).repeat(actor_feat.shape[0], 1, 1).to(actor_feat.device).float()
        actor_feat = torch.cat([actor_feat, ts], dim=-1)

        actor_feat = self.h_proj( actor_feat )
        kpm = hist_padding_mask.view(B*N, -1)[~hist_feat_key_padding]
        for blk in self.h_embed:
            actor_feat = blk(actor_feat, key_padding_mask=kpm)
        actor_feat = torch.max(actor_feat, axis=1).values
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )

        #actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        mask_indices = torch.nonzero(~hist_feat_key_padding, as_tuple=True)
        actor_feat_tmp.scatter_(0, mask_indices[0].unsqueeze(1).expand(-1, self.embed_dim), actor_feat)

        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])
        
        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        actor_feat += actor_type_embed
        
        return actor_feat
    
    
class EMPStage1(nn.Module):
    def __init__(self, emp_model):
        super().__init__()
        self.lane_embed = emp_model.lane_embed
        self.lane_type_embed = emp_model.lane_type_embed
        
    def forward(self, data):
        lane_padding_mask = data["lane_padding_mask"]

        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, (~lane_padding_mask[..., None]).float()], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)
        
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        lane_feat += lane_type_embed
        
        return lane_feat
    

class EMPStage2(nn.Module):
    def __init__(self, emp_model):
        super().__init__()
        self.pos_embed = emp_model.pos_embed
        self.blocks = emp_model.blocks
        self.norm = emp_model.norm
        self.history_steps = emp_model.history_steps

    def forward(self, data, actor_feat, lane_feat):
        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, self.history_steps-1], data["lane_angles"]], dim=1)    

        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)      
        pos_embed = self.pos_embed(pos_feat)

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat([data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1)           

        x_encoder = x_encoder + pos_embed

        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)
        
        return x_encoder
    
class EMPStage3(nn.Module):
    def __init__(self, emp_model):
        super().__init__()
        self.dense_predictor = emp_model.dense_predictor
        self.decoder = emp_model.decoder
        self.future_steps = emp_model.future_steps
        
    def forward(self, data, x_encoder):
        B, N = data["x"].shape[0], data["x"].shape[1]
        
        x_agent = x_encoder[:, 0] 
        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, -1, self.future_steps, 2)

        key_padding_mask = torch.cat([data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1)   
        y_hat, pi = self.decoder(x_agent, x_encoder, key_padding_mask, N)
        
        y_hat_eps = y_hat[:, :, -1]

        return {
            "y_hat": y_hat,
            "pi": pi,
            "y_hat_others": y_hat_others,
            "y_hat_eps": y_hat_eps,
            "x_agent": x_agent
        }
    
    
class DistillationTrainer(pl.LightningModule):
    def __init__(
        self,
        dim=128,
        historical_steps=50,
        future_steps=60,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs = 10,
        epochs = 60,
        weight_decay: float = 1e-4,
        decoder: str = "detr",
        student = None,
        with_hints = True,
    ) -> None:
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.warmup_epoch_schedule = [2, 2, 2, warmup_epochs]
        self.epochs = epochs
        # epochs_per_stage = self.epochs // 5
        self.epoch_schedule = [5, 5, 5, epochs - 15]
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.history_steps = historical_steps
        self.future_steps = future_steps
        self.submission_handler = SubmissionAv2()

        if student is not None:
            student_emp = student
        else:
            student_emp = EMP(
                embed_dim=dim,
                encoder_depth=encoder_depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=drop_path,
                decoder=decoder
            )  

        if pretrained_weights is not None:
            student_emp.load_from_checkpoint(pretrained_weights)
            
        teacher_path = "checkpoints/empd.ckpt"
        teacher_trainer = Trainer.load_from_checkpoint(teacher_path)
        teacher_emp = teacher_trainer.net
        
        for param in teacher_emp.parameters():
            param.requires_grad = False
            
        stages = [EMPStage0, EMPStage1, EMPStage2, EMPStage3]
        self.student = nn.ModuleList([stage(student_emp) for stage in stages])
        self.teacher = nn.ModuleList([stage(teacher_emp) for stage in stages])

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
                "brier-minFDE6": brierMinFDE(k=6)
            }
        )
        self.val_metrics = metrics.clone(prefix="val_")
        self.curr_ep = 0
        
        self.with_hints = with_hints
        if with_hints:
            print("Training with hints.")
            self.stage = -1
        else:
            print("Training without hints.")
            self.stage = 3
        return
    
    def forward(self, data):
        actor_feat = self.student[0](data)
        lane_feat = self.student[1](data)
        x_encoder = self.student[2](data, actor_feat, lane_feat)
        return self.student[3](data, x_encoder)
    
    def predict(self, data):
        with torch.no_grad():
            out = self(data)
        predictions, prob = self.submission_handler.format_data(
            data, out["y_hat"], out["pi"], inference=True
        ) # type: ignore
        return predictions, prob  
    
    def cal_loss(self, out, data, batch_idx=0):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        loss = 0
        B = y_hat.shape[0]
        B_range = range(B)

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(-1)

        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[B_range, best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)

        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
        loss += agent_reg_loss + agent_cls_loss
        
        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.history_steps:]
        others_reg_loss = F.smooth_l1_loss(y_hat_others[others_reg_mask], y_others[others_reg_mask])
        loss += others_reg_loss
    
        return {
            "loss": loss,
            "reg_loss": agent_reg_loss.item(),
            "cls_loss": agent_cls_loss.item(),
            "others_reg_loss": others_reg_loss.item(),
        }
        
    def distillation_loss(self, out_student, out_teacher, data):
        y_s, pi_s, y_s_others = out_student["y_hat"], out_student["pi"], out_student["y_hat_others"]
        y_t, y_t_others = out_teacher["y_hat"], out_teacher["y_hat_others"]

        loss = 0
        B = y_s.shape[0]
        B_range = range(B)

        l2_norm = torch.norm(y_s[..., :2] - y_t[..., :2], dim=-1).sum(-1)

        best_mode = torch.argmin(l2_norm, dim=-1)
        y_s_best = y_s[B_range, best_mode]
        y_t_best = y_t[B_range, best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_s_best[..., :2], y_t_best[..., :2])

        agent_cls_loss = F.cross_entropy(pi_s, best_mode.detach())
        loss += agent_reg_loss + agent_cls_loss
        
        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.history_steps:]
        others_reg_loss = F.smooth_l1_loss(y_s_others[others_reg_mask], y_t_others[others_reg_mask])
        loss += others_reg_loss
    
        return {
            "loss": loss,
            "reg_loss": agent_reg_loss.item(),
            "cls_loss": agent_cls_loss.item(),
            "others_reg_loss": others_reg_loss.item(),
        }
   
    def hint_loss(self, out_student, out_teacher):
        student_mean, teacher_mean = torch.mean(out_student, dim=-1), torch.mean(out_teacher, dim=-1)
        loss = F.mse_loss(student_mean, teacher_mean)
        return loss
        
    def on_train_epoch_start(self):
        if self.with_hints and (self.current_epoch == 0 or self.current_epoch >= sum(self.epoch_schedule[:self.stage+1])):
            self.stage += 1
            print(f"Now training Stage {self.stage}")
            self.trainer.strategy.setup_optimizers(self.trainer)
            for i, stage in enumerate(self.student):
                for param in stage.parameters():
                    param.requires_grad = (i <= self.stage)
                        
    def training_step(self, data, batch_idx):
        if self.stage == 0:
            out_0_student = self.student[0](data)
            out_0_teacher = self.teacher[0](data)
            hint_loss_0 = self.hint_loss(out_0_student, out_0_teacher)
            self.log("loss_0", hint_loss_0, on_step=True, on_epoch=True, prog_bar=False)
            return hint_loss_0
        
        if self.stage == 1:
            out_1_student = self.student[1](data)
            out_1_teacher = self.teacher[1](data)
            hint_loss_1 = self.hint_loss(out_1_student, out_1_teacher)
            self.log("loss_1", hint_loss_1, on_step=True, on_epoch=True, prog_bar=False)
            return hint_loss_1
        
        out_0_student = self.student[0](data)
        out_0_teacher = self.teacher[0](data)
        out_1_student = self.student[1](data)
        out_1_teacher = self.teacher[1](data)
        out_2_student = self.student[2](data, out_0_student, out_1_student)
        out_2_teacher = self.teacher[2](data, out_0_teacher, out_1_teacher)
        if self.stage == 2:
            hint_loss_2 = self.hint_loss(out_2_student, out_2_teacher)
            self.log("loss_2", hint_loss_2, on_step=True, on_epoch=True, prog_bar=False)
            return hint_loss_2
        
        out_3_student = self.student[3](data, out_2_student)
        out_3_teacher = self.teacher[3](data, out_2_teacher)
        data_losses = self.cal_loss(out_3_student, data)
        teacher_losses = self.distillation_loss(out_3_student, out_3_teacher, data)
        final_loss = data_losses["loss"] + teacher_losses["loss"]
        for k, v in data_losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return final_loss
    
    def validation_step(self, data, batch_idx):
        out = self(data)

        losses = self.cal_loss(out, data, -1)
        metrics = self.val_metrics(out, data["y"][:, 0])

        self.log(
            "val/reg_loss",
            losses["reg_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        for k in self.val_scores.keys(): self.val_scores[k].append(metrics[k].item())

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        
    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)

    def test_step(self, data, batch_idx) -> None:
        out = self(data)
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def on_validation_start(self) -> None:
        self.val_scores = {"val_MR": [], "val_minADE1": [], "val_minADE6": [], "val_minFDE1": [], "val_minFDE6": [], "val_brier-minFDE6": []}

    def on_validation_end(self) -> None:      
        print( " & ".join( ["{:5.3f}".format(np.mean(self.val_scores[k])) for k in ["val_MR", "val_minADE6", "val_minFDE6", "val_brier-minFDE6"]] ) )
        self.curr_ep += 1
        
    def configure_optimizers(self):
        """Make a new optimizer with scheduler optimizing the passed list."""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
            nn.GRUCell
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
            nn.Parameter
        )
        
        param_dict = {}
        
        for i in range(self.stage + 1):
            for module_name, module in self.student[i].named_modules():
                for param_name, param in module.named_parameters():
                    full_param_name = (
                        "%s.%s" % (module_name, param_name) if module_name else param_name
                    )
                    if "bias" in param_name:
                        no_decay.add(full_param_name)
                    elif "weight" in param_name:
                        if isinstance(module, whitelist_weight_modules):
                            decay.add(full_param_name)
                        elif isinstance(module, blacklist_weight_modules):
                            no_decay.add(full_param_name)
                    elif not ("weight" in param_name or "bias" in param_name):
                        no_decay.add(full_param_name)
                        
            for param_name, param in self.student[i].named_parameters():
                param_dict[param_name] = param
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0
            assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.stage == 3:
            optimizer = torch.optim.AdamW(
                optim_groups, lr=self.lr, weight_decay=self.weight_decay
            )
            print(f"Initializing a new scheduler running for {self.epoch_schedule[self.stage]} epochs with {self.warmup_epoch_schedule[self.stage]} warmup epochs")
            scheduler = WarmupCosLR(
                optimizer=optimizer,
                lr=self.lr,
                min_lr=1e-6,
                warmup_epochs=self.warmup_epoch_schedule[self.stage],
                epochs=self.epoch_schedule[self.stage],
            )
            return [optimizer], [scheduler]
        else:
            optimizer = torch.optim.AdamW(
                optim_groups, lr=0.005, weight_decay=self.weight_decay
            )
            return optimizer
    
    def summarize_submodules(self):
        summary = []
        for name, submodule in self.named_children():
            trainable = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
            untrainable = sum(p.numel() for p in submodule.parameters() if not p.requires_grad)
            summary.append((name, trainable, untrainable))
        return summary
        