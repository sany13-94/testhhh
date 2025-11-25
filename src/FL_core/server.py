import torch
import wandb
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import sys
import multiprocessing as mp
import random
import os
import pandas as pd
from .client import Client
from .client_selection.config import *
from pathlib import Path
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
 # PATCH: evaluate global model on each client's validation set and average
    
class FeatureExtractor(nn.Module):
    """
    Wrap a classifier and return penultimate features.
    Works for ResNet-like models (conv trunk -> avgpool -> fc).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            # up to avgpool (exclude the final fc)
            self.backbone = nn.Sequential(*(list(model.children())[:-1]))
            self.feat_dim = model.fc.in_features
        else:
            # You can adapt for other models if needed
            raise RuntimeError("FeatureExtractor: please adapt forward() for your model type.")

    def forward(self, x):
        z = self.backbone(x)    # (B, C, 1, 1)
        z = torch.flatten(z, 1) # (B, C)
        return z


@torch.no_grad()
def compute_macro_prototype_from_loader(dataloader, feature_net, device, max_batches=3):
    """
    Compute a macro prototype: average of per-class mean features.
    Uses up to max_batches to keep cost low.
    """
    feature_net.eval()
    class_sums, class_counts = {}, {}
    n_batches = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = feature_net(x)  # [B, D]
        for c in y.unique().tolist():
            mask = (y == c)
            if mask.any():
                cls_vec = z[mask].mean(dim=0)  # mean feature for class c in this batch
                class_sums[c]  = class_sums.get(c, 0) + cls_vec
                class_counts[c] = class_counts.get(c, 0) + 1
        n_batches += 1
        if n_batches >= max_batches:
            break

    if not class_sums:
        return None  # no data?
    per_class = []
    for c, s in class_sums.items():
        per_class.append((s / class_counts[c]).unsqueeze(0))
    macro = torch.cat(per_class, dim=0).mean(dim=0)  # [D]
    return macro.detach().cpu()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)

class Server(object):
    def __init__(self, data, init_model, args, selection, fed_algo, files):
        """
        Server to execute
        ---
        Args
            data: dataset for FL
            init_model: initial global model
            args: arguments for overall FL training
            selection: client selection method
            fed_algo: FL algorithm for aggregation at server
            results: results for recording
        """
        
        self.train_data = data['train']['data']
        self.train_sizes = data['train']['data_sizes']
        self.test_data = data['test']['data']
        self.test_sizes = data['test']['data_sizes']
        self.test_clients = data['test']['data_sizes'].keys()
        self.args = args
   
        meta = data.get('meta', {}) if isinstance(data, dict) else {}
        dom_map = meta.get('domain_assignment', None)

        # also accept object-style datasets that may attach the array on the object
        if dom_map is None and hasattr(data, "domain_assignment"):
          dom_map = getattr(data, "domain_assignment")


        self.domain_assignment = dom_map
        self.domain_map = dom_map   # all our plotting/helpers use self.domain_map
        print(f"[WARN] domain_assignment length={len(self.domain_map)} "
                  f"!=Heatmaps may be misaligned. {self.domain_map}")
        # basic sanity checks (won't crash training, just warn)
        try:
          if self.domain_map is not None:
            K = self.total_num_client
            if len(self.domain_map) != K:
              print(f"[WARN] domain_assignment length={len(self.domain_map)} "
                  f"!= total_num_client={K}. Heatmaps may be misaligned.")
            uniq = sorted(set(int(x) for x in self.domain_map))
            print(f"[INFO] domain IDs seen: {uniq}")
        except Exception as e:
          print(f"[WARN] could not validate domain map: {e}")


        # ---- PARTICIPATION TRACKING ----

        # --- Round timing + result paths ---
        self.round_logs = []          # list of dicts, one per round
        self.cum_time = 0.0           # cumulative wall-clock time (seconds)

        self.results_dir = Path("/kaggle/working/testhhh/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.round_csv = self.results_dir / f"round_log_pov.csv"
        self.acc_time_png = self.results_dir / f"acc_vs_time_pov.png"
        
        self.round_csv_path = self.results_dir / f"round_log_{self.args.method}.csv"
        # Cumulative time across rounds (for resumed runs we may overwrite below)
        self.cum_time = 0.0
        self.round_logs = []  # in-memory copy (optional, for later plotting)
        # We used the dataset adapter to store *validation* sets in the 'test' slot on purpose.
        self.val_data = data['test']['data']           # dict: cid -> Dataset (client-specific validation)
        self.num_clients = len(self.train_data)

     

        self.device = args.device
        
        self.global_model = init_model
        self.selection_method = selection
        self.federated_method = fed_algo
        self.files = files

        self.nCPU = mp.cpu_count() // 2 if args.nCPU is None else args.nCPU

        self.total_num_client = args.total_num_client
        self.num_clients_per_round = args.num_clients_per_round
        self.num_available = args.num_available
        if self.num_available is not None:
            random.seed(args.seed)

        self.total_round = args.num_round
        self.save_results = not args.no_save_results
        self.save_probs = args.save_probs

        if self.save_probs:
            num_local_data = np.array([self.train_sizes[idx] for idx in range(args.total_num_client)])
            num_local_data.tofile(files['num_samples'], sep=',')
            files['num_samples'].close()
            del files['num_samples']

        self.test_on_training_data = False
        # Initialize participation counters
        self.participation_counts = np.zeros(self.total_num_client, dtype=int)
        self.participation_history = []  # (round_idx, [client_ids])

        # inside Server.__init__
        self.simulate_stragglers = getattr(args, "simulate_stragglers", "0,1")
        print(f'stragglers : {self.simulate_stragglers}')
        self.delay_base_sec = getattr(args, "delay_base_sec", 10.0)
        self.delay_jitter_sec = getattr(args, "delay_jitter_sec", 3.0)
        self.delay_prob = getattr(args, "delay_prob", 1.0)
        ## INITIALIZE
        # initialize the training status of each client
        self._init_clients(init_model)

        # after: self._init_clients(init_model)
        self.trainer = self.client_list[0].trainer   # <-- use any client's trainer for eval
  
        # map client → domain (from your PathMNIST adapter); list/array of ints

        # prototype logging
        self.proto_history = []            # list per round: {"round": r, "assignments": {cid: dom}, "sims": {cid: {dom: sim}}}
        self.domain_centroids = None       # dict dom_id -> np.array vector
        self.proto_max_batches = getattr(self.args, "proto_max_batches", 3)  # optional CLI arg

        # initialize the client selection method
        if self.args.method in NEED_SETUP_METHOD:
            self.selection_method.setup(self.train_sizes)

        if self.args.method in LOSS_THRESHOLD:
            self.ltr = 0.0

    def save_checkpoint(self, round_completed: int):
          """
          Save a checkpoint with global model weights and round index.
          This is lightweight (only model + round).
          """
          ckpt_dir = getattr(self.args, "ckpt_dir", "/kaggle/working/testhhh/checkpoints")
          os.makedirs(ckpt_dir, exist_ok=True)

          filename = f"pow_round_{round_completed}.pt"
          ckpt_path = os.path.join(ckpt_dir, filename)

          payload = {
            "round": round_completed,
            "model_state_dict": self.global_model.state_dict(),
            # optional: dump args dict for reference
             "cum_time_sec": float(self.cum_time),
            "args": dict(vars(self.args)),
        }

          torch.save(payload, ckpt_path)
          print(f"[Checkpoint] Saved round {round_completed} -> {ckpt_path}")


    def save_participation_report(self, title: str = "Pow-d"):
        """
        Saves a bar chart and a CSV of client participation counts.
        """
        colors = None
        # simple mapping 0/1/2/3 -> distinct colors
        cmap = {0: "#4e79a7", 1: "#f28e2b", 2: "#e15759", 3: "#76b7b2"}
        out_dir = Path("/kaggle/working/testhhh/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / f"participation_{title.replace(' ', '_')}.png"
        csv_path = out_dir / f"participation_{title.replace(' ', '_')}.csv"

        counts = self.participation_counts.astype(int)
        n_clients = len(counts)
        participated = (counts > 0).sum()
        avg = counts.mean()
        std = counts.std()

        # ---- CSV (round-by-round and totals) ----
        df_total = pd.DataFrame({
        "client_id": list(range(n_clients)),
        "participations": counts
        })
        # (optional) history long-form
        rows = []
        for r, ids in self.participation_history:
          for cid in ids:
            rows.append({"round": r, "client_id": cid})
        df_hist = pd.DataFrame(rows)

        with pd.ExcelWriter(out_dir / f"participation_{title.replace(' ', '_')}.xlsx") as w:
          df_total.to_excel(w, index=False, sheet_name="totals")
          if not df_hist.empty:
            df_hist.to_excel(w, index=False, sheet_name="history")

        # ---- Plot ----
        plt.figure(figsize=(16, 5))
        x = np.arange(n_clients)
        plt.bar(x, counts)
        plt.title(f"Client Participation Distribution - {title}", fontsize=16, pad=12)
        plt.xlabel("Client ID")
        plt.ylabel("Number of Participations")
        plt.xticks(x, [f"Client {i}" for i in x], rotation=30, ha="right")

        # summary box
        text = (
        f"Total Clients: {n_clients}\n"
        f"Participated: {participated} ({participated/n_clients*100:.1f}%)\n"
        f"Avg Participation: {avg:.2f} ± {std:.2f}"
    )
        plt.gca().text(
        0.98, 0.97, text,
        transform=plt.gca().transAxes,
        fontsize=11,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f8edd1", ec="#b9a16b", alpha=0.9)
    )
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

        # also save CSV for quick diffs
        df_total.to_csv(csv_path, index=False)

        print(f"[Participation] saved: {png_path}")

    def _compute_global_reference_prototypes(self, selected_client_ids, batch_size=None, max_batches=None):
      """
      Compute global reference prototypes by averaging all selected clients' prototypes.
      """
      if batch_size is None: 
        batch_size = getattr(self.args, "batch_size", 64)
      if max_batches is None: 
        max_batches = self.proto_max_batches

      all_prototypes = []
    
      for cid in selected_client_ids:
        client = self.client_list[cid]
        proto = client.proto_from_validation(
            self.global_model, 
            self.device,
            batch_size=batch_size, 
            max_batches=max_batches
        )
        if proto is not None:
            all_prototypes.append(proto)
    
      if not all_prototypes:
        return None
    
      # Average all prototypes
      reference_proto = np.stack(all_prototypes, axis=0).mean(axis=0)
      return reference_proto


    def _calculate_prototype_distance_score(self, client_proto, reference_proto):
      """
      Calculate Euclidean distance between client and reference prototypes.
      """
      if client_proto is None or reference_proto is None:
        return 0.0
    
      distance = np.linalg.norm(client_proto - reference_proto)
      return float(distance)


    def _log_selected_client_prototype_scores(self, round_idx, selected_client_ids):
      """
      For selected clients, compute prototypes and calculate distance scores.
      REPLACES: _log_selected_client_prototypes_from_valid()
      """
      batch_size = getattr(self.args, "batch_size", 64)
      max_batches = self.proto_max_batches
    
      # Compute global reference
      reference_proto = self._compute_global_reference_prototypes(
        selected_client_ids, 
        batch_size=batch_size, 
        max_batches=max_batches
    )
    
      scores = {}
      prototypes = {}
    
      if reference_proto is None:
        for cid in selected_client_ids:
            scores[cid] = 0.0
            prototypes[cid] = None
        self.proto_history.append({
            "round": round_idx, 
            "scores": scores, 
            "prototypes": prototypes
        })
        return
    
      # Calculate distance score for each client
      for cid in selected_client_ids:
        client = self.client_list[cid]
        proto = client.proto_from_validation(
            self.global_model, 
            self.device,
            batch_size=batch_size,
            max_batches=max_batches
        )
        
        if proto is None:
            scores[cid] = 0.0
            prototypes[cid] = None
        else:
            distance_score = self._calculate_prototype_distance_score(proto, reference_proto)
            scores[cid] = distance_score
            prototypes[cid] = proto
    
      self.proto_history.append({
        "round": round_idx, 
        "scores": scores, 
        "prototypes": prototypes
    })
    
      if scores:
        avg_score = np.mean([s for s in scores.values() if s > 0])
        print(f"[Proto] Round {round_idx}: Avg distance score = {avg_score:.4f}")

    def evaluate_on_client_validation(self, clients=None):
          """
          Returns:
          avg_acc: float
          per_client: list of (cid, acc, loss)
          """
          if clients is None:
            clients = list(self.train_data.keys())

          per_client = []
          for cid in clients:
            val_ds = self.val_data[cid]  # Dataset (no DataLoader needed; trainer.test builds it)
            res = self.trainer.test(self.global_model, val_ds)  # {'loss': ..., 'acc': ...}
            per_client.append((cid, float(res['acc']), float(res['loss'])))

            # Optional Excel logging (guarded)
            try:
              from pathlib import Path
              import pandas as pd
              xls_path = self.save_results / f"client_valid_accuracy.xlsx"

              #xls_path = Path("/kaggle/working/testhhh/results/client_valid_accuracy.xlsx")
              # load records
              try:
                xls = pd.ExcelFile(xls_path)
                df_records = pd.read_excel(xls, sheet_name="records")
              except Exception:
                df_records = pd.DataFrame(columns=["round","client_id","domain_id","val_loss","val_acc"])

              # append one row
              did = self.domain_map[cid] if self.domain_map is not None else -1
              df_records = pd.concat([df_records, pd.DataFrame([{
                "round": getattr(self, "round_idx", -1),
                "client_id": cid,
                "domain_id": did,
                "val_loss": float(res['loss']),
                "val_acc": float(res['acc']),
            }])], ignore_index=True)

              # recompute averages sheet
              if not df_records.empty and "val_acc" in df_records:
                df_averages = (df_records.groupby("client_id", as_index=False)["val_acc"]
                               .agg(avg_val_acc="mean", count="size")
                               .sort_values("client_id"))
              else:
                df_averages = pd.DataFrame(columns=["client_id","avg_val_acc","count"])
            except Exception:
              pass  # logging is optional; don't block training

          avg_acc = sum(a for _, a, _ in per_client) / max(1, len(per_client))
          return avg_acc, per_client


    def _init_clients(self, init_model):
        """
        initialize clients' model
        ---
        Args
            init_model: initial given global model
        """
        self.client_list = []
        for client_idx in range(self.total_num_client):
            local_train_data = self.train_data[client_idx]
            local_test_data = self.test_data[client_idx] if client_idx in self.test_clients else np.array([])
            c = Client(client_idx, self.train_sizes[client_idx], local_train_data, local_test_data,
                       deepcopy(init_model), self.args)
            self.client_list.append(c)


    def save_proto_score_heatmap(self, title_suffix="Pow-d (prototype scores)"):

      out_dir = self.results_dir
      out_dir.mkdir(parents=True, exist_ok=True)
      png_path = out_dir / f"proto_score_heatmap_{self.args.method}.png"
      csv_path = out_dir / f"proto_scores_{self.args.method}.csv"

      if not self.proto_history:
        print("[ProtoScoreHeatmap] No prototype history.")
        return

      # Matrix: rows = clients (N), columns = rounds (R)
      R = max(h["round"] for h in self.proto_history) + 1
      N = self.total_num_client
      M = np.zeros((N, R), dtype=float)    # ← swapped dimensions (N x R)

      rows = []

      for h in self.proto_history:
        r = int(h["round"])
        scores = h.get("scores", {})

        for cid, score in scores.items():
            cid_int = int(cid)
            M[cid_int, r] = float(score)  # ← row = client, col = round

            if score > 0:
                rows.append({
                    "round": r,
                    "client_id": cid_int,
                    "prototype_distance": float(score)
                })

      pd.DataFrame(rows).to_csv(csv_path, index=False)
      print(f"[ProtoScoreHeatmap/CSV] saved: {csv_path}")

      fig, ax = plt.subplots(
        figsize=(min(20, max(10, R * 0.4)), min(12, max(6, N * 0.22)))
    )

      sns.heatmap(
        M,
        cmap="viridis",
        cbar_kws={"label": "Prototype distance (domain diversity)"},
        ax=ax,
        linewidths=0,
        rasterized=True
    )

      # Axis labels
      ax.set_xlabel("Round", fontsize=14, fontweight="bold")
      ax.set_ylabel("Client ID", fontsize=14, fontweight="bold")
      ax.set_title(
        f"Prototype-based selection pattern — {title_suffix}",
        fontsize=16, fontweight="bold"
    )

      # Y-axis = clients
      ax.set_yticks(np.arange(N) + 0.5)
      ax.set_yticklabels([f"Client {i}" for i in range(N)],
                       rotation=0, ha="right", fontsize=8)

      # X-axis = rounds
      if R > 50:
        tick_interval = R // 20
      elif R > 20:
        tick_interval = 5
      else:
        tick_interval = 1

      x_ticks = np.arange(0, R, tick_interval)
      ax.set_xticks(x_ticks + 0.5)
      ax.set_xticklabels([str(i) for i in x_ticks], fontsize=8)

      plt.tight_layout()
      plt.savefig(png_path, dpi=300, bbox_inches="tight")
      plt.close()

      print(f"[ProtoScoreHeatmap] saved: {png_path}")

      non_zero = M[M > 0]
      if len(non_zero) > 0:
        print("[ProtoScoreHeatmap] Score statistics:")
        print(f"  - Min: {non_zero.min():.4f}")
        print(f"  - Max: {non_zero.max():.4f}")
        print(f"  - Mean: {non_zero.mean():.4f}")
        print(f"  - Std: {non_zero.std():.4f}")


    def train(self,start_round: int = 0):
        """
        FL training
        """
        ## ITER COMMUNICATION ROUND
        for round_idx in range(start_round, self.total_round):

            print(f'\n>> ROUND {round_idx}')
            round_t0 = time.time()

            ## GET GLOBAL MODEL
            #self.global_model = self.trainer.get_model()
            self.global_model = self.global_model.to(self.device)
            ## ITER COMMUNICATION ROUND
            # set clients
            client_indices = [*range(self.total_num_client)]
            
            if self.num_available is not None:
                print(f'> available clients {self.num_available}/{len(client_indices)}')
                np.random.seed(self.args.seed + round_idx)
                client_indices = np.random.choice(client_indices, self.num_available, replace=False)
                self.save_selected_clients(round_idx, client_indices)
            
            
            # set client selection methods
            # initialize selection methods by setting given global model
            if self.args.method in NEED_INIT_METHOD:
                local_models = [self.client_list[idx].trainer.get_model() for idx in client_indices]
                self.selection_method.init(self.global_model, local_models)
                del local_models
            # candidate client selection before local training
            if self.args.method in CANDIDATE_SELECTION_METHOD:
                # np.random.seed((self.args.seed+1)*10000000 + round_idx)
                print(f'> candidate client selection {self.args.num_candidates}/{len(client_indices)}')
                client_indices = self.selection_method.select_candidates(client_indices, self.args.num_candidates)


            ## PRE-CLIENT SELECTION
            # client selection before local training (for efficiency)
            if self.args.method in PRE_SELECTION_METHOD:
                # np.random.seed((self.args.seed+1)*10000 + round_idx)
                print(f'> pre-client selection {self.num_clients_per_round}/{len(client_indices)}')
                client_indices = self.selection_method.select(self.num_clients_per_round, client_indices, None)
                print(f'selected clients: {sorted(client_indices)[:10]}')


            ## CLIENT UPDATE (TRAINING)
            local_losses, accuracy, local_metrics = self.train_clients(client_indices,round_idx)


            ## CLIENT SELECTION
            if self.args.method not in PRE_SELECTION_METHOD:
                print(f'> post-client selection {self.num_clients_per_round}/{len(client_indices)}')
                kwargs = {'n': self.num_clients_per_round, 'client_idxs': client_indices, 'round': round_idx}
                kwargs['results'] = self.files['prob'] if self.save_probs else None
                # select by local models(gradients)
                if self.args.method in NEED_LOCAL_MODELS_METHOD:
                    local_models = [self.client_list[idx].trainer.get_model() for idx in client_indices]
                    selected_client_indices = self.selection_method.select(**kwargs, metric=local_models)
                    del local_models
                # select by local losses
                else:
                    selected_client_indices = self.selection_method.select(**kwargs, metric=local_metrics)
                if self.args.method in CLIENT_UPDATE_METHOD:
                    for idx in client_indices:
                        self.client_list[idx].update_ema_variables(round_idx)
                # update local metrics
                client_indices = np.take(client_indices, selected_client_indices).tolist()
                local_losses = np.take(local_losses, selected_client_indices)
                accuracy = np.take(accuracy, selected_client_indices)
            
            # these clients will actually participate this round
            self.participation_counts[client_indices] += 1
            self.participation_history.append((round_idx, list(client_indices)))


            ## CHECK and SAVE current updates
            # self.weight_variance(local_models) # check variance of client weights
            self.save_current_updates(local_losses, accuracy, len(client_indices), phase='Train', round=round_idx)
            self.save_selected_clients(round_idx, client_indices)


            ## SERVER AGGREGATION
            # DEBUGGING
            assert len(client_indices) == self.num_clients_per_round

            # aggregate local models
            local_models = [self.client_list[idx].trainer.get_model() for idx in client_indices]
            if self.args.fed_algo == 'FedAvg':
                global_model_params = self.federated_method.update(local_models, client_indices)
            else:
                global_model_params = self.federated_method.update(local_models, client_indices, self.global_model, self.client_list)

            # update aggregated model to global model
            self.global_model.load_state_dict(global_model_params)
            
            # after final selection is decided (the clients that will really train this round)
            #self._log_selected_client_prototypes_from_valid(round_idx, client_indices)
            self._log_selected_client_prototype_scores(round_idx, client_indices)
            
            ## TEST
            """
            if round_idx % self.args.test_freq == 0:
                self.global_model.eval()
                # test on train dataset
                if self.test_on_training_data:
                    self.test(self.total_num_client, phase='TrainALL')
                    self.test_on_training_data = False

                # test on test dataset
                self.test(len(self.test_clients), phase='Test')
            """

            # PATCH: remember round index for logging

            # Replace the old "global test" with per-client validation averaging:
            avg_val_acc, per_client_vals = self.evaluate_on_client_validation()
      
            # You can print/log this:
            print(f"[Round {round_idx}] Avg client-val accuracy: {avg_val_acc:.4f}")
            # --- Stop timer & append round record ---
            elapsed = time.time() - round_t0
            self.cum_time += elapsed
            self.round_logs.append({
    "round": int(round_idx),
    "elapsed_sec": float(elapsed),
    "cum_time_sec": float(self.cum_time),
    "avg_val_acc": float(avg_val_acc),
    "method": str(self.args.method),
})
            # ---------------------------------------------------
            # CHECKPOINT: save every args.ckpt_freq rounds
            # ---------------------------------------------------
            ckpt_freq = getattr(self.args, "ckpt_freq", 20)
            if (round_idx + 1) % ckpt_freq == 0 or (round_idx + 1) == self.total_round:
                self.save_checkpoint(round_completed=round_idx + 1)


            # If you use wandb:
            if getattr(self.args, "wandb", False):
              import wandb
              wandb.log({"avg_client_val_acc": avg_val_acc, "round": round_idx})

            del local_models, local_losses, accuracy

        self.save_participation_report(title=self.args.method)

        # --- Persist per-round CSV and Acc-vs-Time plot ---
        try:
          df_round = pd.DataFrame(self.round_logs)
          if not df_round.empty:
            df_round.to_csv(self.round_csv, index=False)
            print(f"[RoundLog] saved: {self.round_csv}")

            # Plot: Average client-validation accuracy vs cumulative time
            plt.figure(figsize=(8, 4))
            plt.plot(df_round["cum_time_sec"], df_round["avg_val_acc"], marker="o")
            plt.xlabel("Cumulative time (s)")
            plt.ylabel("Avg client-val accuracy")
            plt.title(f"Accuracy vs Time — {self.args.method}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.acc_time_png, dpi=180)
            plt.close()
            print(f"[Plot] saved: {self.acc_time_png}")
        except Exception as e:
          print(f"[RoundLog] save failed: {e}")

        #self.save_proto_domain_heatmap(title_suffix=self.args.method)
        self.save_proto_score_heatmap(title_suffix=self.args.method)

        for k in self.files:
            if self.files[k] is not None:
                self.files[k].close()


    def local_training(self, client_idx , cfg=None):
        """
        train one client
        ---
        Args
            client_idx: client index for training
        Return
            result: trained model, (total) loss value, accuracy
        """
        client = self.client_list[client_idx]
        print(len(self.client_list))
        print(type(self.client_list))
        if self.args.method in LOSS_THRESHOLD:
            client.trainer.update_ltr(self.ltr)
        result = client.train(deepcopy(self.global_model), cfg=cfg)
        return result

    def local_testing(self, client_idx):
        """
        test one client
        ---
        Args
            client_idx: client index for test
            results: loss, acc, auc
        """
        client = self.client_list[client_idx]
        result = client.test(self.global_model, self.test_on_training_data)
        return result

    def train_clients(self, client_indices,round_idx):
        """
        train multiple clients (w. or w.o. multi processing)
        ---
        Args
            client_indices: client indices for training
        Return
            trained models, loss values, accuracies
        """
        local_losses, accuracy, local_metrics = [], [], []
        ll, lh = np.inf, 0.
        # local training with multi processing
        if self.args.use_mp:
            iter = 0
            with mp.pool.ThreadPool(processes=self.nCPU) as pool:
                iter += 1
                result = list(pool.imap(self.local_training, client_indices))

                result = {k: [result[idx][k] for idx in range(len(result))] for k in result[0].keys()}
                local_losses.extend(result['loss'])
                accuracy.extend(result['acc'])
                local_metrics.extend(result['metric'])

                progressBar(len(local_losses), len(client_indices),
                            {'loss': sum(result['loss'])/len(result), 'acc': sum(result['acc'])/len(result)})

                if self.args.method in LOSS_THRESHOLD:
                    if min(result['llow']) < ll: ll = min(result['llow'])
                    lh += sum(result['lhigh'])
        # local training without multi processing
        else:
            for client_idx in client_indices:
                client_config = {
            "server_round": round_idx if round_idx is not None else -1,
            "total_rounds": self.total_round,
            "simulate_stragglers": self.simulate_stragglers,  # e.g. "0,1"
            "delay_base_sec": self.delay_base_sec,
            "delay_jitter_sec": self.delay_jitter_sec,
            "delay_prob": self.delay_prob,
        }

                result = self.local_training(client_idx , cfg=client_config)

                

                local_losses.append(result['loss'])
                accuracy.append(result['acc'])
                local_metrics.append(result['metric'])

                if self.args.method in LOSS_THRESHOLD:
                    if result['llow'] < ll: ll = result['llow'].item()
                    lh += result['lhigh']

                progressBar(len(local_losses), len(client_indices), result)

        if self.args.method in LOSS_THRESHOLD:
            lh /= len(client_indices)
            self.ltr = self.selection_method.update(lh, ll, self.ltr)

        print()
        return local_losses, accuracy, local_metrics


    def test(self, num_clients_for_test, phase='Test'):
        """
        test multiple clients
        ---
        Args
            num_clients_for_test: number of clients for test
            TrainALL: test on train dataset
            Test: test on test dataset
        """
        metrics = {'loss': [], 'acc': []}
        if self.args.use_mp:
            iter = 0
            with mp.pool.ThreadPool(processes=self.nCPU) as pool:
                iter += 1
                result = list(tqdm(pool.imap(self.local_testing, [*range(num_clients_for_test)]),
                                   desc=f'>> local testing on {phase} set'))

                result = {k: [result[idx][k] for idx in range(len(result))] for k in result[0].keys()}
                metrics['loss'].extend(result['loss'])
                metrics['acc'].extend(result['acc'])

                progressBar(len(metrics['acc']) * iter, num_clients_for_test, phase='Test',
                            result={'loss': sum(result['loss']) / len(result), 'acc': sum(result['acc']) / len(result)})
        else:
            for client_idx in range(num_clients_for_test):
                result = self.local_testing(client_idx)

                metrics['loss'].append(result['loss'])
                metrics['acc'].append(result['acc'])

                progressBar(len(metrics['acc']), num_clients_for_test, result, phase='Test')

        print()
        self.save_current_updates(metrics['loss'], metrics['acc'], num_clients_for_test, phase=phase)


    def save_current_updates(self, losses, accs, num_clients, phase='Train', round=None):
        """
        update current updated results for recording
        ---
        Args
            losses: losses
            accs: accuracies
            num_clients: number of clients
            phase: current phase (Train or TrainALL or Test)
            round: current round
        Return
            record "Round,TrainLoss,TrainAcc,TestLoss,TestAcc"
        """
        loss, acc = sum(losses) / num_clients, sum(accs) / num_clients

        if phase == 'Train':
            self.record = {}
            self.round = round
        self.record[f'{phase}/Loss'] = loss
        self.record[f'{phase}/Acc'] = acc
        status = num_clients if phase == 'Train' else 'ALL'

        print('> {} Clients {}ing: Loss {:.6f} Acc {:.4f}'.format(status, phase, loss, acc))

        if phase == 'Test':
            wandb.log(self.record)
            if self.save_results:
                if self.test_on_training_data:
                    tmp = '{:.8f},{:.4f},'.format(self.record['TrainALL/Loss'], self.record['TrainALL/Acc'])
                else:
                    tmp = ''
                rec = '{},{:.8f},{:.4f},{}{:.8f},{:.4f}\n'.format(self.round,
                                                                  self.record['Train/Loss'], self.record['Train/Acc'], tmp,
                                                                  self.record['Test/Loss'], self.record['Test/Acc'])
                self.files['result'].write(rec)

    def save_selected_clients(self, round_idx, client_indices):
        """
        save selected clients' indices
        ---
        Args
            round_idx: current round
            client_indices: clients' indices to save
        """
        self.files['client'].write(f'{round_idx+1},')
        np.array(client_indices).astype(int).tofile(self.files['client'], sep=',')
        self.files['client'].write('\n')

    def weight_variance(self, local_models):
        """
        calculate the variances of model weights
        ---
        Args
            local_models: local clients' models
        """
        variance = 0
        for k in tqdm(local_models[0].state_dict().keys(), desc='>> compute weight variance'):
            tmp = []
            for local_model_param in local_models:
                tmp.extend(torch.flatten(local_model_param.cpu().state_dict()[k]).tolist())
            variance += torch.var(torch.tensor(tmp), dim=0)
        variance /= len(local_models)
        print('variance of model weights {:.8f}'.format(variance))



def progressBar(idx, total, result, phase='Train', bar_length=20):
    """
    progress bar
    ---
    Args
        idx: current client index or number of trained clients till now
        total: total number of clients
        phase: Train or Test
        bar_length: length of progress bar
    """
    percent = float(idx) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r> Client {}ing: [{}] {}% ({}/{}) Loss {:.6f} Acc {:.4f}".format(
        phase, arrow + spaces, int(round(percent * 100)), idx, total, result['loss'], result['acc'])
    )
    sys.stdout.flush()