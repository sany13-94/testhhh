from .trainer import Trainer
import numpy as np
import torch.nn.functional as F
import torch
from copy import deepcopy
from FL_core.feature_extractor import FeatureExtractor, compute_macro_prototype_from_loader




class Client(object):
    def __init__(self, client_idx, nTrain, local_train_data, local_test_data, model, args):
        """
        A client
        ---
        Args
            client_idx: index of the client
            nTrain: number of train dataset of the client
            local_train_data: train dataset of the client
            local_test_data: test dataset of the client
            model: given model for the client
            args: arguments for overall FL training
        """
        self.client_idx = client_idx
        self.test_data = local_test_data
        self.device = args.device
        self.trainer = Trainer(model, args)
        self.num_epoch = args.num_epoch  # E: number of local epoch
        self.nTrain = nTrain
        self.loss_div_sqrt = args.loss_div_sqrt
        self.loss_sum = args.loss_sum

        self.labeled_indices = [*range(nTrain)]
        self.labeled_data = local_train_data  # train_data

    @torch.no_grad()
    def proto_from_validation(self, global_model, device, batch_size=64, max_batches=3):
      """
      Build a macro prototype for THIS client from its validation data (self.test_data).
      Falls back to train data if no valid set exists.
      """
      # prefer validation data (your adapter puts per-client valid sets in 'test' slot)
      ds = self.test_data if (self.test_data is not None and len(self.test_data) > 0) else self.labeled_data
      if ds is None or (hasattr(ds, "__len__") and len(ds) == 0):
        return None

      dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

      # feature extractor from GLOBAL model (not the local trainer copy)
      feat_net = FeatureExtractor(global_model).to(device)
      proto = compute_macro_prototype_from_loader(dl, feat_net, device, max_batches=max_batches)
      return None if proto is None else proto

    def train(self, global_model, cfg=None):
        """
        train each client
        ---
        Args
            global_model: given current global model
        Return
            result = model, loss, acc
        """
        # SET MODEL
        self.trainer.set_model(global_model)

        # TRAIN
        if self.num_epoch == 0:  # no SGD updates
            result = self.trainer.train_E0(self.labeled_data)
            
        else:
            #result = self.trainer.train(self.labeled_data)
            result = self.trainer.train(self.labeled_data, self.client_idx,cfg=cfg,)
        #result['model'] = self.trainer.get_model()

        # total loss / sqrt (# of local data)
        if self.loss_div_sqrt:  # total loss / sqrt (# of local data)
            result['metric'] *= np.sqrt(len(self.labeled_data))  # loss * n_k / np.sqrt(n_k)
        elif self.loss_sum:
            result['metric'] *= len(self.labeled_data)  # total loss
        
        return result

    def test(self, model, test_on_training_data=False):
        # TEST
        if test_on_training_data:
            # test on training dataset
            result = self.trainer.test(model, self.labeled_data)
        else:
            # test on test dataset
            result = self.trainer.test(model, self.test_data)
        return result

    def get_client_idx(self):
        return self.client_idx