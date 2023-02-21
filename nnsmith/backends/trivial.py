import os
import warnings
from typing import Dict, Tuple

import numpy as np
import torch
from multipledispatch import dispatch

from nnsmith.backends.factory import BackendCallable, BackendFactory
from nnsmith.materialize.torch import TorchModel


class TrivialFactory(BackendFactory):
    def __init__(self, target="cpu", optmax: bool = False, **kwargs):
        super().__init__(target, optmax)
        if self.target == "cpu":
            self.device = torch.device("cpu")
        elif self.target == "cuda":
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            raise ValueError(
                f"Unknown target: {self.target}. Only `cpu` and `cuda` are supported."
            )

    @property
    def system_name(self) -> str:
        return "trivial"

    @dispatch(TorchModel)
    def make_backend(self, model: TorchModel) -> BackendCallable:
        torch_net = model.torch_model.to(self.device).eval()

        def closure(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            input_ts = {
                k: torch.from_numpy(v).to(self.device) for k, v in inputs.items()
            }
            with torch.no_grad():
                output: Tuple[torch.Tensor] = torch_net(*input_ts.values())
            return {
                k: v.cpu().detach().resolve_conj().numpy()
                if v.is_conj()
                else v.cpu().detach().numpy()
                for k, v in zip(torch_net.output_like.keys(), output)
            }

        return closure
