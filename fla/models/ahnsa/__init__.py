# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.ahnsa.configuration_ahnsa import AHNSAConfig
from fla.models.ahnsa.modeling_ahnsa import AHNSAForCausalLM, AHNSAModel

AutoConfig.register(AHNSAConfig.model_type, AHNSAConfig, exist_ok=True)
AutoModel.register(AHNSAConfig, AHNSAModel, exist_ok=True)
AutoModelForCausalLM.register(AHNSAConfig, AHNSAForCausalLM, exist_ok=True)


__all__ = ['AHNSAConfig', 'AHNSAForCausalLM', 'AHNSAModel']
