from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from sas_memory.controllers import GroupController
from scripts.analyze_semantic_single_path_controls import check_scope, fresh_initial_fingerprint
from scripts.run_controller_trial import state_fingerprint


class SinglePathScopeTests(unittest.TestCase):
    def model(self):
        config = {'hidden_size':8, 'num_attention_heads':2, 'num_key_value_heads':1, 'head_dim':2,
                  'linear_num_key_heads':2, 'linear_key_head_dim':3, 'linear_num_value_heads':2}
        model = nn.Module()
        model.config = SimpleNamespace(**config, layer_types=['linear_attention']*3+['full_attention'])
        model.model = nn.Module()
        layers=[]
        for _ in range(3):
            layer=nn.Module(); layer.linear_attn=nn.Module()
            layer.linear_attn.in_proj_a=nn.Linear(8,2,bias=False)
            layer.linear_attn.in_proj_b=nn.Linear(8,2,bias=False)
            layer.linear_attn.key_dim=6
            layer.linear_attn.in_proj_qkv=nn.Linear(8,18,bias=False)
            layers.append(layer)
        layer=nn.Module(); layer.self_attn=nn.Module(); layer.self_attn.head_dim=2
        layer.self_attn.q_proj=nn.Linear(8,8,bias=False)
        layer.self_attn.k_proj=nn.Linear(8,2,bias=False)
        layers.append(layer)
        model.model.layers=nn.ModuleList(layers)
        model.requires_grad_(False)
        return model,config

    def test_reconstructed_seed_matches_real_controller_and_preserves_rng(self):
        for mode in ('gdn_only','attention_only'):
            model,config=self.model()
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(1701)
                controller=GroupController(model,0,3,mode,adapt_gdn_keys=True)
            try:
                spec={'mode':mode,'rank':3,'token_policy':'all','parameters':sum(p.numel() for p in controller.parameters())}
                check_scope(controller.description(),spec,config,0)
                before=torch.get_rng_state().clone()
                reconstructed=fresh_initial_fingerprint(config,spec,1701)
                self.assertTrue(torch.equal(before,torch.get_rng_state()))
                self.assertEqual(reconstructed,state_fingerprint(controller))
                self.assertNotEqual(reconstructed,fresh_initial_fingerprint(config,spec,1702))
            finally:
                controller.close()

    def test_cross_component_projection_is_rejected(self):
        model,config=self.model()
        controller=GroupController(model,0,3,'gdn_only',adapt_gdn_keys=True)
        try:
            description=deepcopy(controller.description())
            spec={'mode':'gdn_only','rank':3,'parameters':description['trainable_parameters']}
            description['output_shapes']['attention_q_proj']=[4,3]
            with self.assertRaises(ValueError):
                check_scope(description,spec,config,0)
        finally:
            controller.close()


if __name__ == '__main__':
    unittest.main()
