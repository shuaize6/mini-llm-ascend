"""小模型与 HF 对拍，不下载权重；特意令 Q 投影宽度不等于 hidden size。"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from mini_llm_ascend.engine.model import ModelConfig, Qwen3, RMSNorm, apply_rope, causal_mask, rotary_factors
from mini_llm_ascend.engine.weights import load_model


def tiny_models():
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=41, hidden_size=24, intermediate_size=40,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=64, tie_word_embeddings=True,
                        eos_token_id=2, bos_token_id=1, pad_token_id=0, sliding_window=None)
    config._attn_implementation = "eager"
    hf = Qwen3ForCausalLM(config).eval()
    model = Qwen3(ModelConfig.from_dict(config.to_dict())).eval()
    model.load_state_dict(hf.state_dict(), strict=True)
    return hf, model


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.hf, cls.model = tiny_models()

    def test_rmsnorm_has_no_mean_subtraction(self):
        x = torch.tensor([[3.0, 4.0]])
        norm = RMSNorm(2, 1e-6)
        expected = x / torch.sqrt(torch.tensor(12.5 + 1e-6))
        torch.testing.assert_close(norm(x), expected)

    def test_rope_half_pairing_and_position_offset(self):
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        cos, sin = rotary_factors(torch.tensor([0]), 4, 10000.0, torch.float32)
        torch.testing.assert_close(apply_rope(x, cos, sin), x)
        torch.testing.assert_close(apply_rope(x, torch.zeros_like(cos), torch.ones_like(sin)),
                                   torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]))
        cos, sin = rotary_factors(torch.tensor([3]), 4, 10000.0, torch.float32)
        angles = torch.tensor([3.0, 0.03, 3.0, 0.03])
        torch.testing.assert_close(cos.flatten(), angles.cos())
        torch.testing.assert_close(sin.flatten(), angles.sin())

    def test_causal_mask_includes_history(self):
        visible = causal_mask(2, 3, device="cpu", dtype=torch.float32)[0, 0] == 0
        self.assertEqual(visible.tolist(), [[True, True, True, True, False], [True]*5])

    def test_rope_matches_hf_for_real_head_dim_and_decode_positions(self):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        config = Qwen3Config(head_dim=128, rope_theta=1000000.0, sliding_window=None)
        rope = Qwen3RotaryEmbedding(config)
        x = torch.zeros(1, 1, 1024)
        for position in (list(range(21)), [21], [22], [127]):
            ids = torch.tensor(position)
            expected = rope(x, ids[None])
            actual = rotary_factors(ids, 128, config.rope_theta, x.dtype)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a[:, 0], b, atol=0, rtol=0)

    def test_all_position_logits_match_hf(self):
        ids = torch.tensor([[1, 4, 8, 6]])
        with torch.inference_mode():
            expected = self.hf(ids, use_cache=False).logits
        torch.testing.assert_close(self.model(ids, last_only=False), expected, atol=1e-5, rtol=1e-4)
        self.assertIs(self.model.lm_head.weight, self.model.model.embed_tokens.weight)

    def test_cache_chunk_and_decode_match_full_recompute(self):
        ids = torch.tensor([[1, 4, 8, 6, 3, 7]])
        cache = self.model.new_cache(8)
        # 有历史的多 token chunk 也覆盖，避免只测试 Q=1 而漏掉 mask 偏移。
        for start, end in ((0, 2), (2, 5), (5, 6)):
            actual = self.model(ids[:, start:end], cache, last_only=False)
            expected = self.model(ids[:, :end], last_only=False)[:, start:end]
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            self.assertEqual(cache.length, end)

    def test_reject_bad_input_and_unsupported_config(self):
        for ids in (torch.tensor([[41]]), torch.tensor([[-1]]), torch.empty(1, 0, dtype=torch.long),
                    torch.tensor([[1.0]]), torch.ones(2, 1, dtype=torch.long)):
            with self.assertRaises(ValueError):
                self.model(ids)
        for key, value in (("rope_scaling", {}), ("use_sliding_window", True),
                           ("attention_bias", True), ("hidden_act", "gelu")):
            data = self.hf.config.to_dict()
            data[key] = value
            with self.assertRaises(ValueError):
                ModelConfig.from_dict(data)

    def test_checkpoint_loader_strictness_and_tied_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(self.hf.config.to_dict()))
            original = {k: v.detach().clone() for k, v in self.hf.state_dict().items()}
            original.pop("lm_head.weight")  # safetensors 常只保存共享参数的一份。
            save_file(original, str(root / "model.safetensors"))
            loaded = load_model(root)
            self.assertIs(loaded.lm_head.weight, loaded.model.embed_tokens.weight)
            ids = torch.tensor([[1, 4]])
            torch.testing.assert_close(loaded(ids), self.model(ids))
            variants = []
            missing = dict(original)
            missing.pop("model.layers.0.self_attn.q_norm.weight")
            variants.append(missing)
            variants.append(dict(original, unexpected=torch.zeros(1)))
            wrong_shape = dict(original)
            wrong_shape["model.norm.weight"] = torch.zeros(2)
            variants.append(wrong_shape)
            variants.append(dict(original, **{"lm_head.weight": original["model.embed_tokens.weight"] + 1}))
            nonfinite = dict(original)
            nonfinite["model.norm.weight"] = torch.full_like(original["model.norm.weight"], float("nan"))
            variants.append(nonfinite)
            for variant in variants:
                save_file(variant, str(root / "model.safetensors"))
                with self.assertRaises(ValueError):
                    load_model(root)


if __name__ == "__main__":
    unittest.main()
