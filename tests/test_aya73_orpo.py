"""ORPO loss-math pins (#73 custom Trainer; TRL 1.17 dropped ORPO).

Extracts the pure helpers from the Modal module via AST and checks
them against hand-computed values. Needs torch (env-skipped where
absent); run under a CPU-torch env before any ORPO GPU leg.
"""

import ast
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APP = ROOT / "infra" / "modal" / "finetune_aya73_followup.py"


def _helpers():
    tree = ast.parse(APP.read_text())
    want = {"_mean_response_logprob", "_log_odds", "_orpo_loss",
            "_orpo_tokenize", "_orpo_map_row", "_orpo_collate",
            "_to_messages", "_chat_ids"}
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in want]
    assert {n.name for n in nodes} == want
    ns: dict = {"torch": torch}
    exec(compile(ast.Module(nodes, []), str(APP), "exec"), ns)
    return ns


HELP = _helpers()


def test_mean_response_logprob_masks_prompt():
    fn = HELP["_mean_response_logprob"]
    logits = torch.tensor([[[2.0, 0.0, 0.0, 0.0],
                            [0.0, 2.0, 0.0, 0.0],
                            [0.0, 0.0, 2.0, 0.0]]])
    labels = torch.tensor([[-100, 1, 2]])
    got = fn(logits, labels)
    # logprob of the argmax token over 4 classes with logit gap 2.
    lp = 2.0 - math.log(math.exp(2.0) + 3.0)
    assert got.shape == (1,)
    assert abs(got.item() - lp) < 1e-6


def test_log_odds_matches_definition():
    fn = HELP["_log_odds"]
    for mean_lp in (-0.5, -1.0, -2.5, -5.0):
        p = math.exp(mean_lp)
        want = math.log(p / (1.0 - p))
        assert abs(fn(torch.tensor(mean_lp)).item() - want) < 1e-5


def test_orpo_loss_matches_paper():
    fn = HELP["_orpo_loss"]
    mean_w = torch.tensor([-0.8, -1.5])
    mean_l = torch.tensor([-2.0, -1.0])
    sft = torch.tensor(1.25)
    total, or_loss, acc = fn(mean_w, mean_l, sft, beta=0.1)

    def log_odds(m):
        p = math.exp(m)
        return math.log(p / (1.0 - p))

    ratios = [log_odds(w) - log_odds(l)
              for w, l in zip((-0.8, -1.5), (-2.0, -1.0))]
    want_or = sum(-math.log(1.0 / (1.0 + math.exp(-r)))
                  for r in ratios) / 2.0
    assert abs(or_loss.item() - want_or) < 1e-5
    assert abs(total.item() - (1.25 + 0.1 * want_or)) < 1e-5
    assert acc.item() == 0.5  # -0.8 > -2.0 but -1.5 < -1.0


def test_orpo_loss_backward_finite():
    fn = HELP["_orpo_loss"]
    mean_w = torch.tensor([-1.0], requires_grad=True)
    mean_l = torch.tensor([-2.0], requires_grad=True)
    sft = torch.tensor(1.0, requires_grad=True)
    total, or_loss, _ = fn(mean_w, mean_l, sft, beta=0.1)
    total.backward()
    assert math.isfinite(total.item())
    assert math.isfinite(or_loss.item())
    assert mean_w.grad is not None and mean_l.grad is not None
    # Better chosen -> lower loss; worse rejected -> lower loss.
    assert mean_w.grad.item() < 0.0
    assert mean_l.grad.item() > 0.0


def test_orpo_collate_pads_sides():
    import torch

    collate = HELP["_orpo_collate"]

    class FakeTok:
        pad_token_id = 0

    rows = [
        {"chosen_input_ids": [1, 2, 3], "chosen_labels": [-100, 2, 3],
         "rejected_input_ids": [1, 4], "rejected_labels": [-100, 4]},
        {"chosen_input_ids": [1, 5], "chosen_labels": [-100, 5],
         "rejected_input_ids": [1, 6, 7, 8],
         "rejected_labels": [-100, 6, 7, 8]},
    ]
    batch = collate(FakeTok(), rows)
    assert batch["chosen_input_ids"].shape == (2, 3)
    assert batch["rejected_input_ids"].shape == (2, 4)
    assert batch["chosen_input_ids"][1].tolist() == [1, 5, 0]
    assert batch["chosen_labels"][1].tolist() == [-100, 5, -100]
    assert batch["chosen_attention_mask"].dtype == torch.bool
    assert batch["chosen_attention_mask"][1].tolist() == [True, True,
                                                          False]


def test_orpo_trainer_two_steps_cpu(tmp_path):
    """End-to-end ORPO plumbing: map -> collate -> Trainer.train()."""
    transformers = pytest.importorskip("transformers")
    datasets = pytest.importorskip("datasets")

    from transformers import (GPT2Config, GPT2LMHeadModel,
                              GPT2TokenizerFast, Trainer,
                              TrainingArguments)

    tree = ast.parse(APP.read_text())
    mixin = [n for n in tree.body
             if isinstance(n, ast.ClassDef)
             and n.name == "_ORPOTrainerMixin"][0]
    ns = dict(HELP)
    exec(compile(ast.Module([mixin], []), str(APP), "exec"), ns)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }}: "
                         "{{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}assistant:{% endif %}")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=64,
                                       vocab_size=tok.vocab_size))
    raw_rows = [
        {"system": "sys", "prompt": f"ask {i}",
         "chosen": "good answer here",
         "rejected": "bad answer here"}
        for i in range(4)
    ]
    raw = datasets.Dataset.from_list(raw_rows)
    ds = raw.map(lambda r: HELP["_orpo_map_row"](tok, r),
                 remove_columns=raw.column_names)
    assert set(ds.column_names) == {"chosen_input_ids", "chosen_labels",
                                    "rejected_input_ids",
                                    "rejected_labels"}

    class ORPOTrainer(ns["_ORPOTrainerMixin"], Trainer):
        def compute_loss(self, model, inputs, **kwargs):
            del kwargs
            total, _, _ = self.orpo_compute_loss(model, inputs)
            return total

    args = TrainingArguments(
        output_dir=str(tmp_path), max_steps=2,
        per_device_train_batch_size=2, report_to="none",
        remove_unused_columns=False, seed=73,
    )
    trainer = ORPOTrainer(
        model=model, args=args, train_dataset=ds,
        data_collator=lambda rows: HELP["_orpo_collate"](tok, rows),
        processing_class=tok)
    stats = trainer.train()
    assert math.isfinite(stats.training_loss)


def test_orpo_tokenize_masks_prompt():
    fn = HELP["_orpo_tokenize"]

    class FakeTok:
        def apply_chat_template(self, msgs, tokenize, **kw):
            assert tokenize
            ids = []
            for m in msgs:
                ids += [len(m["role"]), len(m["content"])]
            if kw.get("add_generation_prompt"):
                ids.append(-1)
            return ids

    out = fn(FakeTok(), [{"role": "user", "content": "hi"}],
             [{"role": "assistant", "content": "hello"}])
    assert out["input_ids"] == [4, 2, 9, 5]
    assert out["labels"] == [-100, -100, -100, 5]
    # Left-truncation keeps the response tail; prompt mask shrinks.
    out = fn(FakeTok(), [{"role": "user", "content": "hi"}],
             [{"role": "assistant", "content": "hello"}], max_length=2)
    assert out["input_ids"] == [9, 5]
    assert out["labels"] == [-100, -100]


def test_chat_ids_handles_dict_shape():
    """transformers 5 returns a BatchEncoding from apply_chat_template."""
    fn = HELP["_chat_ids"]

    class DictTok:
        def apply_chat_template(self, msgs, tokenize, **kw):
            assert tokenize
            return {"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]}

    class FakeBatchEncoding:
        def keys(self):
            return ["input_ids", "attention_mask"]

        def __getitem__(self, key):
            return {"input_ids": [7, 8, 9],
                    "attention_mask": [1, 1, 1]}[key]

    class BatchTok:
        def apply_chat_template(self, msgs, tokenize, **kw):
            assert tokenize
            return FakeBatchEncoding()

    class ListTok:
        def apply_chat_template(self, msgs, tokenize, **kw):
            assert tokenize
            return [7, 8, 9]

    assert fn(DictTok(), []) == [7, 8, 9]
    assert fn(BatchTok(), []) == [7, 8, 9]
    assert fn(ListTok(), []) == [7, 8, 9]
