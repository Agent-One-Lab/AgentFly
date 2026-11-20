"""Routing replay for K2-Horizon under the FSDP actor (``verl.models.transformers.k2_horizon``).

A tiny randomly initialised K2-Horizon model (the repo's own code, from the local HF cache) on CPU: 3 dense
layers and 2 routed ones, 6 MLP experts (top 2) and 4 value experts (top 2).
"""
import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from agentfly.verl.models.transformers.k2_horizon import (  # noqa: E402
    PAD,
    get_routing_replay,
    install_routing_replay,
)

K2 = "IFM/K2-Horizon-MoVA-36B-A4B"
LAYERS, MLP_K, VALUE_K = 5, 2, 2


@pytest.fixture(scope="module")
def model():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        config = transformers.AutoConfig.from_pretrained(
            K2, trust_remote_code=True, vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=LAYERS,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, rope_head_dim=16, moe_intermediate_size=32,
            num_experts=6, num_experts_per_tok=MLP_K, mova_num_experts=4, mova_num_experts_per_tok=VALUE_K,
            max_position_embeddings=256, attn_implementation="eager",
        )
        torch.manual_seed(0)
        model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True).eval()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"K2-Horizon model code unavailable: {e}")
    install_routing_replay(model)
    return model


class Recorder:
    """The model's own choices, in the record's layout: [positions, layers, MLP columns + value columns]."""

    def __init__(self, model, positions):
        self.replay = get_routing_replay(model)
        self.rows = torch.full((positions, LAYERS, MLP_K + VALUE_K), PAD, dtype=torch.uint8)
        self.original = self.replay.topk

    def __enter__(self):
        def topk(x, k, *args, **kwargs):
            out = self.original(x, k, *args, **kwargs)
            layer, column, _ = self.replay.current
            self.rows[:, layer, column : column + k] = out.indices.to(torch.uint8)
            return out

        self.replay.topk = topk
        return self

    def __exit__(self, *exc):
        del self.replay.topk


def logits(model, ids, **kwargs):
    with torch.no_grad():
        return model(input_ids=ids, **kwargs).logits


def test_the_models_own_choices_replay_exactly(model):
    ids = torch.randint(0, 128, (2, 12))
    with Recorder(model, 24) as rec:
        free = logits(model, ids)
    assert (rec.rows[:, 3:] != PAD).all() and (rec.rows[:, :3] == PAD).all()  # layers 0-2 are dense
    replay = get_routing_replay(model)
    replay.set(rec.rows.view(2, 12, LAYERS, -1))
    assert torch.equal(logits(model, ids), free)
    assert replay.replayed == 24 * 2 * 2 and replay.free == 0  # 2 routed layers x 2 routers
    replay.set(None)


def test_other_choices_change_the_output_and_are_used(model):
    ids = torch.randint(0, 128, (1, 10))
    with Recorder(model, 10) as rec:
        free = logits(model, ids)
    other = rec.rows.clone()
    other[:, 3:, :MLP_K] = (other[:, 3:, :MLP_K] + 1) % 6  # every MLP choice shifted by one expert
    replay = get_routing_replay(model)
    replay.set(other.view(1, 10, LAYERS, -1))
    with Recorder(model, 10) as seen:
        changed = logits(model, ids)
    replay.set(None)
    assert not torch.allclose(changed, free)
    assert torch.equal(seen.rows[:, 3:, :MLP_K], other[:, 3:, :MLP_K])


def test_positions_without_a_record_route_freely(model):
    ids = torch.randint(0, 128, (1, 10))
    with Recorder(model, 10) as rec:
        free = logits(model, ids)
    partial = rec.rows.clone()
    partial[5:] = PAD  # no record for the last five positions
    partial[0, 3, :MLP_K] = 0  # a row that is not a set of distinct experts
    replay = get_routing_replay(model)
    replay.set(partial.view(1, 10, LAYERS, -1))
    assert torch.equal(logits(model, ids), free)
    assert replay.free == 5 * 2 * 2 + 1
    replay.set(None)


def test_a_record_with_mlp_columns_only_leaves_the_value_router_free(model):
    ids = torch.randint(0, 128, (1, 10))
    with Recorder(model, 10) as rec:
        free = logits(model, ids)
    replay = get_routing_replay(model)
    replay.set(rec.rows[:, :, :MLP_K].reshape(1, 10, LAYERS, MLP_K).contiguous())
    assert torch.equal(logits(model, ids), free)
    assert replay.replayed == 10 * 2  # the MLP router of the two routed layers
    replay.set(None)


def test_packed_rows_follow_the_attention_mask(model):
    """verl's remove-padding path: the unpadded tokens of all rows run as one sequence."""
    a, b = torch.randint(0, 128, (1, 7)), torch.randint(0, 128, (1, 4))
    with Recorder(model, 7) as rec_a:
        logits(model, a)
    padded = torch.full((2, 7, LAYERS, MLP_K + VALUE_K), PAD, dtype=torch.uint8)
    padded[0] = rec_a.rows
    mask = torch.tensor([[1] * 7, [1] * 4 + [0] * 3])
    replay = get_routing_replay(model)
    replay.set(padded, attention_mask=mask, packed=True)
    assert replay.routes.shape[0] == 11 and torch.equal(replay.routes[:7], rec_a.rows)
    with pytest.raises(RuntimeError, match="recorded positions"):
        logits(model, b)  # 4 positions against 11 recorded
    replay.set(None)


def test_the_backward_pass_recomputes_with_the_same_choices(model):
    """Gradient checkpointing runs the layers again during backward; the record must still be attached."""
    ids = torch.randint(0, 128, (1, 10))
    with Recorder(model, 10) as rec:
        logits(model, ids)
    other = rec.rows.clone()
    other[:, 3:, :MLP_K] = (other[:, 3:, :MLP_K] + 1) % 6
    replay = get_routing_replay(model)

    def grad(checkpointing):
        model.zero_grad()
        (model.gradient_checkpointing_enable if checkpointing else model.gradient_checkpointing_disable)()
        model.train()
        replay.set(other.view(1, 10, LAYERS, -1))
        model(input_ids=ids).logits.float().log_softmax(-1)[0, :-1].gather(-1, ids[0, 1:, None]).sum().backward()
        model.eval()
        return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    plain, checkpointed = grad(False), grad(True)
    model.gradient_checkpointing_disable()
    replay.set(None)
    assert torch.allclose(plain, checkpointed, atol=1e-6)


def test_a_jagged_record_is_taken_as_it_is(model):
    """The trainer hands the engine the record as a nested tensor, one jagged row per sequence."""
    a = torch.randint(0, 128, (1, 7))
    with Recorder(model, 7) as rec_a:
        free = logits(model, a)
    nested = torch.nested.nested_tensor_from_jagged(rec_a.rows, offsets=torch.tensor([0, 7]))
    replay = get_routing_replay(model)
    replay.set(nested)
    assert torch.equal(replay.routes, rec_a.rows) and torch.equal(logits(model, a), free)
    assert replay.replayed == 7 * 2 * 2
    replay.set(None)
