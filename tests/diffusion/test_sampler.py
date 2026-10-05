# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import pytest
import torch

from vllm_omni.diffusion.sampler import EulerSampler, ResMultistepSampler, Sampler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# Fixed outputs recorded from H3 before migration (13ea5ff91).
_REFERENCE: dict[str, Any] = {
    "source": "H3 sampler before migration (13ea5ff91)",
    "sigmas": [1.0, 0.9, 0.6, 0.2, 0.0],
    "initial": [-0.75, 0.25, 1.5],
    "cases": [
        {
            "name": "euler",
            "dtype": "float32",
            "outputs": [
                [-0.684999942779541, 0.24500000476837158, 1.40749990940094],
                [-0.4418333172798157, 0.27116668224334717, 1.1624165773391724],
                [0.014355584979057312, 0.39462223649024963, 0.8699555397033691],
                [0.5043066740036011, 0.6183866858482361, 0.7609866857528687],
            ],
        },
        {
            "name": "euler",
            "dtype": "float16",
            "outputs": [
                [-0.68505859375, 0.2449951171875, 1.4072265625],
                [-0.44189453125, 0.271240234375, 1.162109375],
                [0.01432037353515625, 0.39453125, 0.86962890625],
                [0.50439453125, 0.6181640625, 0.7607421875],
            ],
        },
        {
            "name": "euler",
            "dtype": "bfloat16",
            "outputs": [
                [-0.68359375, 0.2451171875, 1.40625],
                [-0.44140625, 0.271484375, 1.1640625],
                [0.01434326171875, 0.39453125, 0.87109375],
                [0.50390625, 0.6171875, 0.76171875],
            ],
        },
        {
            "name": "res_multistep",
            "dtype": "float32",
            "outputs": [
                [-0.684999942779541, 0.24500000476837158, 1.40749990940094],
                [-0.3429059088230133, 0.3557170629501343, 1.2289958000183105],
                [0.3096112310886383, 0.6082637906074524, 0.9815794229507446],
                [0.5928833484649658, 0.6824791431427002, 0.7944738268852234],
            ],
        },
        {
            "name": "res_multistep",
            "dtype": "float16",
            "outputs": [
                [-0.68505859375, 0.2449951171875, 1.4072265625],
                [-0.343017578125, 0.355712890625, 1.228515625],
                [0.309326171875, 0.6083984375, 0.98193359375],
                [0.5927734375, 0.6826171875, 0.794921875],
            ],
        },
        {
            "name": "res_multistep",
            "dtype": "bfloat16",
            "outputs": [
                [-0.68359375, 0.2451171875, 1.40625],
                [-0.341796875, 0.35546875, 1.2265625],
                [0.3125, 0.60546875, 0.98046875],
                [0.59375, 0.6796875, 0.796875],
            ],
        },
    ],
}


@pytest.mark.parametrize("case", _REFERENCE["cases"], ids=lambda c: f"{c['name']}-{c['dtype']}")
def test_matches_pre_migration_h3_trajectory(case):
    dtype = getattr(torch, case["dtype"])
    state = torch.tensor(_REFERENCE["initial"], dtype=dtype)
    sampler = Sampler(case["name"], _REFERENCE["sigmas"])
    for step, expected in enumerate(case["outputs"]):
        denoised = state * 0.3 + (step + 1) * 0.125
        state = sampler.step(state, denoised, step)
        assert state.dtype == dtype
        torch.testing.assert_close(state, torch.tensor(expected, dtype=dtype), rtol=0, atol=0)


@pytest.mark.parametrize("name", [None, "euler", "res_multistep"])
def test_first_step_and_zero_sigma_endpoint(name):
    state = torch.tensor([2.0, -1.0])
    denoised = torch.tensor([0.25, 0.5])
    sampler = Sampler(name, [1.0, 0.5, 0.0])
    state = sampler.step(state, denoised, 0)
    torch.testing.assert_close(state, torch.tensor([1.125, -0.25]), rtol=0, atol=0)
    torch.testing.assert_close(sampler.step(state, denoised, 1), denoised, rtol=0, atol=0)
    single_step = Sampler(name, [1.0, 0.0])
    torch.testing.assert_close(single_step.step(state, denoised, 0), denoised, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["euler", "res_multistep"])
@pytest.mark.parametrize(
    "sigmas",
    [[], [1], [1, 1, 0], [0, 1], [1, -0.1], [1.1, 0], [float("nan"), 0], [float("inf"), 0]],
)
def test_rejects_invalid_schedule(name, sigmas):
    with pytest.raises(ValueError, match="sigmas"):
        Sampler(name, sigmas)


def test_default_and_unknown_sampler():
    assert Sampler(None, [1, 0]).name == "euler"
    with pytest.raises(ValueError, match="Unsupported sampler"):
        Sampler("unknown", [1, 0])


@pytest.mark.parametrize("name", ["euler", "res_multistep"])
def test_rejects_skipped_repeated_and_exhausted_steps(name):
    sampler = Sampler(name, [1, 0.5, 0])
    state = torch.ones(2)
    for invalid in [-1, 1, 2]:
        with pytest.raises(ValueError, match="once, in order"):
            sampler.step(state, state, invalid)
    sampler.step(state, state, 0)
    with pytest.raises(ValueError, match="once, in order"):
        sampler.step(state, state, 0)
    sampler.step(state, state, 1)
    with pytest.raises(ValueError, match="once, in order"):
        sampler.step(state, state, 2)


def test_history_is_private_and_survives_caller_mutation():
    sigmas = [1.0, 0.75, 0.25, 0.0]
    first, second = ResMultistepSampler(sigmas), ResMultistepSampler(sigmas)
    prediction = torch.tensor([0.2, -0.3], requires_grad=True)
    saved = prediction.detach().clone()
    state = torch.ones(2)
    first_state = first.step(state, prediction, 0)
    second_state = second.step(state, -prediction, 0)
    with torch.no_grad():
        prediction.add_(100)
    torch.testing.assert_close(first.old_denoised, saved)
    torch.testing.assert_close(second.old_denoised, -saved)
    assert not first.old_denoised.requires_grad
    assert first.old_denoised.data_ptr() != prediction.data_ptr()
    for sampler, initial_prediction, initial_state in [(first, saved, first_state), (second, -saved, second_state)]:
        independent = ResMultistepSampler(sigmas)
        expected = independent.step(state, initial_prediction, 0)
        expected = independent.step(expected, saved, 1)
        actual = sampler.step(initial_state, saved, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(sampler.step(actual, saved, 2), saved, rtol=0, atol=0)
        assert sampler.old_denoised is None


@pytest.mark.parametrize("step", [0, 1])
@pytest.mark.parametrize("invalid", [torch.ones(3), torch.tensor([float("nan"), 0]), torch.ones(2, dtype=torch.int64)])
def test_invalid_prediction_does_not_advance_history(step, invalid):
    sampler = ResMultistepSampler([1, 0.75, 0.25, 0])
    state = torch.ones(2)
    if step:
        state = sampler.step(state, state, 0)
    history = sampler.old_denoised
    with pytest.raises(ValueError):
        sampler.step(state, invalid, step)
    assert sampler.step_index == step
    assert sampler.old_denoised is history


def test_euler_does_not_initialize_res_coefficients(monkeypatch):
    def unexpected_coefficients(sigmas):
        pytest.fail("Euler must not compute RES coefficients")

    monkeypatch.setattr(ResMultistepSampler, "compute_coefficients", staticmethod(unexpected_coefficients))
    euler = EulerSampler([1, 0.5, 0])
    sampler = Sampler("euler", [1, 0.5, 0])
    assert not hasattr(euler, "old_denoised")
    assert not hasattr(sampler, "old_denoised")
    state, denoised = torch.ones(2), torch.zeros(2)
    torch.testing.assert_close(sampler.step(state, denoised, 0), euler.step(state, denoised, 0))


def test_h3_legacy_imports_use_shared_implementation():
    from vllm_omni.diffusion.models.minimax_h3.sampling import (
        H3SampleSolver,
        create_h3_sample_solver,
        res_multistep_coeffs,
    )
    from vllm_omni.diffusion.models.minimax_h3.scheduling_minimax_h3_euler_ancestral import (
        minimax_h3_euler_eta0_step,
    )

    assert H3SampleSolver is Sampler
    assert create_h3_sample_solver is Sampler
    assert minimax_h3_euler_eta0_step is EulerSampler.step_denoised
    assert res_multistep_coeffs is ResMultistepSampler.compute_coefficients
