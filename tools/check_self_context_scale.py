"""Constructed-input checks for a proposed operator, not a trained model."""
import json
import torch
from experiments.mechanism_retry.model import LayerSwap


def contextual_difference(module, self_output, cross_output):
    def affine(x):
        grouped = x.reshape(-1, module.groups, module.group_channels)
        return (grouped * module.group_scale[None, :, None] + module.group_bias[None, :, None]).flatten(1)
    anchor, cross = affine(self_output), affine(cross_output)
    eps = torch.finfo(anchor.dtype).eps if module.norm.eps is None else module.norm.eps
    scale = (anchor.square().mean(-1, keepdim=True) + eps).sqrt()
    # Both terms use the same self-context scale; no stop-gradient is proposed.
    return module.activation(module.norm.weight * cross / scale) - module.activation(module.norm.weight * anchor / scale)


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(42)
    layer = LayerSwap("retry_centered").encoder.layers[0].long
    u = torch.ones(1, 256)
    delta = torch.zeros_like(u, requires_grad=True)
    old = LayerSwap.transform(layer, delta) - LayerSwap.transform(layer, torch.zeros_like(delta))
    new = contextual_difference(layer, u, u + delta)
    old_derivative = torch.autograd.grad(old[0, 0], delta)[0][0, 0]
    new_derivative = torch.autograd.grad(new[0, 0], delta)[0][0, 0]
    eps = torch.finfo(u.dtype).eps
    scale = (u.square().mean(-1, keepdim=True) + eps).sqrt()
    z = u / scale
    expected = (torch.sigmoid(z) + z * torch.sigmoid(z) * (1 - torch.sigmoid(z))) / scale
    torch.testing.assert_close(old_derivative, torch.tensor(.5 / eps**.5), rtol=1e-6, atol=0)
    torch.testing.assert_close(new_derivative, expected[0, 0], rtol=1e-6, atol=0)
    responses = []
    for amplitude in (.01, .1):
        change = amplitude * torch.ones_like(u)
        responses.append(dict(amplitude=amplitude,
                              old_first_coordinate=float((LayerSwap.transform(layer, change) - LayerSwap.transform(layer, torch.zeros_like(change)))[0, 0].detach()),
                              proposed_first_coordinate=float(contextual_difference(layer, u, u + change)[0, 0].detach())))
    assert responses[1]["old_first_coordinate"] / responses[0]["old_first_coordinate"] < 1.01
    assert responses[1]["proposed_first_coordinate"] / responses[0]["proposed_first_coordinate"] > 8
    with torch.no_grad():
        layer.group_bias.fill_(.4)
        layer.group_scale.fill_(2)
        layer.norm.weight.fill_(1.3)
    torch.testing.assert_close(contextual_difference(layer, u, u), torch.zeros_like(u), rtol=0, atol=0)
    print(json.dumps(dict(scope="constructed attention outputs only; not molecular evidence", optimization_steps=0,
                          old_initial_local_derivative=old_derivative.item(),
                          proposed_initial_local_derivative_at_unit_context=new_derivative.item(),
                          responses=responses, learned_affine_zero_check=True), indent=2, allow_nan=False))
