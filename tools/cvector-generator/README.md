# cvector-generator

This example demonstrates how to generate a control vector using gguf models.

Related PRs:
- [Add support for control vectors](https://github.com/ggml-org/llama.cpp/pull/5970)
- (Issue) [Generate control vector using llama.cpp](https://github.com/ggml-org/llama.cpp/issues/6880)
- [Add cvector-generator example](https://github.com/ggml-org/llama.cpp/pull/7514)

## Examples

```sh
# CPU only
./cvector-generator -m ./llama-3.Q4_K_M.gguf

# With GPU
./cvector-generator -m ./llama-3.Q4_K_M.gguf -ngl 99

# With advanced options
./cvector-generator -m ./llama-3.Q4_K_M.gguf -ngl 99 --pca-iter 2000 --pca-batch 100

# Using mean value instead of PCA
./cvector-generator -m ./llama-3.Q4_K_M.gguf --method mean

# To see help message
./cvector-generator -h
# Then, have a look at "cvector" section
```

## Tips and tricks

If you have multiple lines per prompt, you can escape the newline character (change it to `\n`). For example:

```
<|im_start|>system\nAct like a person who is extremely happy.<|im_end|>
<|im_start|>system\nYou are in a very good mood today<|im_end|>
```

Example to use output file with `llama-cli`:

(Tips: The control vector works better when apply to layers higher than 10)

```sh
./llama-cli -m ./llama-3.Q4_K_M.gguf -p "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSing a song<|im_end|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" --special --control-vector-scaled ./control_vector.gguf 0.8 --control-vector-layer-range 10 31
```

## Projection mode (directional ablation)

A control vector is normally added to the hidden state of each layer. With `--control-vector-mode project` the same file is used the other way round: the component of each layer's output along the vector is removed. This is "directional ablation" (Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction*) done at inference time: the weights are untouched, and the effect is switched off or re-scaled by loading a different vector.

For the output `x` of a layer and the vector `v` of that layer:

```
add:     x' = x + v
project: x' = x - |v| * (x . u) * u,   u = v / |v|
```

The norm of the vector is the strength of the projection: `|v| = 1` removes the component, `0 < |v| < 1` removes that fraction of it, `|v| = 2` reflects it, and a zero vector leaves the layer untouched. The sign of `v` does not matter (`v` and `-v` span the same direction), unlike in add mode. Both `--method pca` and `--method mean` write unit-norm directions, so the scale given to `--control-vector-scaled` is the strength:

```sh
# remove the direction from the outputs of layers 10..31
./llama-cli -m ./llama-3.Q4_K_M.gguf --control-vector-scaled ./control_vector.gguf 1.0 --control-vector-layer-range 10 31 --control-vector-mode project

# remove half of it
./llama-cli -m ./llama-3.Q4_K_M.gguf --control-vector-scaled ./control_vector.gguf 0.5 --control-vector-layer-range 10 31 --control-vector-mode project
```

`llama-server` takes the same flags. When several vectors are given they are summed first, as in add mode, and the sum is the direction. A strength above 1 reflects the component; it is there for experiments and usually hurts the model. In the C API the mode is the last argument of `llama_set_adapter_cvec_ext()`.

To ablate a behaviour, generate the vector as for add mode, with the positive prompts showing the behaviour and the negative prompts showing its absence; in project mode the orientation of the result is irrelevant.
