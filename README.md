# The deployment of Sparse autoencoder

This repository shows how to train a Sparse autoencoder using residual stream extract from GPT2-small model.
The entire repository is developed based on the repository of OpenAI(https://github.com/openai/sparse_autoencoder).

### Code structure

[get_activations_from_gpt2.py](./sparse_autoencoder/get_activations_from_gpt2.py) shows how to prepare training data.

[train.py](./sparse_autoencoder/train.py) shows how to train a Sparse autoencoder using the data we've collected. Unlike other versions, this code perfectly solves dead latents dilemma leveraging auxk and some weights initialization techniques.

[interpretation.py](./sparse_autoencoder/interpretation.py) marks top-activation texts for 384 latent dimensions of the trained Sparse autoencoder.

# The following content is directly excerpted from OpenAI's repository: 

# Sparse autoencoders

This repository hosts:
- sparse autoencoders trained on the GPT2-small model's activations.
- a visualizer for the autoencoders' features

### Install

```sh
pip install git+https://github.com/openai/sparse_autoencoder.git
```

### Code structure

See [sae-viewer](./sae-viewer/README.md) to see the visualizer code, hosted publicly [here](https://openaipublic.blob.core.windows.net/sparse-autoencoder/sae-viewer/index.html).

See [model.py](./sparse_autoencoder/model.py) for details on the autoencoder model architecture.
See [train.py](./sparse_autoencoder/train.py) for autoencoder training code.
See [paths.py](./sparse_autoencoder/paths.py) for more details on the available autoencoders.

### Example usage

```py
import torch
import blobfile as bf
import transformer_lens
import sparse_autoencoder

# Extract neuron activations with transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
device = next(model.parameters()).device

prompt = "This is an example of a prompt that"
tokens = model.to_tokens(prompt)  # (1, n_tokens)
with torch.no_grad():
    logits, activation_cache = model.run_with_cache(tokens, remove_batch_dim=True)

layer_index = 6
location = "resid_post_mlp"

transformer_lens_loc = {
    "mlp_post_act": f"blocks.{layer_index}.mlp.hook_post",
    "resid_delta_attn": f"blocks.{layer_index}.hook_attn_out",
    "resid_post_attn": f"blocks.{layer_index}.hook_resid_mid",
    "resid_delta_mlp": f"blocks.{layer_index}.hook_mlp_out",
    "resid_post_mlp": f"blocks.{layer_index}.hook_resid_post",
}[location]

with bf.BlobFile(sparse_autoencoder.paths.v5_32k(location, layer_index), mode="rb") as f:
    state_dict = torch.load(f)
    autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
    autoencoder.to(device)

input_tensor = activation_cache[transformer_lens_loc]

input_tensor_ln = input_tensor

with torch.no_grad():
    latent_activations, info = autoencoder.encode(input_tensor_ln)
    reconstructed_activations = autoencoder.decode(latent_activations, info)

normalized_mse = (reconstructed_activations - input_tensor).pow(2).sum(dim=1) / (input_tensor).pow(2).sum(dim=1)
print(location, normalized_mse)
```
