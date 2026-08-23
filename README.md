# RoutingGPT

A small experimental GPT-style language model using **routing-based sparse key/value attention** to reduce the computational and memory cost of attention at longer context lengths.

> **Status:** Experimental / research project

> **Note:** In every reference to "0.25T", 0.25 can be any number bertween zero and one. 0.25 is simply a ratio which decides the strength of Routing Attention. Same goes for any references of "25%"

RoutingGPT is designed to investigate whether dynamically selecting a subset of tokens as attention keys and values can provide useful long-context efficiency while retaining good quality.

---

## Overview

Standard self-attention allows every token to attend to every other token:

$$O(T^2)$$

where $T$ is the sequence length.

RoutingGPT introduces a lightweight router (linear layer) that assigns a score to every token and selects the highest scoring tokens to be used as keys and values. The number of tokens chosen is simply T multiplied by the ratio.

With a routing ratio of 25% (0.25):

$$K = 0.25T$$

The resulting attention operation becomes:

$$T \times K$$

or:

$$T \times 0.25T = 0.25T^2$$

This means the attention operation performs about 25% as many query and key interactions as dense attention.

---

## How routing attention works

For an input:

A small routing network produces one score per token.
The highest-scoring tokens are selected:
T tokens -> Top-K router -> 0.25T tokens
Only these selected tokens are projected into keys and values.
The queries still contain every token, while keys and values contain only the selected tokens.
Attention is then performed using F.scaled_dot_product_attention()

---

## Why route keys and values?

In dense attention, the attention tensor is huuuge, containing the attention score for every token paired with every other token.
In dense attention:
```
Q: T
K: T
V: T
Attention: T * T
```

In routing attention:
```
Q: T
K: 0.25T
V: 0.25T
Attention: T * 0.25T
```

For example, at a sequence length of 16,384 (ratio 0.25):
```
Dense:
16,384 × 16,384 = 268,435,456

Routing:
16,384 × 4,096 = 67,108,864
```
This is a 75% reduction in query key interactions.

---

## Architecture

The current model is a small GPT-style decoder-only Transformer.
Default configuration:

| Parameters          | Value             |
| --------------------|------------------ |
| Vocabulary	      | 50,257            |
| Embedding dimension | 512               |
| Attention heads	  | 8                 |
| Transformer layers  | 8                 |
| Context length	  | 512               |
| Routing ratio	      | 25%               |
| Batch size	      | 4                 |
| Optimizer	          | AdamW             |
| Learning rate	      | 3e-4              |
| Min learning rate   | 3e-5              |
| Scheduler           | CosineAnnealingLR |
| Dataset	          | TinyStories       | 
| Training steps      | 25,000            |
| Tokenizer	          | GPT-2             |

---

## Efficiency

The number of query key interactions is always about 25% of dense attention.
However, total model memory is not reduced by 75%
The actual VRAM usage therefore depends on the full training configuration.

## Current limitations

**The router is currently a very simple linear layer:**
```python
self.router = nn.Linear(embed_dim, 1)
```
It must learn which tokens are useful as keys and values.

**No KV cache yet.**
Inference currently recomputes the model over the entire context for every generated token.

---

## License

RoutingGPT is licensed under the **Apache License 2.0**.

You are free to use, modify, distribute, and use this project commercially, subject to the terms of the Apache License 2.0.

Copyright © 2026 Noah De Angelis

See [LICENSE](LICENSE) for the full license text.
