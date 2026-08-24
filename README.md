# RoutingGPT

A small experimental GPT-style language model using **routing-based sparse key/value attention** to reduce the computational and memory cost of attention at longer context lengths.

> **Status:** Experimental / research project

**Note:** The routing ratio can be any value between 0 and 1. `0.50` is simply an example ratio, meaning that 50% of the tokens are selected as keys/values. The same applies to references such as `50%`. `50%` is the current recommended value to use for Routing Attention based on the available test results.

RoutingGPT is designed to investigate whether dynamically selecting a subset of tokens as attention keys and values can provide useful long-context efficiency while retaining good quality.

---

## Overview

Standard self-attention allows every token to attend to every other token:

$$O(T^2)$$

where $T$ is the sequence length.

RoutingGPT introduces a lightweight router (linear layer) that assigns a score to every token and selects the highest scoring tokens to be used as keys and values. The number of tokens chosen is simply T multiplied by the ratio.

With a routing ratio of 50% (0.50):

$$K = 0.50T$$

The resulting attention operation becomes:

$$T \times K$$

or:

$$T \times 0.50T = 0.50T^2$$

This means the attention operation performs about 50% as many query and key interactions as dense attention.

---

## How routing attention works

For an input:

A small routing network produces one score per token.
The highest-scoring tokens are selected:
T tokens -> Top-K router -> 0.50T tokens
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
K: 0.50T
V: 0.50T
Attention: T * 0.50T
```

For example, at a sequence length of 16,384 (ratio 0.50):
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
| Vocabulary	        | 50,257            |
| Embedding dimension | 512               |
| Attention heads	    | 8                 |
| Transformer layers  | 8                 |
| Context length	    | 512               |
| Routing ratio	      | 50%               |
| Batch size	        | 4                 |
| Optimizer	          | AdamW             |
| Learning rate	      | 3e-4              |
| Min learning rate   | 3e-5              |
| Scheduler           | CosineAnnealingLR |
| Dataset	            | TinyStories       | 
| Training steps      | 25,000            |
| Tokenizer	          | GPT-2             |

---

## Efficiency

The number of query key interactions is always about 50% of dense attention.
However, total model memory is not reduced by 50%
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

## Current tests at 512 context
| Type   | Results                                                  | Thoughts                                                       |
| ------ | -------------------------------------------------------- | -------------------------------------------------------------- |
| Dense  | Step 25000/25000, Loss: 1.5717, LR: 3.00e-05 Time: 0.05s | Best quality, highest attention cost                           |
| 50%    | Step 25000/25000, Loss: 1.7993, LR: 3.00e-05 Time: 0.05s | ~14.5% higher loss, substantially fewer attention interactions |
| 25%    | Step 25000/25000, Loss: 2.2641, LR: 3.00e-05 Time: 0.03s | ~44% higher loss, but ~40% faster/step                         |

So 50% ratio+ appears to be the way to go. 75% Results coming soon.
Larger context tests would help massively. May come soon.

---

## License

RoutingGPT is licensed under the **Apache License 2.0**.

You are free to use, modify, distribute, and use this project commercially, subject to the terms of the Apache License 2.0.

Copyright © 2026 Noah De Angelis

See [LICENSE](LICENSE) for the full license text.
