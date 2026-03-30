We thank the reviewer for the thorough evaluation and constructive suggestions.
We address each point below.

   


# Weaknesses

**W1. Limited number of samples.**

We reproduced the data preparation pipeline from Kuratov et al. and re-ran all experiments on the same 50 samples used in the original full cramming paper. We present the updated version of progressive cramming table [Table 6](https://drive.google.com/file/d/148Cxg2jOx_-MIH5XC0I-oPbwVLG0Ktiz/view?usp=sharing). Our main conclusions remain valid on this data.

However, the expanded evaluation revealed an important finding: for all models
except Llama-3.1-8B, activation alignment leads to significant degradation in
both compressed tokens and Information Gain. The best results across all experiments
are achieved with **low-dimensional linear projection alone**. Only Llama-3.1-8B
shows competitive performance when combining low-dimensional projection with activation
alignment; for the remaining models (Pythia-1.4B, SmolLM2-1.7B, Gemma-3-4B-pt) the
combination does not improve over projection-only variants.

   
 

**W2. Causality analysis for 'attention hijacking' and 'degraded downstream performance'**

We thank the reviewer for this suggestion. We conducted causal intervention experiments
using **attention knockout** - masking pre-softmax logits corresponding to the
compression embedding to $-\infty$ at selected layers, thereby completely removing its
influence on the residual stream. Three protocols were used: (1) **per-layer knockout**
(masking at a single layer), (2) **forward cumulative knockout** (masking layers
$ 0 $ through $ k $), and (3) **reverse cumulative knockout** (masking layers $ L{-}1 $
down to $ k $). Each condition is evaluated on reconstruction accuracy and downstream
capability (HellaSwag). Llama3.1-8B model was used for this ablation.

Figures links:
* [Per-layer knockout](https://drive.google.com/file/d/1nsBwmqWAdzuaQFlCQMC8lAWjqUMpTHrB/view?usp=sharing)
* [Cummulative knockout (both forward and reverse knockout)](https://drive.google.com/file/d/1JI-N2oTszYtTH8mY_ZYG8op7r2PfiWWS/view?usp=sharing)

The results reveal that **attention mass and causal importance are dissociated**:

* Per-layer knockout at early layers degrades reconstruction while slightly improving downstream accuracy, indicating that early layers causally drive the downstream disruption.
* Forward cumulative knockout rapidly restores downstream accuracy once the first several layers are masked
* Reverse cumulative knockout does not recover downstream performance until masking reaches the early layers.


This **forward/reverse asymmetry** is the core causal argument: downstream collapse
is driven by early-layer interactions with the compression embedding. Late layers, despite
exhibiting the highest attention mass, have minimal causal impact on downstream performance
when knocked out  -  the observed attention concentration possibly is a symptom rather than the cause.



   
 

---


   
 

# Questions

### Q1. Limited number od samples

See **W1**.


   
 

### Q2. Causality analysis for attention hijacking

See **W2**.


   
 

### Q3. Downstream evaluation with progressive cramming.

We present downstream evaluation results for progressive cramming in [Table 7](https://drive.google.com/file/d/1DIjiTpyIqQ4xuW0k9Q7mjCOFuXUPz2YX/view?usp=sharing). For progressive cramming, we report metrics computed exclusively over fully converged samples (i.e., those achieving perfect reconstruction). The performance drop relative to the baseline is comparable for both full and progressive cramming across all models, ranging from approximately 34%--38%.

Note that Gemma-3-4B shows 0% convergence under progressive cramming, meaning no sample achieved perfect token-by-token reconstruction. Notably, full cramming still yields a relatively small performance drop for Gemma (54.97% vs. 57.07%). This finding requires further research.

   
 

---

   
 

## Erratum: Downstream evaluation bug

We identified a bug in the perplexity computation for the HellaSwag and ARC benchmarks in the originally submitted version. Specifically, the compression embedding introduced an off-by-one token shift that was not accounted for during likelihood scoring, resulting in the near-random accuracy reported in the original paper. The corrected results are shown above.

