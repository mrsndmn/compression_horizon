We thank the reviewer for the thoughtful comments and suggestions for strengthening the paper. Our responses follow below.

 
 
# Weaknesses:

**W1. Single evaluation dataset.**

To address this issue we reproduced fanfics dataset from Kuratov et al and runned experiments on it. Results may be found in [Table 11](https://drive.google.com/file/d/1iggPKMXXLJ_uL018BLOI3Ummv780ayF7/view?usp=sharing).

The key findings are consistent with those observed on PG-19. Information gain follows the same patterns across configurations.

The low-dimensional structure of compression trajectories also generalizes to Fanfics data. While PCA 99% values are somewhat larger than those observed on PG-19-likely reflecting the greater distributional diversity of fan-fiction text-they remain at least an order of magnitude below the embedding dimensionality.

 
 

**W2. No proposal for introducing valuable semantic properties**

We agree-cramming optimizes solely for token reconstruction and does not explicitly
target semantic properties. Introducing such properties is outside the scope of this
work. Compression approaches that do optimize for semantics (e.g., Gisting, Activation
Beacon, AutoCompressors, ICAE) are discussed in the related work section and represent
a complementary research direction.

 
 

**W3. Higher cost of progressive optimization procedure**

Progressive cramming optimization is about $2\times$ slower than full cramming. The main
bottleneck in current implementation is that within a batch, a single slowly converging sample blocks new token addition for all other samples.

 
 

---

 
 

# Questions:

**Q1. How does the speed (in terms of optimization steps) of progressive optimization relate to vanilla cramming?**

Full cramming was limited to 10k steps. Progressive cramming, due to its autoregressive approach, may finish earlier. On average, optimization steps remain below 10k for progressive cramming:

* Pythia-1.4B:   $6844 \pm 2043$
* SmolLM2-1.7B:  $9391 \pm 2068$
* Gemma-3-4B-pt: $8954 \pm 2900$

Experiments for Llama 8B were not conducted due to time constraints.

**Q2. How semantic properties of the resulting representations can be evaluated other than by concatenation with the queries?**

**Generative benchmark.** We additionally report evaluation results on **5-shot MMLU** in [Table 9](https://drive.google.com/file/d/1_NJSBkvhEyUdA45HD2QD8IxIEaQEpb9t/view?usp=sharing). Compressing only few-shot examples causes modest degradation ($-2\%$ to $-15\%$), while compressing the full prefix leads to complete failure ($0\%$ valid answers), consistent with our findings on likelihood-based benchmarks. See also answer to reviewer $\texttt{eppq}$, W3.

**Different perplexity computation strategies.** We evaluated the semantic properties of compression embeddings on the **HellaSwag** and **ARC** benchmarks using different perplexity computation strategies. Specifically, we varied which token logits were included when computing perplexity:


* **baseline** – standard perplexity, no compression embeddings
* **compression** – perplexity computed with compression embeddings
* **compression only** – perplexity computed using only compression embeddings in place of the context

There are also **\*_endings** experiments modifications where perplexity was computed only with answer logits.

By comparing performance across these methods, we can isolate the contributions of context versus compression tokens and evaluate the robustness of semantic representations in compressed form (see [Table 10](https://drive.google.com/file/d/1-qex3vN5c7WCSDDHi2bbBT29Cgy0oYZO/view?usp=sharing)).

Based on this results we can conclude that **compression** is consistently better on all benchmarks than **compression only**. While **\*_endings** experiments are inconsistent across diffetent experiments setting and benchmarks.

**Q3. How can the number of optimization steps be reduced?**

Our low-dimensional projection partially addresses this-it converges faster and compresses fewer tokens. Further reduction could involve training an encoder to predict a good starting point, left for future work.

 
 

---

 
 

**Erratum: Downstream evaluation bug**

After submission, we discovered a bug in the perplexity computation for the HellaSwag and ARC benchmarks. The compression embedding introduced an off-by-one token shift that was not accounted for during likelihood scoring, causing the near-random downstream accuracy reported in the original paper. After correction, HellaSwag accuracy under cramming ranges from 34%-38% for most models, representing a consistent but moderate drop from baseline in line with our central claim that high reconstruction accuracy does not imply preservation of downstream-relevant semantics. Corrected results are presented in [Table 7](https://drive.google.com/file/d/1DIjiTpyIqQ4xuW0k9Q7mjCOFuXUPz2YX/view?usp=sharing). See also response to Reviewer $\texttt{neRn}$, Q3.
