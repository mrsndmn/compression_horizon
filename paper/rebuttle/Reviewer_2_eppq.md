We appreciate the reviewer's careful reading and valuable feedback. Below we respond to each concern.


   
 
# Weaknesses:

**W1. Limited number of samples.**

We agree that 10 samples are statistically limited, so we re-ran all experiments on the
50-sample subset from Kuratov et al. Updated
results are in [Table 6](https://drive.google.com/file/d/148Cxg2jOx_-MIH5XC0I-oPbwVLG0Ktiz/view?usp=sharing).

Also see **W1** in our answer to reviewer $\texttt{neRn}$.


   
 


**W2. Imperfect compression for HellaSwag and ARC evaluation**

Due to HellaSwag and ARC benchmarks has short sequences (mostly less than 100 tokens), it would not be a problem to cram the sequences. Below we report the portion of fully converged samples for SmolLM2 (full cramming):
* HellaSwag: 96%
* ARC-Easy: 98%
* ARC-Challenge: 97%

In [Table 8](https://drive.google.com/file/d/1B4ALS1umyqq70ScHf-_1XtVp04rdG6eY/view?usp=sharing) we report benchmarks results only for fully converged samples for SmolLM2 model (full cramming setup). Imperfect reconstruction has minimal impact on benchmark performance. This confirms that the observed degradation is driven primarily by the compression embedding itself, rather than by reconstruction errors.

We also evaluated percent of fully converged samples and benchmarks results for progressive cramming in [Table 7](https://drive.google.com/file/d/1DIjiTpyIqQ4xuW0k9Q7mjCOFuXUPz2YX/view?usp=sharing). For all models except gemma there is convergence above 95%. For more details see our response to reviewer $\texttt{neRn}$ to **Q1**.


   
 

**W3. Narrow downstream evaluation relying only on HellaSwag and ARC**

We agree that HellaSwag and ARC alone probe a limited slice of downstream behavior.
To address this, we additionally evaluated on **5-shot MMLU** (512 samples) under
three compression modes:
* $\texttt{few-shot}$ - only few-shot examples are compressed
* $\texttt{full-prefix}$ - both few-shot examples and the question are compressed
* $\texttt{random}$ - a random embedding replaces the compression embedding as a control

The results in [Table 9](https://drive.google.com/file/d/1_NJSBkvhEyUdA45HD2QD8IxIEaQEpb9t/view?usp=sharing) are consistent with our earlier findings. When only $\texttt{few-shot}$ tokens are
compressed, all models shows accuracy degradation (ranging from
$-2\%$ for Gemma-3-4B to $-15\%$ for Llama-3.1-8B), confirming that the model retains
generative capability when uncompressed tokens follow the compression embedding. In
contrast, $\texttt{full-prefix}$ compression where compression embedding compresses full prefix leads to **complete failure**: accuracy drops to
near zero and the models produce no valid (parseable) answers, indicating total collapse
of the generative process. Notably, the $\texttt{random}$ baseline causes negligible
degradation, showing that the mere presence of an out-of-distribution
token is not sufficient to explain the failure it is specifically the optimized
compression embedding that disrupts downstream computation.

These results on a generative benchmark with longer context reinforce the conclusions
drawn from HellaSwag and ARC, and demonstrate that the downstream collapse generalizes
beyond multiple-choice likelihood-based evaluation.


   
 

**W4. Compute expenditure vs. statistical coverage**

The reported 2800 GPU-hours represent total compute including all debugging,
hyperparameter sweeps, and failed runs not the cost of final experiments alone.

Progressive cramming is inherently expensive: current implementation was optimized through dynamic batching and code optimization, but it remains roughly $2\times$ slower than full cramming.

As noted in our response to reviewer $\texttt{neRn}$ to **W1**, we have now extended all experiments to 50 samples following Kuratov et al., and the main conclusions remain stable [Table 6](https://drive.google.com/file/d/148Cxg2jOx_-MIH5XC0I-oPbwVLG0Ktiz/view?usp=sharing).


   
 

**W5. Paper length and organization**

We agree that the main body can be made more substantial. In the revised version we will
reorganize the paper, ensuring the paper makes full use of the available page budget.


   
 


<!-- **Follow-up: Justification for progressive cramming** -->

We appreciate the reviewer's continued engagement. We address the remaining concerns below.


   
 

**Why progressive cramming is necessary despite higher compute cost.**

As discussed in Section 3.2, **full cramming** suffers from the *"last 1% problem"*: even at 97–99% teacher-forcing accuracy, the remaining errors concentrate at the earliest token positions. A single early-token mismatch causes autoregressive collapse - greedy convergence drops below 2%. Only **progressive cramming** reliably achieves 100% reconstruction accuracy from the compression embedding, which is a necessary condition for usable autoregressive generation.


   
 

**More accurate estimation of embedding capacity.**

Progressive cramming achieves an Information Gain of $4391 \pm 1408$, compared to $3292 \pm 320$ for full cramming (Kuratov et al.). The higher mean indicates that full cramming underestimates the true embedding capacity. The wider standard deviation is expected: progressive cramming pushes closer to the actual capacity limit, where sample-to-sample variation naturally increases.


   
 

**Insights into the optimization landscape.**

The progressive cramming setup allows us to trace the optimization trajectory, which revealed that it is inherently low-dimensional. This insight led to the low-rank projection technique that improves the cramming process, as demonstrated in [Table 6](https://drive.google.com/file/d/148Cxg2jOx_-MIH5XC0I-oPbwVLG0Ktiz/view?usp=sharing) (see our response to W1).


   
 

**50 samples**

We note that Kuratov et al. (full cramming) also evaluate on 50 samples. Our experimental setup therefore follows established practice in this line of work.


**Erratum: Downstream evaluation bug**

After submission, we discovered a bug in the perplexity computation for the HellaSwag and ARC benchmarks. The compression embedding introduced an off-by-one token shift that was not accounted for during likelihood scoring, causing the near-random downstream accuracy reported in the original paper. After correction, HellaSwag accuracy under cramming ranges from 34%-38% for most models, representing a consistent but moderate drop from baseline in line with our central claim that high reconstruction accuracy does not imply preservation of downstream-relevant semantics. Corrected results are presented in [Table 7](https://drive.google.com/file/d/1DIjiTpyIqQ4xuW0k9Q7mjCOFuXUPz2YX/view?usp=sharing). See also response to Reviewer $\texttt{neRn}$, Q3.
