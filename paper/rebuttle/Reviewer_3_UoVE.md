We are grateful for the detailed and insightful review. We address the raised concerns.


   
 
# Weaknesses:

**W1. Attention-hijacking narrative**

We conducted attention knockout experiments (see answer to reviewer $\texttt{neRn}$, W2):
the forward/reverse knockout asymmetry shows that early-layer interactions with the compression embedding drive downstream degradation, while late-layer attention mass has minimal causal impact. And observed attention concentration possibly is a symptom rather than the cause.


    

---

   
 

**Erratum: Downstream evaluation bug**

After submission, we discovered a bug in the perplexity computation for the HellaSwag and ARC benchmarks. The compression embedding introduced an off-by-one token shift that was not accounted for during likelihood scoring, causing the near-random downstream accuracy reported in the original paper. After correction, HellaSwag accuracy under cramming ranges from 34%-38% for most models, representing a consistent but moderate drop from baseline in line with our central claim that high reconstruction accuracy does not imply preservation of downstream-relevant semantics. Corrected results are presented in [Table 7](https://drive.google.com/file/d/1DIjiTpyIqQ4xuW0k9Q7mjCOFuXUPz2YX/view?usp=sharing). See also response to Reviewer $\texttt{neRn}$, Q3.


---


We thank the reviewer for the valuable feedback and the opportunity to clarify.

   
 



**On the downstream evaluation bug.**

The bug affected absolute accuracy values (corrected HellaSwag: 34-38% instead of ~25%), but the core finding holds: cramming consistently degrades downstream performance despite high reconstruction accuracy. We also evaluated on 5-shot MMLU (see response to Reviewer $\texttt{eppq}$, W3), where full-prefix compression causes complete generation failure. This confirms that the degradation is real and not an artifact of the bug or evaluation protocol.

   
 


**On paper reorganization.**

We agree that the presentation can be improved and will reorganize the manuscript in the revised version. We believe that the core contributions - the identification of the "last 1%" problem, progressive cramming, and the systematic evaluation of downstream performance under compression - provide sufficient evidence to support the paper's conclusions.