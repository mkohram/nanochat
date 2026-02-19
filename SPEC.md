**The GDH (Gated Double Helix) Architecture Specification**

**Version:** 1.4 (Definitive Parallel-Optimized Blueprint)

**Core Philosophy:** A globally shared Latent Sidecar ($S$) interacts with layer-specific Local Streams ($L$) via LoRA-adapted Read-Process-Write cycles. By decoupling **slot-query routing** in the Write Phase from the recurrent state (while write content remains prefix-conditioned through causal processing) and removing the forget gate, the Sidecar updates become purely additive. This eliminates the sequential token bottleneck, allowing the architecture to leverage high-speed parallel prefix scans (cumulative sums) over the sequence dimension, while RMSNorm stabilizes the unbounded memory growth.

### ---

**1\. Notation & Hyperparameters**

* **Sequence & Depth:** $N$ (Sequence Length), $M$ (Total Layers).  
* **Dimensions:** $D$ (Model Dimension), $R$ (Sidecar Register Count), $r$ (LoRA Rank).  
* **Multi-Head Write:** $h$ (Write Heads), where $d\_h \= D/h$.  
* **State Variables:**  
  * $L^{(i)}\_t \\in \\mathbb{R}^{1 \\times D}$: Local token state at timestep $t$, layer $i$.  
  * $S\_{t, i} \\in \\mathbb{R}^{R \\times D}$: Cumulative Sidecar state at timestep $t$, after layer $i$.  
* **Explicit Constraints:**  
  * $D \\pmod{h} \\equiv 0$  
  * $S\_{0, i} \= \\mathbf{0} \\in \\mathbb{R}^{R \\times D}$ for all $i \\in \\{1 \\dots M\\}$ (initial sequence state).  
  * $S\_{t, 0} \= \\mathbf{0}$ for all $t$ (layer-0 sidecar anchor for the parallel layer scan in this v1.4 formulation).

### ---

**2\. The Parameter Taxonomy & Scopes**

Weights are strictly partitioned by their functional phase (Read vs. Write). Within each phase, parameters are either Global (shared across all $M$ layers) or Local (specific to layer $i$).

#### **A. Read Phase Parameters**

*The local token queries the global Sidecar state.*

* **Global (Shared across all $M$ layers):**  
  * $W^{K\\\_read}\_{global}, W^{V\\\_read}\_{global} \\in \\mathbb{R}^{D \\times D}$ (Projects Sidecar state into Keys and Values).  
* **Local (Specific to layer $i$):**  
  * $W\_{base, i}^{Q\\\_read} \\in \\mathbb{R}^{D \\times D}$ (Base Query projection; shared across all tokens within layer $i$).  
  * $A\_i^{Q\\\_read} \\in \\mathbb{R}^{D \\times r}, B\_i^{Q\\\_read} \\in \\mathbb{R}^{r \\times D}$ (LoRA Query Adapters).  
  * $W^O\_{read, i} \\in \\mathbb{R}^{D \\times D}$ (Output mixer).  
  * $W^g\_{eff, i} \\in \\mathbb{R}^{D \\times 1}$ (Scalar Read Gate).

#### **B. Write Phase Parameters**

*The permanent slot embeddings query the local token's output.*

* **Global (Shared across all $M$ layers):**  
  * $E\_{slots} \\in \\mathbb{R}^{R \\times D}$ (Permanent, learnable slot addresses).  
  * $W^{Q\\\_write}\_{global} \\in \\mathbb{R}^{D \\times D}$ (Projects slot embeddings into Queries).  
* **Local (Specific to layer $i$):**  
  * $W\_{base, i}^{K\\\_write}, W\_{base, i}^{V\\\_write} \\in \\mathbb{R}^{D \\times D}$ (Base Key/Value projections; shared across all tokens within layer $i$).  
  * $A\_i^{K\\\_write}, B\_i^{K\\\_write}$ and $A\_i^{V\\\_write}, B\_i^{V\\\_write}$ (LoRA Key/Value Adapters).  
  * $W^O\_{write, i} \\in \\mathbb{R}^{D \\times D}$ (Output mixer).

**Effective Weight Formula (Pre-computed for inference):**

$$W\_{eff, i}^{Q\\_read} = W\_{base, i}^{Q\\_read} + A\_i^{Q\\_read} B\_i^{Q\\_read}$$
$$W\_{eff, i}^{K\\_write} = W\_{base, i}^{K\\_write} + A\_i^{K\\_write} B\_i^{K\\_write}$$
$$W\_{eff, i}^{V\\_write} = W\_{base, i}^{V\\_write} + A\_i^{V\\_write} B\_i^{V\\_write}$$

### ---

**3\. Initialization Rules (Mandatory)**

1. **Zero-Init Mixers:** Initialize $W^O\_{read, i}$ and $W^O\_{write, i}$ to strictly $0$. This ensures the Sidecar starts completely disconnected, and the model behaves exactly like a standard Transformer at initialization, preventing early training instability.  
2. **Symmetry Breaking:** Initialize $E\_{slots}$ from a standard normal distribution $\\mathcal{N}(0, 1)$ to ensure immediate routing differentiation across the $R$ slots.  
3. **State Reset:** Because the forget gate is removed, the Sidecar state $S$ must be manually reset to $\\mathbf{0}$ at document boundaries or predefined chunk intervals during training to prevent unbounded numerical saturation.

### ---

**4\. The Forward Pass (Layer $i$)**

The computations below describe the operations for a single timestep $t$, but because recurrent state dependencies within the Write Phase have been severed, **all $N$ tokens in the sequence are computed in parallel** for a given layer. The sequence accumulates horizontally over $t$, and the model executes vertically layer-by-layer over $M$.

#### **Phase I: The Read Phase**

*The token fetches context from the previously accumulated Sidecar state of the prior layer.*

1. **Local Query Generation:**  
   $$x\_{read} \= \\text{RMSNorm}(L^{(i)}\_t) \\quad \\in \\mathbb{R}^{1 \\times D}$$  
   $$Q\_{loc} \= x\_{read} W\_{eff, i}^{Q\\\_read} \\quad \\in \\mathbb{R}^{1 \\times D}$$  
2. **Normalized Global Retrieval:**  
   $$\\hat{S}\_{t, i-1} \= \\text{RMSNorm}(S\_{t, i-1}) \\quad \\in \\mathbb{R}^{R \\times D}$$  
   $$K\_{mem} \= \\hat{S}\_{t, i-1} W^{K\\\_read}\_{global} \\quad \\in \\mathbb{R}^{R \\times D}$$  
   $$V\_{mem} \= \\hat{S}\_{t, i-1} W^{V\\\_read}\_{global} \\quad \\in \\mathbb{R}^{R \\times D}$$  
   $$Z\_{read} \= \\text{Softmax}\\left( \\frac{Q\_{loc} K\_{mem}^T}{\\sqrt{D}} \\right) V\_{mem} \\quad \\in \\mathbb{R}^{1 \\times D}$$  
3. **Scalar Gated Injection:**  
   $$g\_{read} \= \\sigma\\left( x\_{read} W^g\_{eff, i} \\right) \\quad \\in \\mathbb{R}^{1 \\times 1}$$  
   $$\\tilde{L}^{(i)}\_t \= L^{(i)}\_t \+ g\_{read} \\left( Z\_{read} W^O\_{read, i} \\right)$$

#### **Phase II: The Process Phase**

*Standard Pre-Norm Transformer processing.*

$$\\hat{L}^{(i)}\_t \= \\tilde{L}^{(i)}\_t \+ \\text{SelfAttention}(\\text{RMSNorm}(\\tilde{L}^{(i)}\_t); W\_{self})$$

$$L^{(i, \\text{out})}\_t \= \\hat{L}^{(i)}\_t \+ \\text{FFN}(\\text{RMSNorm}(\\hat{L}^{(i)}\_t); W\_{mlp})$$

#### **Phase III: The Parallel Write Phase**

*Tokens generate prefix-conditioned update proposals (each token's write depends on tokens $1\dots t$ through causal Phase II), routed by the static $E\_{slots}$.*

1. **Local Vector Generation:**  
   $$x\_{write} \= \\text{RMSNorm}(L^{(i, \\text{out})}\_t) \\quad \\in \\mathbb{R}^{1 \\times D}$$  
   $$K\_{upd} \= x\_{write} W\_{eff, i}^{K\\\_write} \\quad \\in \\mathbb{R}^{1 \\times D}$$  
   $$V\_{upd} \= x\_{write} W\_{eff, i}^{V\\\_write} \\quad \\in \\mathbb{R}^{1 \\times D}$$  
2. **Static Slot Queries:**  
   $$Q\_{slots} \= E\_{slots} W^{Q\\\_write}\_{global} \\quad \\in \\mathbb{R}^{R \\times D}$$  
3. **Multi-Head Parallel Routing:**  
   Split $K\_{upd}, V\_{upd}$, and $Q\_{slots}$ into $h$ heads. For head $j \\in \\{1 \\dots h\\}$:  
   $$\\alpha\_j \= \\text{Softmax}\_{dim=0}\\left( \\frac{Q\_{slots, j} K\_{upd, j}^T}{\\sqrt{d\_h}} \\right) \\quad \\in \\mathbb{R}^{R \\times 1}$$  
   $$\\Delta\_{raw, j} \= \\alpha\_j V\_{upd, j} \\quad \\in \\mathbb{R}^{R \\times d\_h}$$  
4. **Mixing and Prefix Scan (The cumsum step):**  
   Concatenate all heads to form $\\Delta\_{raw} \\in \\mathbb{R}^{R \\times D}$. The proposed update for token $t$ is:  
   $$\\Delta\_{t, i} \= \\Delta\_{raw} W^O\_{write, i}$$  
   **Sequence-Level Parallel Accumulation:**  
   The Sidecar state at layer $i$ incorporates the prior layer's state plus the cumulative sum of all token writes at the current layer up to time $t$:  
   $$S\_{t, i} \= S\_{t, i-1} \+ \\sum\_{k=1}^{t} \\Delta\_{k, i}$$

### ---

**5\. Asymptotic Complexity**

* **Local Processing (Self-Attention):** $O(M \\cdot N^2 \\cdot D)$ per sequence.  
* **Sidecar Mechanics:** $O(M \\cdot N \\cdot R \\cdot D^2)$ parallel operations for computing Read/Write projections.  
* **Total Space Complexity (Inference):** strictly $O(M \\cdot N \\cdot D)$ for the standard KV cache, plus a fixed $O(R \\cdot D)$ bound for the Latent Sidecar register.