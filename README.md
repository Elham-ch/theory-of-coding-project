
## theory-of-coding-project
## Simulation of Huffman compression + Hamming error correction with BER plotting and Shannon threshold comparison. Built for Theory of Coding course.
=======
## Theory of Coding Project — Huffman + Hamming Coding with Shannon Threshold Visualization

This project combines **Huffman compression** and **Hamming error correction** to simulate a **noisy communication channel** and evaluate performance against the **Shannon limit**. It visualizes how the **bit error rate (BER)** changes based on the probability of bit flips, using the expression:

\[
x = \frac {1 - p}{p}
\]

---

###  Objectives

- Compress a random symbol sequence using **Huffman Coding**
- Encode the compressed bitstream using **Hamming (n, k) block code**
- Inject noise into the codewords with **bit-flip probability `p`**
- Correct single-bit errors using **Hamming decoding**
- **Calculate BER** and visualize it against the **Shannon threshold**

---

###  How to Run

```bash
python hamming.py
```

The program will prompt:

| Prompt | Description | Example |
|--------|-------------|---------|
| `Enter length of random string` | Number of source symbols to generate | `50000` |
| `Enter r for Hamming code` | Determines block size `n = 2^r - 1` | `3`, `4`, or `5` |

---

### Code Structure

| Section | Functionality |
|---------|---------------|
| **Huffman Coding** | Build frequency tree, assign bit codes, compress source |
| **Hamming Matrix Construction** | Generate `G` (Generator) and `H` (Parity) matrices dynamically |
| **Noise Injection** | Flip bits based on probability `p` |
| **Hamming Decoding** | Compute syndrome and correct 1-bit errors |
| **BER Measurement** | Compare original vs. decoded bitstreams |
| **Plotting** | Draw BER curve vs. `(1 - p) / p` using log scale |

---

###  Example Output

-  Huffman Codes Generated  
-  G & H Matrices Printed  
-  Random String Created  
-  Noise Added and Corrected  
-  Final Decoded Message  
-  BER Curve Plotted

The final plot **shows how reliable the channel is compared to the Shannon limit**.

---

###  Requirements

Install dependencies with:

```bash
pip install numpy scipy matplotlib
```

---

###  Theoretical Background

| Concept | Description |
|---------|-------------|
| **Huffman Coding** | Optimal prefix coding for compression |
| **Hamming Code** | Linear code that detects and corrects 1-bit errors |
| **Shannon Limit** | Defines theoretical maximum channel capacity |

---

###  License / Usage

This project was developed for academic purposes under the **Theory of Coding** course.  
You may reuse or extend it for learning and research with proper credit.

---

###  Author

**Eli**

Theory of Coding – University Project

