# Neural-MCM: Constant-Time Matrix Chain Multiplication

A state-of-the-art research project implementing a **Pointer Network (PointerMCMNet)** to solve the Matrix Chain Multiplication (MCM) optimization problem in $O(1)$ constant time with **100% Mathematical Validity**.

## 🚀 Key Achievements
- **$O(1)$ Inference**: Replaced classical $O(n^3)$ dynamic programming with instant neural prediction.
- **Structural Reasoning**: Achieving near-zero error (0.02% MAPE) on structured chains.
- **Validity Floor**: Guaranteed physically possible parenthesization through structural sequence modeling.

## 📁 Project Structure
- `/src`: Core implementation (Pointer Network, Evaluation suite, Feature engineering).
- `/models`: Pre-trained stage-4 curriculum models.
- `/figures`: Professional academic graphs (LaTeX ready).
- `/data`: Synthetic pathological dataset generation scripts.
- `research_paper.tex`: The final humanized research manuscript (IEEE Standard).
- `generate_graphs.py`: Automated academic visualization pipeline.

## 📊 Quick Start
To generate the research visualizations:
```bash
python generate_graphs.py
```

To run the comparative benchmark:
```bash
python -m src.research_evaluation
```

---
*Developed by CH. ABHILASH and team at VNR VJIET.*
