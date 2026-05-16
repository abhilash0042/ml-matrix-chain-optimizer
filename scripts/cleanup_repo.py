import os
import shutil

FILES_TO_REMOVE = [
    "ABSTRACT.md",
    "FINAL_REPORT.md",
    "LITERATURE_REVIEW.md",
    "PROBLEM_STATEMENT.md",
    "RESEARCH_REPORT.md",
    "FINAL_RESULTS.txt",
    "project_analysis.txt",
    "pointer_network_mcm_explanation.txt",
    "transformer_mcm_explanation.txt",
    "research_paper.tex",
    "conference_101719.tex",
    "generate_10_examples.py",
    "generate_graphs.py",
    "src/api.py",
    "src/inference.py",
    "src/predict.py",
    "src/train.py",
    "src/research_evaluation.py",
    "src/evaluate_pointer.py",
    "src/evaluate_transformer_v2.py",
    "src/evaluate_tree_models.py",
    "src/train_pointer.py",
    "src/train_transformer_v2.py",
    "src/train_tree_models.py",
    "src/test_pointer_pipeline.py",
    "src/test_tree_inference.py",
    "src/verify_data.py",
    "src/benchmark_hybrid.py",
    "src/interactive_inference.py",
    "data/generate_data_v3.py",
    "data/feature_extractor_v4.py",
    "data/feature_extractor_v3.py",
    "data/pointer_features.py",
    "data/enrich_splits.py",
    "benchmarks/diag_log.txt",
    "benchmarks/eval_log.txt",
    "benchmarks/evaluation_results.txt",
    "benchmarks/final_report.md",
    "benchmarks/model_verdict.md",
    "models/pointer_stage1.pth",
    "models/pointer_stage2.pth",
    "models/pointer_stage3.pth",
    "models/pointer_stage4.pth",
    "models/transformer_v2_stage1.pth",
    "models/transformer_v2_stage2.pth",
    "models/transformer_v2_checkpoint.pth",
    "placeholder_feature_importance.png",
    "placeholder_latency.png",
    "placeholder_validity.png"
]

def cleanup():
    print("Starting repository cleanup...")
    count = 0
    for f in FILES_TO_REMOVE:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"Removed: {f}")
                count += 1
            except Exception as e:
                print(f"Error removing {f}: {e}")
    
    print(f"\nCleanup complete. Removed {count} redundant files.")
    print("Your repository is now organized into docs/, paper/, scripts/, and src/ subdirectories.")

if __name__ == "__main__":
    cleanup()
