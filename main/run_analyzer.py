import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    eval_path = os.path.join(args.run_dir, "eval_results.json")
    with open(eval_path, "r") as f:
        eval_results = json.load(f)

    # eval_results[level][problem_id][sample_id]["correctness"]
    correct_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
    # good_problems = []
    for level in eval_results:
        for problem_id in eval_results[level]:
            samples = eval_results[level][problem_id]
            total = len(samples)
            correct = sum(1 for sample_id in samples if samples[sample_id].get("correctness", False))
            correct_counts[correct] += 1
            # if correct >= 5:
                # good_problems.append(int(problem_id))

            # print(f"Level {level} Problem {problem_id}: {correct}/{total} correct")
    
    # Summary: for 0, 1, ..., 8, print how many problems have that many correct samples
    for correct in correct_counts:
        print(f"Correct {correct}/8: {correct_counts[correct]} problems")
    
    import matplotlib.pyplot as plt

    # Prepare data for plotting
    correct_values = list(correct_counts.keys())
    problem_counts = [correct_counts[c] for c in correct_values]

    plt.figure(figsize=(8, 5))
    plt.bar(correct_values, problem_counts, color='skyblue')
    plt.xlabel('Number of Correct Samples (out of 8)')
    plt.ylabel('Number of Problems')
    plt.title('Distribution of Correct Samples')
    plt.xticks(correct_values)
    plt.tight_layout()
    plt.savefig(os.path.join(args.run_dir, "correct_counts.png"))

    # print(f"Good problems: {good_problems}")


if __name__ == "__main__":
    main()
