
import json
import matplotlib.pyplot as plt
import os

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['history']['steps'], data['history']['val_losses']

def main():
    file1 = 'plots/20M.json'
    file2 = 'plots/20M_test.json'
    output_file = 'plots/compare_20M.png'

    if not os.path.exists(file1):
        print(f"Error: {file1} not found.")
        return
    if not os.path.exists(file2):
        print(f"Error: {file2} not found.")
        return

    steps1, losses1 = load_data(file1)
    steps2, losses2 = load_data(file2)

    plt.figure(figsize=(10, 6))
    plt.plot(steps1, losses1, label='20M', marker='o')
    plt.plot(steps2, losses2, label='20M_test', marker='x')

    plt.title('Validation Loss Comparison: 20M vs 20M_test')
    plt.xlabel('Steps')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(output_file)
    print(f"Comparison plot saved to {output_file}")

if __name__ == "__main__":
    main()
