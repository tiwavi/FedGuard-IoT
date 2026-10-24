import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curves():
    # Data manually extracted from your Phase 2 and Phase 3 console outputs
    rounds_standard = [0, 1, 2, 3, 4, 5]
    acc_standard = [74.76, 70.94, 73.37, 77.14, 79.74, 81.82]
    
    rounds_privacy = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    acc_privacy = [30.09, 70.94, 70.94, 70.94, 71.10, 72.63, 74.90, 76.65, 79.88, 82.01, 84.63]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Plot Standard FL
    plt.plot(rounds_standard, acc_standard, marker='o', linestyle='-', linewidth=2, 
             color='blue', label='Standard FedAvg (No Privacy)')
    
    # Plot Privacy FL
    plt.plot(rounds_privacy, acc_privacy, marker='s', linestyle='--', linewidth=2, 
             color='red', label='Privacy-Preserving FedAvg (Noise = 0.01)')
    
    # Formatting
    plt.title('Federated Learning: Accuracy vs. Communication Rounds', fontsize=14, fontweight='bold')
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Global Model Accuracy (%)', fontsize=12)
    plt.xticks(range(0, 11))
    plt.ylim(20, 100)
    plt.legend(loc='lower right', fontsize=11)
    
    # Add baseline reference line
    plt.axhline(y=90.44, color='green', linestyle=':', label='Centralized Baseline (90.44%)')
    plt.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('federated_learning_comparison.png', dpi=300)
    print("✅ Visualization saved as 'federated_learning_comparison.png'")
    plt.show()

if __name__ == "__main__":
    plot_learning_curves()