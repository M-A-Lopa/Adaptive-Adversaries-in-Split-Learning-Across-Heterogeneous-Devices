import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

RESULTS_DIR  = './results'
CIFAR10_CSV  = f'{RESULTS_DIR}/vanilla_sl_results_CIFAR10.csv'
MNIST_CSV    = f'{RESULTS_DIR}/vanilla_sl_results_MNIST.csv'
SAVE_PATH    = f'{RESULTS_DIR}/comparison_cifar10_vs_mnist.png'


def load_results(path, dataset_name):

    if not os.path.exists(path):
        raise FileNotFoundError(f"\nERROR: {path} not found.\n"
            f"Make sure you have run main.py with DATASET='{dataset_name}' "
            f"and RUN_TRAINING=True before running this script.")
    df = pd.read_csv(path)
    required = {'epoch', 'train_loss', 'train_accuracy', 'test_accuracy'}

    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns. Found: {df.columns.tolist()}")
    print(f"Loaded {dataset_name}: {len(df)} epochs")
    return df


def print_summary(df_cifar, df_mnist):

    print("\n" + "=" * 60)
    print("   FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'CIFAR-10':>12} {'MNIST':>12}")
    print("-" * 60)

    metrics = [
        ('Best Test Accuracy (%)',
         df_cifar['test_accuracy'].max(),
         df_mnist['test_accuracy'].max()),
        ('Final Test Accuracy (%)',
         df_cifar['test_accuracy'].iloc[-1],
         df_mnist['test_accuracy'].iloc[-1]),
        ('Final Train Accuracy (%)',
         df_cifar['train_accuracy'].iloc[-1],
         df_mnist['train_accuracy'].iloc[-1]),
        ('Final Train Loss',
         df_cifar['train_loss'].iloc[-1],
         df_mnist['train_loss'].iloc[-1]),
        ('Total Epochs',
         len(df_cifar),
         len(df_mnist)),
]

    for label, c_val, m_val in metrics:
        print(f"  {label:<28} {c_val:>12.4f} {m_val:>12.4f}")

    print("=" * 60)


def plot_comparison(df_cifar, df_mnist):

    epochs_c = df_cifar['epoch'].values
    epochs_m = df_mnist['epoch'].values

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    
    COLOR_CIFAR_TRAIN = '#1f77b4'   
    COLOR_CIFAR_TEST  = '#aec7e8'  
    COLOR_MNIST_TRAIN = '#d62728'  
    COLOR_MNIST_TEST  = '#ffbb78'  

    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs_c, df_cifar['train_loss'],
             color=COLOR_CIFAR_TRAIN, linewidth=2, label='CIFAR-10')
    ax1.plot(epochs_m, df_mnist['train_loss'],
             color=COLOR_MNIST_TRAIN, linewidth=2, label='MNIST')
    ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_c, df_cifar['test_accuracy'],
             color=COLOR_CIFAR_TRAIN, linewidth=2, label='CIFAR-10')
    ax2.plot(epochs_m, df_mnist['test_accuracy'],
             color=COLOR_MNIST_TRAIN, linewidth=2, label='MNIST')
    ax2.set_title('Test Accuracy', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs_c, df_cifar['train_accuracy'],
             color=COLOR_CIFAR_TRAIN, linewidth=2, label='CIFAR-10 Train')
    ax3.plot(epochs_c, df_cifar['test_accuracy'],
             color=COLOR_CIFAR_TEST,  linewidth=2,
             linestyle='--', label='CIFAR-10 Test')
    ax3.plot(epochs_m, df_mnist['train_accuracy'],
             color=COLOR_MNIST_TRAIN, linewidth=2, label='MNIST Train')
    ax3.plot(epochs_m, df_mnist['test_accuracy'],
             color=COLOR_MNIST_TEST,  linewidth=2,
             linestyle='--', label='MNIST Test')
    ax3.set_title('Train vs Test Accuracy', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])

    categories    = ['Train Acc\n(Final)', 'Test Acc\n(Final)', 'Test Acc\n(Best)']
    cifar_values  = [df_cifar['train_accuracy'].iloc[-1], df_cifar['test_accuracy'].iloc[-1], df_cifar['test_accuracy'].max()]
    mnist_values  = [df_mnist['train_accuracy'].iloc[-1], df_mnist['test_accuracy'].iloc[-1], df_mnist['test_accuracy'].max()]

    x      = np.arange(len(categories))
    width  = 0.32

    bars_c = ax4.bar(x - width/2, cifar_values, width, color=COLOR_CIFAR_TRAIN, label='CIFAR-10', edgecolor='white', linewidth=0.8)
    bars_m = ax4.bar(x + width/2, mnist_values, width, color=COLOR_MNIST_TRAIN, label='MNIST', edgecolor='white', linewidth=0.8)

    for bar in bars_c:
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.4,
                 f'{bar.get_height():.1f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars_m:
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.4,
                 f'{bar.get_height():.1f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax4.set_title('Final Accuracy Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=10)
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_ylim(0, 110)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Vanilla Split Learning — CIFAR-10 vs MNIST Comparison\n''Simple CNN | Cut Layer 2 | 50 Epochs',fontsize=14, fontweight='bold', y=1.01)

    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nComparison plot saved → {SAVE_PATH}")

if __name__ == "__main__":
    print("=" * 60)
    print("   DATASET COMPARISON — CIFAR-10 vs MNIST")
    print("=" * 60)

    df_cifar = load_results(CIFAR10_CSV, 'CIFAR10')
    df_mnist  = load_results(MNIST_CSV,  'MNIST')

    print_summary(df_cifar, df_mnist)

    plot_comparison(df_cifar, df_mnist)