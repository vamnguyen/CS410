import argparse
from src.experiment import run_full_experiment

def main():
    parser = argparse.ArgumentParser(description='Run DE and CEM experiments')
    parser.add_argument('--mssv', type=int, required=True, help='Student ID to use as base for random seeds')
    args = parser.parse_args()
    
    print(f"Running experiments with MSSV: {args.mssv}")
    all_results, tables = run_full_experiment(args.mssv)
    
    print("Experiments completed successfully!")
    print("Results saved in 'results' directory")
    print("Figures saved in 'figures' directory")
    print("Log files saved in 'logs' directory")

if __name__ == "__main__":
    main() 