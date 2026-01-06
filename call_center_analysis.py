"""
Call Center Simulation - Interactive Analysis Tool
==================================================
Program untuk simulasi dan analisis call center dengan multiple agents
Menggunakan SimPy untuk Discrete Event Simulation

Usage:
    python call_center_analysis.py [num_agents] [sim_time]
    Contoh: python call_center_analysis.py 2 1000
"""

import pandas as pd
import numpy as np
from simpy import Environment, Resource
import sys
import os
from pathlib import Path

# Configuration
RANDOM_SEED = 42
DEFAULT_SIM_TIME = 1000
DEFAULT_NUM_AGENTS = 1

# Get script directory
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "dataset" / "simulated_call_centre.csv"


def load_dataset():
    """Load call centre dataset dan hitung parameters"""
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"✓ Dataset loaded: {len(df)} records")

        # Calculate inter-arrival times
        df['CALL_TIME'] = pd.to_datetime(df['CALL_TIME'])
        df = df.sort_values('CALL_TIME')
        arrivals = df['CALL_TIME'].diff().dt.total_seconds() / 60
        inter_arrival_mean = arrivals.mean()
        inter_arrival_std = arrivals.std()

        # Calculate service times
        duration_mean = df['DURATION'].mean()
        duration_std = df['DURATION'].std()

        return {
            'dataframe': df,
            'inter_arrival_mean': inter_arrival_mean,
            'inter_arrival_std': inter_arrival_std,
            'duration_mean': duration_mean,
            'duration_std': duration_std
        }
    except FileNotFoundError:
        print(f"✗ Dataset tidak ditemukan di {DATASET_PATH}")
        print("  Menggunakan default parameters...")
        return {
            'dataframe': None,
            'inter_arrival_mean': 2.5,
            'inter_arrival_std': 2.0,
            'duration_mean': 3.5,
            'duration_std': 2.0
        }


def customer_process(env, customer_id, agent, records, arrival_time, service_duration):
    """Process untuk setiap customer"""
    # Customer arrives
    arrival = env.now

    # Request agent
    with agent.request() as request:
        # Wait for agent to be free
        yield request
        start_service = env.now

        # Service duration
        yield env.timeout(service_duration)
        finish_service = env.now

    # Record metrics
    records.append({
        'customer_id': customer_id,
        'arrival_time': arrival_time,
        'start_service': start_service,
        'finish_service': finish_service,
        'waiting_time': start_service - arrival,
        'service_duration': service_duration,
        'system_time': finish_service - arrival
    })


def run_simulation(num_agents, sim_time, seed, dataset_params):
    """Jalankan simulasi"""
    np.random.seed(seed)
    env = Environment()
    agent = Resource(env, capacity=num_agents)
    records = []
    customer_id = [0]  # Mutable counter

    def customer_generator():
        """Generate customers"""
        while True:
            inter_arrival = np.random.exponential(
                dataset_params['inter_arrival_mean'])
            yield env.timeout(inter_arrival)

            customer_id[0] += 1
            service_duration = max(0.1, np.random.normal(
                dataset_params['duration_mean'],
                dataset_params['duration_std']
            ))

            env.process(customer_process(
                env, customer_id[0], agent, records,
                env.now, service_duration
            ))

    # Start generator
    env.process(customer_generator())

    # Run simulation
    env.run(until=sim_time)

    # Create dataframe
    df = pd.DataFrame(records)

    if len(df) == 0:
        return df, 0

    # Calculate metrics
    total_busy_time = (df['service_duration'].sum())
    utilization = (total_busy_time / (num_agents * sim_time)) * 100

    return df, utilization


def print_metrics(df, utilization, scenario_name, num_agents):
    """Print simulation results"""
    if len(df) == 0:
        print(f"\n✗ {scenario_name}: No data")
        return

    print(f"\n{'='*60}")
    print(f"{scenario_name} ({num_agents} agent{'s' if num_agents > 1 else ''})")
    print('='*60)
    print(f"Total calls processed: {len(df)}")
    print(f"Utilization: {utilization:.2f}%")
    print(f"\nWaiting Time (menit):")
    print(f"  Average: {df['waiting_time'].mean():.2f}")
    print(f"  Min: {df['waiting_time'].min():.2f}")
    print(f"  Max: {df['waiting_time'].max():.2f}")
    print(f"  Std Dev: {df['waiting_time'].std():.2f}")
    print(f"\nSystem Time (menit):")
    print(f"  Average: {df['system_time'].mean():.2f}")
    print(f"  Median: {df['system_time'].median():.2f}")


def compare_scenarios(df_a, util_a, df_b, util_b, num_agents_a, num_agents_b):
    """Compare two scenarios"""
    print(f"\n\n{'='*60}")
    print("PERBANDINGAN SKENARIO")
    print('='*60)

    comparison = {
        'Metric': [
            'Total Calls',
            'Avg Waiting Time (min)',
            'Max Waiting Time (min)',
            'Avg System Time (min)',
            'Utilization (%)',
            'Throughput (calls/hour)'
        ],
        f'Skenario A ({num_agents_a} agent)': [
            len(df_a),
            f"{df_a['waiting_time'].mean():.2f}",
            f"{df_a['waiting_time'].max():.2f}",
            f"{df_a['system_time'].mean():.2f}",
            f"{util_a:.2f}",
            f"{len(df_a)/1000*60:.2f}"
        ],
        f'Skenario B ({num_agents_b} agents)': [
            len(df_b),
            f"{df_b['waiting_time'].mean():.2f}",
            f"{df_b['waiting_time'].max():.2f}",
            f"{df_b['system_time'].mean():.2f}",
            f"{util_b:.2f}",
            f"{len(df_b)/1000*60:.2f}"
        ]
    }

    comparison_df = pd.DataFrame(comparison)
    print("\n" + str(comparison_df.to_string(index=False)))

    # Analysis
    print(f"\n{'─'*60}")
    print("ANALISIS:")
    if len(df_b) > len(df_a):
        increase = ((len(df_b) - len(df_a)) / len(df_a)) * 100
        print(
            f"✓ Dengan {num_agents_b} agent, throughput meningkat {increase:.1f}%")

    wait_improvement = df_a['waiting_time'].mean() - \
        df_b['waiting_time'].mean()
    if wait_improvement > 0:
        print(
            f"✓ Waiting time berkurang {wait_improvement:.2f} menit ({(wait_improvement/df_a['waiting_time'].mean())*100:.1f}%)")
    else:
        print(f"✗ Waiting time bertambah {abs(wait_improvement):.2f} menit")

    if util_b < 100:
        print(f"✓ Sistem stabil dengan utilization {util_b:.2f}%")
    else:
        print(f"✗ SISTEM OVERLOAD! Utilization {util_b:.2f}% > 100%")


def main():
    """Main program"""
    print("\n" + "="*60)
    print("CALL CENTER SIMULATION ANALYSIS")
    print("="*60)

    # Parse arguments
    num_agents_a = int(sys.argv[1]) if len(
        sys.argv) > 1 else DEFAULT_NUM_AGENTS
    num_agents_b = int(sys.argv[2]) if len(
        sys.argv) > 2 else DEFAULT_NUM_AGENTS + 1
    sim_time = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_SIM_TIME

    print(f"\nConfiguration:")
    print(f"  Scenario A: {num_agents_a} agent")
    print(f"  Scenario B: {num_agents_b} agents")
    print(f"  Simulation time: {sim_time} minutes")

    # Load dataset
    dataset_params = load_dataset()

    print(f"\nDataset Parameters:")
    print(
        f"  Inter-arrival mean: {dataset_params['inter_arrival_mean']:.2f} min")
    print(f"  Service time mean: {dataset_params['duration_mean']:.2f} min")

    # Run simulations
    print(f"\n{'─'*60}")
    print("Running simulations...")

    df_a, util_a = run_simulation(
        num_agents_a, sim_time, RANDOM_SEED, dataset_params)
    df_b, util_b = run_simulation(
        num_agents_b, sim_time, RANDOM_SEED + 1, dataset_params)

    # Print results
    print_metrics(df_a, util_a, "SKENARIO A", num_agents_a)
    print_metrics(df_b, util_b, "SKENARIO B", num_agents_b)

    # Comparison
    compare_scenarios(df_a, util_a, df_b, util_b, num_agents_a, num_agents_b)

    # Save results
    output_dir = SCRIPT_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    df_a.to_csv(output_dir / "scenario_a.csv", index=False)
    df_b.to_csv(output_dir / "scenario_b.csv", index=False)

    print(f"\n✓ Results saved to {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
