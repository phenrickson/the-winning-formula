"""
Soccer Pass Sequence Simulation
This script simulates sequences of soccer passes following a negative binomial distribution
and visualizes the results.
"""

import numpy as np
from scipy import stats
import polars as pl
from plotnine import *


def simulate_pass_sequences(n_simulations=10000):
    """
    Simulate pass sequences using negative binomial distribution.
    Returns counts for sequences of 1-8 passes.

    Parameters:
    - n_simulations: number of sequences to simulate

    Returns:
    - Array of counts for each pass length (1-8)
    """
    # Parameters for negative binomial to match historical soccer data
    # Based on 42 First Division matches 1957-58 data
    n, p = 0.8, 0.45  # Adjusted to match the rapid decay in pass frequency

    # Generate sequences
    sequences = stats.nbinom.rvs(n=n, p=p, size=n_simulations)

    # Clip sequences to maximum 9 passes (including 9 and over category)
    sequences = np.clip(sequences, 0, 9)  # Now including 0 passes

    # Count occurrences of each sequence length
    unique, counts = np.unique(sequences, return_counts=True)

    # Ensure we have counts for all lengths 0-9
    all_counts = np.zeros(10)  # 0 to 9 (where 9 represents "9 and over")
    all_counts[unique] = counts

    return all_counts


def plot_pass_distribution(df):
    """
    Create a bar plot showing the distribution of pass sequences using plotnine.

    Parameters:
    - df: polars DataFrame containing pass sequence data
    """
    # Convert polars DataFrame to pandas for plotnine
    plot_df = df.to_pandas()

    # Create plot
    # Round percentages for labels
    plot_df["percentage_label"] = plot_df["percentage"].round(1).astype(str) + "%"

    plot = (
        ggplot(plot_df, aes(x="passes", y="count"))
        + geom_bar(stat="identity", fill="steelblue", alpha=0.8)
        + geom_text(
            aes(label="percentage_label"),
            va="bottom",
            position=position_dodge(width=0.9),
            size=8,
            format_string="{}",
        )
        + labs(
            x="Number of Passes in Sequence",
            y="Frequency",
            title="Distribution of Pass Sequences in Soccer",
        )
        + theme_minimal()
        + theme(figure_size=(12, 6))
    )

    # Save plot to figures directory
    plot.save("figures/pass_distribution.png", dpi=300)


def main():
    """
    Main execution function.
    """
    # Simulate pass sequences
    pass_counts = simulate_pass_sequences()

    # Create a polars DataFrame for summary statistics
    total_sequences = np.sum(pass_counts)
    df = pl.DataFrame(
        {
            "passes": [str(i) if i < 9 else "9+" for i in range(10)],
            "count": pass_counts,
            "percentage": (pass_counts / total_sequences) * 100,
        }
    )

    print("\nPass Sequence Summary:")
    print("-" * 50)
    print(
        df.select(
            [
                pl.col("passes").alias("Number of Passes"),
                pl.col("count").round(0).alias("Count"),
                pl.col("percentage").round(1).alias("Percentage"),
            ]
        )
    )

    # Plot the distribution
    plot_pass_distribution(df)


if __name__ == "__main__":
    main()
