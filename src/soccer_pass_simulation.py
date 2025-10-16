"""
Soccer Pass Sequence Simulation
This script simulates sequences of soccer passes following a negative binomial distribution
and visualizes the results.
"""

import numpy as np
from scipy import stats
import polars as pl
from plotnine import *
from sklearn.linear_model import LogisticRegression
import pandas as pd


def fit_goal_probability(df_counts, df_goals):
    """
    Fit logistic regression to predict goal probability based on number of passes.

    Parameters:
    - df_counts: polars DataFrame containing pass sequence counts
    - df_goals: polars DataFrame containing goals data

    Returns:
    - Fitted model and prepared data
    """
    # Prepare data for logistic regression
    data = []
    X_data = []
    for i in range(10):
        sequence_type = str(i) if i < 9 else "9+"
        total_sequences = int(
            df_counts.filter(pl.col("passes") == sequence_type)["count"].item()
        )
        goals = int(df_goals.filter(pl.col("passes") == sequence_type)["goals"].item())

        # Create a row for each sequence (1 for goal, 0 for no goal)
        data.extend([1] * goals)
        data.extend([0] * (total_sequences - goals))

        # Add corresponding pass numbers
        X_data.extend([i] * total_sequences)

    # Fit logistic regression
    X = np.array(X_data).reshape(-1, 1)
    y = np.array(data)
    model = LogisticRegression()
    model.fit(X, y)

    return model, X, y


def plot_goal_probabilities(
    df_counts, df_goals_dec, df_goals_inc, df_goals_base, df_goals_act
):
    """
    Plot goal probabilities from logistic regression for all patterns.
    """
    # Create prediction data
    X_pred = np.linspace(0, 9, 100).reshape(-1, 1)

    # Plot decreasing pattern
    model_dec, _, _ = fit_goal_probability(df_counts, df_goals_dec)
    y_pred_dec = model_dec.predict_proba(X_pred)[:, 1]
    plot_data_dec = pd.DataFrame(
        {"passes": X_pred.flatten(), "probability": y_pred_dec}
    )
    plot_dec = (
        ggplot(plot_data_dec, aes(x="passes", y="probability"))
        + geom_line(size=1.5, color="darkred")
        + theme_minimal()
        + theme(figure_size=(12, 6))
        + scale_y_continuous(limits=[0, 0.50])
    )
    plot_dec.save("figures/goal_probabilities_decreasing.png", dpi=300)

    # Plot increasing pattern
    model_inc, _, _ = fit_goal_probability(df_counts, df_goals_inc)
    y_pred_inc = model_inc.predict_proba(X_pred)[:, 1]
    plot_data_inc = pd.DataFrame(
        {"passes": X_pred.flatten(), "probability": y_pred_inc}
    )
    plot_inc = (
        ggplot(plot_data_inc, aes(x="passes", y="probability"))
        + geom_line(size=1.5, color="steelblue")
        + theme_minimal()
        + theme(figure_size=(12, 6))
        + scale_y_continuous(limits=[0, 0.50])
    )
    plot_inc.save("figures/goal_probabilities_increasing.png", dpi=300)

    # Plot uniform pattern
    model_base, _, _ = fit_goal_probability(df_counts, df_goals_base)
    y_pred_base = model_base.predict_proba(X_pred)[:, 1]
    plot_data_base = pd.DataFrame(
        {"passes": X_pred.flatten(), "probability": y_pred_base}
    )
    plot_base = (
        ggplot(plot_data_base, aes(x="passes", y="probability"))
        + geom_line(size=1.5, color="gray")
        + theme_minimal()
        + theme(figure_size=(12, 6))
        + scale_y_continuous(limits=[0, 0.50])
    )
    plot_base.save("figures/goal_probabilities_uniform.png", dpi=300)

    # Plot actual pattern
    model_act, _, _ = fit_goal_probability(df_counts, df_goals_act)
    y_pred_act = model_act.predict_proba(X_pred)[:, 1]
    plot_data_act = pd.DataFrame(
        {"passes": X_pred.flatten(), "probability": y_pred_act}
    )
    plot_act = (
        ggplot(plot_data_act, aes(x="passes", y="probability"))
        + geom_line(size=1.5, color="purple")
        + theme_minimal()
        + theme(figure_size=(12, 6))
        + scale_y_continuous(limits=[0, 0.50])
    )
    plot_act.save("figures/goal_probabilities_actual.png", dpi=300)


def get_goal_rates(base_rate=0.1, pattern="uniform", steepness=1.0):
    """
    Generate goal probabilities for each pass sequence length based on pattern.

    Parameters:
    - base_rate: base probability of scoring (default 10%)
    - pattern: how goal probability varies with sequence length
              "uniform" - same rate for all sequences
              "decreasing" - higher rate for shorter sequences
              "increasing" - higher rate for longer sequences
              "actual" - weakly increasing with some noise
    - steepness: how quickly the probability changes with sequence length (1.0 = linear)

    Returns:
    - Array of goal probabilities for each sequence length
    """
    sequence_lengths = np.arange(10)

    if pattern == "uniform":
        return np.full(10, base_rate)
    elif pattern == "decreasing":
        # Exponentially decreasing probabilities
        rates = base_rate * np.exp(-steepness * sequence_lengths / 10)
        return (
            rates / rates.max() * base_rate * 2
        )  # Scale to make base_rate the average
    elif pattern == "increasing":
        # Exponentially increasing probabilities
        rates = base_rate * np.exp(steepness * sequence_lengths / 10)
        return (
            rates / rates.max() * base_rate * 2
        )  # Scale to make base_rate the average
    elif pattern == "actual":
        # Weakly increasing with subtle noise
        base_trend = np.linspace(0.8, 1.3, 10)  # Weak linear increase
        noise = (
            np.array([0, 0.05, -0.02, 0.03, -0.02, 0.02, -0.03, 0.02, -0.01, 0]) * 0.5
        )
        rates = base_rate * (base_trend + noise)
        return rates
    else:
        raise ValueError(
            "Pattern must be 'uniform', 'decreasing', 'increasing', or 'actual'"
        )


def simulate_goals_for_sequences(
    pass_counts, base_rate=0.1, pattern="uniform", steepness=1.0
):
    """
    Simulate goals for each passing sequence length with varying scoring rates.

    Parameters:
    - pass_counts: array of counts for each pass length
    - base_rate: base probability of scoring (default 10%)
    - pattern: how goal probability varies with sequence length
              "uniform" - same rate for all sequences
              "decreasing" - higher rate for shorter sequences
              "increasing" - higher rate for longer sequences
    - steepness: how quickly the probability changes with sequence length (1.0 = linear)

    Returns:
    - Array of goal counts for each pass length
    """
    # Get goal probabilities for each sequence length
    goal_rates = get_goal_rates(base_rate, pattern, steepness)

    # Generate goals using binomial distribution with varying rates
    goals = np.array(
        [
            stats.binom.rvs(n=int(count), p=rate)
            for count, rate in zip(pass_counts, goal_rates)
        ]
    )

    return goals, goal_rates


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


def plot_goals_percentage(df_counts, df_goals, pattern_name=""):
    """
    Create a stacked bar chart showing goals vs non-goals for each pass sequence.

    Parameters:
    - df_counts: polars DataFrame containing pass sequence counts
    - df_goals: polars DataFrame containing goals data
    """
    # Prepare data for stacked bar chart
    data = []
    for i in range(10):
        sequence_type = str(i) if i < 9 else "9+"
        total_sequences = df_counts.filter(pl.col("passes") == sequence_type)[
            "count"
        ].item()
        goals = df_goals.filter(pl.col("passes") == sequence_type)["goals"].item()
        non_goals = total_sequences - goals

        # Add goals row
        data.append(
            {
                "passes": sequence_type,
                "count": goals,
                "type": "Goals",
                "percentage": (goals / total_sequences) * 100,
            }
        )
        # Add non-goals row
        data.append(
            {
                "passes": sequence_type,
                "count": non_goals,
                "type": "Non-Goals",
                "percentage": (non_goals / total_sequences) * 100,
            }
        )

    # Convert to polars DataFrame
    df_stacked = pl.DataFrame(data)

    # Convert to pandas for plotnine
    plot_df = df_stacked.to_pandas()

    # Create plot
    # Create percentage labels
    plot_df["label"] = plot_df["percentage"].round(1).astype(str) + "%"

    plot = (
        ggplot(plot_df, aes(x="passes", y="count", fill="type"))
        + geom_bar(stat="identity", position="stack")
        # + geom_text(
        #     aes(label="label"),
        #     position=position_stack(vjust=0.5),
        #     size=8,
        # )
        + labs(
            x="Number of Passes in Sequence",
            y="Count",
            title=f"Goals vs Non-Goals by Pass Sequence Length",
            fill="Outcome",
        )
        + scale_fill_manual(values=["darkred", "lightgray"])
        + theme_minimal()
        + theme(figure_size=(12, 6))
    )

    # Save plot to figures directory
    plot.save(f"figures/goals_percentage_{pattern_name.lower()}.png", dpi=300)


def plot_goals_distribution(df, pattern_name=""):
    """
    Create a bar plot showing the distribution of goals per pass sequence using plotnine.

    Parameters:
    - df: polars DataFrame containing goals data
    """
    # Convert polars DataFrame to pandas for plotnine
    plot_df = df.to_pandas()

    # Create plot
    # Round percentages for labels
    plot_df["percentage_label"] = plot_df["percentage"].round(1).astype(str) + "%"

    plot = (
        ggplot(plot_df, aes(x="passes", y="goals"))
        + geom_bar(stat="identity", fill="darkred", alpha=0.8)
        + geom_text(
            aes(label="percentage_label"),
            va="bottom",
            position=position_dodge(width=0.9),
            size=8,
            format_string="{}",
        )
        + labs(
            x="Number of Passes in Sequence",
            y="Number of Goals",
            title=f"Goals Scored by Pass Sequence Length",
        )
        + theme_minimal()
        + theme(figure_size=(12, 6))
    )

    # Save plot to figures directory
    plot.save(f"figures/goals_distribution_{pattern_name.lower()}.png", dpi=300)


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

    # Plot the pass distribution
    plot_pass_distribution(df)

    # Simulate and plot goals with decreasing pattern
    print("\nDecreasing Pattern (higher scoring rate for shorter sequences):")
    print("=" * 70)
    goals_dec, rates_dec = simulate_goals_for_sequences(
        pass_counts,
        base_rate=0.1,
        pattern="decreasing",
        steepness=2.0,
    )

    print("\nGoal Rates by Sequence Length:")
    print("-" * 50)
    for i, rate in enumerate(rates_dec):
        sequence_type = str(i) if i < 9 else "9+"
        print(f"{sequence_type} passes: {rate:.1%}")

    df_goals_dec = pl.DataFrame(
        {
            "passes": [str(i) if i < 9 else "9+" for i in range(10)],
            "goals": goals_dec,
            "percentage": (goals_dec / goals_dec.sum()) * 100,
        }
    )

    print("\nGoals Summary:")
    print("-" * 50)
    print(
        df_goals_dec.select(
            [
                pl.col("passes").alias("Number of Passes"),
                pl.col("goals").alias("Goals"),
                pl.col("percentage").round(1).alias("Percentage"),
            ]
        )
    )

    plot_goals_distribution(df_goals_dec, "decreasing_pattern")
    plot_goals_percentage(df, df_goals_dec, "decreasing_pattern")

    # Simulate and plot goals with increasing pattern
    print("\nIncreasing Pattern (higher scoring rate for longer sequences):")
    print("=" * 70)
    goals_inc, rates_inc = simulate_goals_for_sequences(
        pass_counts,
        base_rate=0.1,
        pattern="increasing",
        steepness=2.0,
    )

    print("\nGoal Rates by Sequence Length:")
    print("-" * 50)
    for i, rate in enumerate(rates_inc):
        sequence_type = str(i) if i < 9 else "9+"
        print(f"{sequence_type} passes: {rate:.1%}")

    df_goals_inc = pl.DataFrame(
        {
            "passes": [str(i) if i < 9 else "9+" for i in range(10)],
            "goals": goals_inc,
            "percentage": (goals_inc / goals_inc.sum()) * 100,
        }
    )

    print("\nGoals Summary:")
    print("-" * 50)
    print(
        df_goals_inc.select(
            [
                pl.col("passes").alias("Number of Passes"),
                pl.col("goals").alias("Goals"),
                pl.col("percentage").round(1).alias("Percentage"),
            ]
        )
    )

    plot_goals_distribution(df_goals_inc, "increasing_pattern")
    plot_goals_percentage(df, df_goals_inc, "increasing_pattern")

    # Simulate and plot goals with uniform pattern
    print("\nUniform Goal Distribution:")
    print("=" * 70)
    goals_uni, rates_uni = simulate_goals_for_sequences(
        pass_counts,
        base_rate=0.1,
        pattern="uniform",
        steepness=1.0,
    )

    print("\nGoal Rates by Sequence Length:")
    print("-" * 50)
    for i, rate in enumerate(rates_uni):
        sequence_type = str(i) if i < 9 else "9+"
        print(f"{sequence_type} passes: {rate:.1%}")

    df_goals_uni = pl.DataFrame(
        {
            "passes": [str(i) if i < 9 else "9+" for i in range(10)],
            "goals": goals_uni,
            "percentage": (goals_uni / goals_uni.sum()) * 100,
        }
    )

    print("\nGoals Summary:")
    print("-" * 50)
    print(
        df_goals_uni.select(
            [
                pl.col("passes").alias("Number of Passes"),
                pl.col("goals").alias("Goals"),
                pl.col("percentage").round(1).alias("Percentage"),
            ]
        )
    )

    plot_goals_distribution(df_goals_uni, "uniform")
    plot_goals_percentage(df, df_goals_uni, "uniform")

    # Simulate and plot goals with actual pattern
    print("\nActual Goal Distribution:")
    print("=" * 70)
    goals_act, rates_act = simulate_goals_for_sequences(
        pass_counts,
        base_rate=0.1,
        pattern="actual",
        steepness=1.0,
    )

    print("\nGoal Rates by Sequence Length:")
    print("-" * 50)
    for i, rate in enumerate(rates_act):
        sequence_type = str(i) if i < 9 else "9+"
        print(f"{sequence_type} passes: {rate:.1%}")

    df_goals_act = pl.DataFrame(
        {
            "passes": [str(i) if i < 9 else "9+" for i in range(10)],
            "goals": goals_act,
            "percentage": (goals_act / goals_act.sum()) * 100,
        }
    )

    print("\nGoals Summary:")
    print("-" * 50)
    print(
        df_goals_act.select(
            [
                pl.col("passes").alias("Number of Passes"),
                pl.col("goals").alias("Goals"),
                pl.col("percentage").round(1).alias("Percentage"),
            ]
        )
    )

    plot_goals_distribution(df_goals_act, "actual")
    plot_goals_percentage(df, df_goals_act, "actual")

    # Plot goal probabilities for all patterns
    plot_goal_probabilities(df, df_goals_dec, df_goals_inc, df_goals_uni, df_goals_act)


if __name__ == "__main__":
    main()
