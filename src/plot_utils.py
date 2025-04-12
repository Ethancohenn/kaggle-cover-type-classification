import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_zone_cover_type_distribution(df, zone_names, title_prefix):
    """
    For each zone in 'zone_names', plot the distribution of Cover_Type for rows where the zone is active.
    """
    n_zones = len(zone_names)
    fig, axes = plt.subplots(n_zones, 1, figsize=(8, 4 * n_zones))
    if n_zones == 1:
        axes = [axes]
    
    for ax, zone in zip(axes, zone_names):
        zone_df = df[df[zone] == 1]
        total = len(zone_df)
        sns.countplot(x="Cover_Type", data=zone_df, palette="viridis", ax=ax)
        ax.set_title(f"{title_prefix}{zone} (N = {total})")
        ax.set_xlabel("Cover_Type")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

def display_zone_distributions(df, zone_names):
    """
    Display a table with the percentage distribution of Cover_Type for each zone..
    """
    distributions = {}
    for zone in zone_names:
        zone_df = df[df[zone] == 1]
        dist = zone_df["Cover_Type"].value_counts(normalize=True).sort_index()
        distributions[zone] = dist
    return pd.DataFrame(distributions).fillna(0)
