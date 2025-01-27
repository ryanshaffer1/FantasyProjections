
import matplotlib.pyplot as plt

# Bold text on everything
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["axes.labelweight"] = "bold"

def plot_earnings_by_week(bet_results, individual_stats=True, fig=None):

    # Track cumulative earnings for each week
    overall_results = compute_cum_earnings(bet_results)

    # Generate new figure if not plotting on a pre-existing fig
    if not fig:
        fig = plt.subplots()[0]

    # Plot earnings over each week
    ax = fig.axes[0]
    ax.plot(overall_results['Year-Week'], overall_results['Cum Earnings'], 'k', linewidth=2)
    if individual_stats:
        for player_prop in ['Pass Yds','Rush Yds','Rec Yds']:
            stat_results = compute_cum_earnings(bet_results, player_prop=player_prop)
            ax.plot(stat_results['Year-Week'], stat_results['Cum Earnings'], linestyle='--')
        # Add legend
        ax.legend(['Overall Earnings']+['Pass Yds','Rush Yds','Rec Yds'])

    # Format: add labels, correct tick marks, and grid
    ax.set_title('Cumulative Gambling Earnings', weight='bold')
    ax.set_xlabel('Week')
    ax.set_xticks(ticks=list(overall_results['Year-Week'].unique()),
                  labels=list(overall_results['Year-Week'].unique()),
                  rotation=45)
    ax.set_ylabel('Earnings (Units)')
    plt.grid(True)

    # "Display" plot (won't really be displayed until plt.show is called again without block=False)
    plt.show(block=False)

def compute_cum_earnings(bet_results_df, player_prop=None):
    # Copy input and reset index
    cum_earnings_df = bet_results_df.copy().reset_index()

    # Optionally filter on a specific player prop (e.g. Pass Yds)
    if player_prop is not None:
        cum_earnings_df = cum_earnings_df[cum_earnings_df['Player Prop Stat']==player_prop]

    # Track year and week combined, and sort chronologically
    cum_earnings_df['Year-Week'] = cum_earnings_df.apply(lambda x: f'{x['Year']}-{str(x['Week']).rjust(2,'0')}', axis=1)
    cum_earnings_df = cum_earnings_df.set_index('Year-Week').sort_index()
    # Cumulative sum of earnings at the end of each week
    cum_earnings_df['Cum Earnings'] = cum_earnings_df['Earnings'].cumsum()
    cum_earnings_df = cum_earnings_df.reset_index().drop_duplicates(subset=['Year-Week'],keep='last')
    # Reduce to only the necessary columns
    cum_earnings_df = cum_earnings_df[['Year-Week','Cum Earnings']]

    return cum_earnings_df
