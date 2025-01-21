
import matplotlib.pyplot as plt

# Bold text on everything
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["axes.labelweight"] = "bold"

def plot_earnings_by_week(bet_results, individual_stats=True, fig=None):

    # Track cumulative earnings for each week
    overall_results = bet_results.copy()
    overall_results['Cum Earnings'] = overall_results['Earnings'].cumsum()
    overall_results = overall_results.reset_index().drop_duplicates(subset=['Year','Week'],keep='last')

    # Generate new figure if not plotting on a pre-existing fig
    if not fig:
        fig = plt.subplots()[0]

    # Plot earnings over each week
    ax = fig.axes[0]
    ax.plot(overall_results['Week'], overall_results['Cum Earnings'], 'k', linewidth=2)
    if individual_stats:
        for player_prop in ['Pass Yds','Rush Yds','Rec Yds']:
            stat_results = bet_results[bet_results['Player Prop Stat']==player_prop].copy()
            stat_results['Cum Earnings'] = stat_results['Earnings'].cumsum()
            stat_results = stat_results.reset_index().drop_duplicates(subset=['Year','Week'],keep='last')
            ax.plot(stat_results['Week'], stat_results['Cum Earnings'], linestyle='--')
        # Add legend
        ax.legend(['Overall Earnings']+['Pass Yds','Rush Yds','Rec Yds'])

    # Format: add labels, correct tick marks, and grid
    ax.set_title('Cumulative Gambling Earnings', weight='bold')
    ax.set_xlabel('Week')
    ax.set_xticks(list(overall_results['Week'].unique()))
    ax.set_ylabel('Earnings (Units)')
    plt.grid(True)

    # "Display" plot (won't really be displayed until plt.show is called again without block=False)
    plt.show(block=False)
