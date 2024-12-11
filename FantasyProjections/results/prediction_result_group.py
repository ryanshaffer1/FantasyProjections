"""Creates and exports classes to be used in processing the results of Fantasy Football score/stat predictors.

    Classes:
        PredictionResultGroup : List of PredictionResult objects, used to simultaneously process/visualize their results.
"""

from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class PredictionResultGroup():
    """List of PredictionResult objects, used to simultaneously process/visualize their results.
        Args:
            results (list): list of PredictionResult objects to process as a group

        Additional Class Attributes:
            names (list): list of names for each PredictionResult object (in the same order as the results list)        
        
        Public Methods:
            plot_all : Evaluates a given plot function on each PredictionResult object, either separately or together    
    """

    # CONSTRUCTOR
    results: list

    def __post_init__(self):
         # Evaluates as part of the Constructor.
        # Generates attributes that are not simple data copies of inputs.
        self.names = [result.predictor_name for result in self.results]


    # PUBLIC METHODS

    def plot_all(self, plot_func, *args, **kwargs):
        """Evaluates a given plot function on each PredictionResult object, either separately or together

            Args:
                plot_func (function): Plotting method within PredictionResult class definition
                *args: Additional arguments to pass onto the plot_func method
            
            Keyword-Args:
                together (bool, optional): True to plot all PredictionResult objects on same figure, False for separate figures. Defaults to False.
                **kwargs: Additional keyword arguments to pass onto the plot_func method
        """

        # kwargs consumed by the plot_all wrapper
        together = kwargs.get('together',False)
        kwargs.pop('together',None)

        # If plotting together, create figure and add it to arguments passed into plot function
        if together:
            fig = plt.subplots()[0]
            kwargs['fig'] = fig

        # Call plot function on all results in the group (passing unused kwargs to plot function)
        for result in self.results:
            plot_func(result, *args, **kwargs)

        # Add a legend if plotting together
        if together:
            # fig.axes[0].legend(self.names)
            fig.legend(self.names)
