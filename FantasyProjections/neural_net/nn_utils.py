"""Creates and exports helper functions commonly used when manipulating Neural Networks.

    Functions:
        compare_net_sizes : Returns whether two Neural Networks are the same shape/size (i.e. same number of layers and parameters).
"""

def compare_net_sizes(model1, model2):
    """Returns whether two Neural Networks are the same shape/size (i.e. same number of layers and parameters).

        Does not compare whether the two models have the same values for their parameters, or names for their layers.

        Args:
            model1 (NeuralNetwork): First NeuralNetwork object to compare
            model2 (NeuralNetwork): Second NeuralNetwork object to compare

        Returns:
            bool: Whether the two input networks are the same shape/size
    """

    for child1, child2 in zip(model1.children(), model2.children()):
        if (not isinstance(child1, type(child2))) or (len(child1) != len(child2)):
            return False
        for layer1, layer2 in zip(child1, child2):
            if not isinstance(layer1, type(layer2)):
            # if type(layer1) != type(layer2):
                return False
            if hasattr(layer1, 'in_features'):
                if not hasattr(layer2, 'in_features'):
                    return False
                if layer1.in_features != layer2.in_features:
                    return False
            if hasattr(layer1, 'out_features'):
                if not hasattr(layer2, 'out_features'):
                    return False
                if layer1.out_features != layer2.out_features:
                    return False
    return True
