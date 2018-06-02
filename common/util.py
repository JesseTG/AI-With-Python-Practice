import numpy
from numpy import ndarray
from matplotlib import pyplot
from sklearn.metrics import classification_report


def is_jupyter() -> bool:
    return 'get_ipython' in globals()


def visualize_classifier(classifier, X: ndarray, Y: ndarray, title="", mesh_step_size=0.01, marker_size=75) -> None:
    min_x, min_y = X[:, 0].min() - 1.0, X[:, 1].min() - 1.0
    max_x, max_y = X[:, 0].max() + 1.0, X[:, 1].max() + 1.0

    # Make a grid with coordinates that covers the range of points
    grid_x, grid_y = numpy.meshgrid(
        numpy.arange(min_x, max_x, mesh_step_size),
        numpy.arange(min_y, max_y, mesh_step_size)
    )  # type: ndarray, ndarray

    output = classifier.predict(
        numpy.c_[grid_x.ravel(), grid_y.ravel()]
    )  # type: ndarray

    output = output.reshape(grid_x.shape)  # type: ndarray

    # Create a plot
    pyplot.figure()

    # Set the title
    pyplot.title(title)

    # Pick out some colors
    pyplot.pcolormesh(grid_x, grid_y, output, cmap=pyplot.cm.gray)

    # Plot the training points
    pyplot.scatter(X[:, 0], X[:, 1], c=Y, s=marker_size,
                   edgecolors='black', linewidth=1, cmap=pyplot.cm.Paired)

    # Set the plot's boundaries
    pyplot.xlim(grid_x.min(), grid_x.max())
    pyplot.ylim(grid_y.min(), grid_y.max())

    # Draw ticks on each axis
    pyplot.xticks(numpy.arange(int(min_x), int(max_x), 1.0))
    pyplot.yticks(numpy.arange(int(min_y), int(max_y), 1.0))

    pyplot.show()

    
def print_classification_report(class_test, test_predictions, class_train, train_predictions, target_names):
    print('#' * 60)
    print("Classifier Performance on Test Dataset:")
    print(classification_report(class_test, test_predictions, target_names=target_names))
    print('#' * 60)
    print("Classifier Performance on Training Dataset:")
    print(classification_report(class_train, train_predictions, target_names=target_names))
    print('#' * 60)
