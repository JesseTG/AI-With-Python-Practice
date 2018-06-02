import ipywidgets as widgets


solver = widgets.Dropdown(
    options=["liblinear", "saga", "sag", "lbfgs", "newton-cg"],
    description='Solver'
)

random_seed = widgets.BoundedIntText(
    value=0,
    min=0,
    max=int(2**32 - 1),
    description='Random Seed',
    disabled=False
)

random_state = random_seed

max_iter = widgets.BoundedIntText(
    value=1000,
    description='Max Iterations',
    min=1
)

test_size = widgets.FloatSlider(
    value=0.2,
    min=0.05,
    max=0.95,
    step=0.05,
    continuous_update=False,
    description="% Test Data"
)

num_folds = widgets.BoundedIntText(value=3, min=2, continuous_update=False, description="# of Folds")