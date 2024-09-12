import operator
import random
import functools
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools, gp
from scripts.preprocessing.data_loader import (
    get_train_test_splits,
    get_combined_dataset,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scripts.visualisations.plotting import plot_averaged_heatmap
from multiprocessing import Pool


def protected_div(x, y):
    return np.divide(x, y, out=np.zeros_like(x), where=y != 0)


def if_then_else(condition, output1, output2):
    return np.where(condition, output1, output2)


def constant():
    return random.uniform(0, 1)


def define_primitives(X):
    pset = gp.PrimitiveSet("MAIN", X.shape[1])

    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(if_then_else, 3)

    pset.addEphemeralConstant("constant", functools.partial(constant))

    feature_names = [
        "x",
        "y",
        "bedrock_elevation",
        "precipitation",
        "air_temperature",
        "ocean_temperature",
        "year",
        "distance_to_pole",
        "bedrock_below_sea_level",
        "temperature_difference",
        "log_air_temperature",
        "rolling_std_precipitation",
        "rolling_std_air_temperature",
        "air_temperature_low_45",
    ]

    rename_dict = {f"ARG{i}": name for i, name in enumerate(feature_names)}
    pset.renameArguments(**rename_dict)
    return pset


def vectorized_evaluate(individual, pset, X, y):
    func = gp.compile(expr=individual, pset=pset)

    def vectorized_func(*args):
        return np.frompyfunc(func, len(args), 1)(*args)

    predictions = vectorized_func(*[X[:, i] for i in range(X.shape[1])])
    predictions = np.clip(predictions.astype(float), 0, 1)
    mse = mean_squared_error(y, predictions)
    return (mse,)


def setup_toolbox(pset):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=8)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    return toolbox


def setup_parallel_toolbox(pset, X, y):
    toolbox = setup_toolbox(pset)
    pool = Pool()
    toolbox.register("map", pool.map)
    toolbox.register("evaluate", vectorized_evaluate, pset=pset, X=X, y=y)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    return toolbox, pool


def run_gp(X, y, n_gen=50, pop_size=500, cxpb=0.8, mutpb=0.2):
    pset = define_primitives(X)
    toolbox, pool = setup_parallel_toolbox(pset, X, y)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb, mutpb, n_gen, stats=mstats, halloffame=hof, verbose=True
    )

    pool.close()
    return pop, log, hof


def evaluate_model(model, X, y):
    pset = define_primitives(X)
    func = gp.compile(expr=model, pset=pset)

    def vectorized_func(*args):
        return np.frompyfunc(func, len(args), 1)(*args)

    predictions = vectorized_func(*[X[:, i] for i in range(X.shape[1])])
    predictions = np.clip(predictions.astype(float), 0, 1)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    return mse, rmse, mae, r2, predictions


def evolve_models(X_train, y_train, X_test, y_test):
    models = {}
    error_dict = {}
    for target in ["ice_velocity", "ice_mask", "ice_thickness"]:
        print(f"\nEvolving model for {target}...")
        y_train_target = y_train[target].values
        pop, log, hof = run_gp(X_train.values, y_train_target)
        best_model = hof[0]
        models[target] = best_model

        # Evaluate on train set
        train_mse, train_rmse, train_mae, train_r2, _ = evaluate_model(
            best_model, X_train.values, y_train_target
        )

        # Evaluate on test set
        test_mse, test_rmse, test_mae, test_r2, test_predictions = evaluate_model(
            best_model, X_test.values, y_test[target].values
        )

        # Calculate error for the entire test set
        error = np.abs(y_test[target].values - test_predictions)
        error_dict[f"{target}_error"] = error

        print(f"\nPerformance metrics for {target}:")
        print(f"{'Metric':<10} {'Train':<15} {'Test':<15}")
        print(f"{'-'*40}")
        print(f"{'MSE':<10} {train_mse:<15.6f} {test_mse:<15.6f}")
        print(f"{'RMSE':<10} {train_rmse:<15.6f} {test_rmse:<15.6f}")
        print(f"{'MAE':<10} {train_mae:<15.6f} {test_mae:<15.6f}")
        print(f"{'R2':<10} {train_r2:<15.6f} {test_r2:<15.6f}")

        # Print the best model
        print(f"\nBest model for {target}:")
        print(best_model)

    # Create a DataFrame with the error columns
    error_df = pd.DataFrame(error_dict)

    return models, error_df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_splits(test_size=0.2)
    models, error_df = evolve_models(X_train, y_train, X_test, y_test)
    print("ERRORS", error_df.describe())
    df = get_combined_dataset(X_train, y_train, X_test, y_test)
    df = pd.concat([df, error_df], axis=1)

    for target in ["ice_velocity", "ice_mask", "ice_thickness"]:
        plot_averaged_heatmap(
            df, f"{target}_error", save_path=f"out/images/gp/{target}_error"
        )
