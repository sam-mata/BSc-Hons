# Constants
TARGET = 'ice_velocity'  # The target variable to predict
POPULATION = 15  # Number of individuals in the population

import numpy as np
import operator
import random
from functools import partial
from deap import algorithms, base, creator, gp, tools
from scripts.preprocessing.data_loader import get_train_test_splits
from scripts.preprocessing.preprocessor import apply_minmax_scaling
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import networkx as nx

# Load the data
X_train, X_test, y_train, y_test = get_train_test_splits(test_size=0.2)
X_train, y_train, train_scales = apply_minmax_scaling(X_train, y_train)
X_test, y_test, test_scales = apply_minmax_scaling(X_test, y_test)

def plot_tree(expr):
    nodes, edges, labels = gp.graph(expr)
    
    if len(nodes) == 1:
        print(expr)
        return
    
    if not nodes:  # Handle the case of a single terminal
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, str(expr), ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.title("Best Individual (Single Node)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('best_individual_syntax_tree.png', dpi=300, bbox_inches='tight')
        plt.close()
        return

    # Create a directed graph
    g = nx.DiGraph()
    g.add_edges_from(edges)
    
    # Create a hierarchical layout
    pos = nx.spring_layout(g, k=0.5, iterations=50)
    
    # Find the root node (the one with in-degree 0)
    root = [node for node in g.nodes() if g.in_degree(node) == 0][0]
    
    # Perform a breadth-first search to assign levels
    bfs_edges = list(nx.bfs_edges(g, root))
    levels = {root: 0}
    for parent, child in bfs_edges:
        levels[child] = levels[parent] + 1
    
    # Adjust y-coordinate based on levels
    max_level = max(levels.values())
    for node in pos:
        pos[node] = (pos[node][0], 1 - levels[node] / max_level)
    
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(g, pos, node_size=2000, node_color='lightblue', edgecolors='black')
    
    # Draw edges
    nx.draw_networkx_edges(g, pos, edge_color='gray', arrows=True, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(g, pos, labels, font_size=8, font_weight='bold')
    
    plt.title("Best Individual (Syntax Tree)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('best_individual_syntax_tree.png', dpi=300, bbox_inches='tight')
    plt.close()

def protected_div(left, right):
    try:
        return left / right if abs(right) > 1e-6 else 1.0
    except ZeroDivisionError:
        return 1.0

def if_then_else(condition, out1, out2):
    return out1 if condition > 0 else out2

# Define the primitive set (function set)
pset = gp.PrimitiveSet("MAIN", X_train.shape[1])
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(if_then_else, 3)
pset.addEphemeralConstant("rand", partial(random.uniform, -1, 1))

# Rename the arguments
feature_names = X_train.columns.tolist()
for i, name in enumerate(feature_names):
    pset.renameArguments(**{f'ARG{i}': name})

# Define fitness and individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Define the toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=8)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Define evaluation function
def evalSymbRegCV(individual, points, targets, cv=5):
    func = toolbox.compile(expr=individual)
    scores = []
    kf = KFold(n_splits=cv, shuffle=False)
    
    for train_index, val_index in kf.split(points):
        X_train, X_val = points.iloc[train_index], points.iloc[val_index]
        y_train, y_val = targets.iloc[train_index], targets.iloc[val_index]
        
        try:
            pred = np.array([func(*p) for p in X_val.values])
            rmse = np.sqrt(np.mean((pred - y_val.values.ravel())**2))
            scores.append(rmse)
        except Exception as e:
            scores.append(float('inf'))
    
    avg_rmse = np.mean(scores)
    complexity = len(individual)
    return avg_rmse,

toolbox.register("evaluate", evalSymbRegCV, points=X_train, targets=y_train[TARGET])
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Limit the tree depth
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

# Define the main function
def main():
    pop = toolbox.population(n=POPULATION)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=50, 
                                    stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

if __name__ == "__main__":
    pop, log, hof = main()
    best = hof[0]
    print("Best individual:", best)
    print("Best fitness:", best.fitness.values[0])

    # Evaluate on test set
    func = toolbox.compile(expr=best)
    test_pred = np.array([func(*p) for p in X_test.values])

    # Calculate metrics
    mse, rmse, mae, r2 = calculate_metrics(y_test[TARGET].values, test_pred)
    print(f"Test MSE: {mse}")
    print(f"Test RMSE: {rmse}")
    print(f"Test MAE: {mae}")
    print(f"Test R2: {r2}")

    # Plot the output symbology using custom function
    plot_tree(best)
    
    # Baseline
    et = ExtraTreesRegressor(n_estimators=500, max_depth=10, min_samples_split=5, random_state=307, n_jobs=-1)
    et.fit(X_train, y_train[TARGET])
    et_pred = et.predict(X_test)

    # Calculate metrics for ExtraTrees
    et_mse, et_rmse, et_mae, et_r2 = calculate_metrics(y_test[TARGET].values, et_pred)
    print("\nExtraTrees Baseline:")
    print(f"Test MSE: {et_mse}")
    print(f"Test RMSE: {et_rmse}")
    print(f"Test MAE: {et_mae}")
    print(f"Test R2: {et_r2}")

    # Plot GP vs ExtraTrees predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[TARGET].values, test_pred, alpha=0.5, label='GP Predictions')
    plt.scatter(y_test[TARGET].values, et_pred, alpha=0.5, label='ET Predictions')
    plt.plot([y_test[TARGET].min(), y_test[TARGET].max()], 
            [y_test[TARGET].min(), y_test[TARGET].max()], 
            'r--', label='Perfect Prediction')
    plt.xlabel(f'True {TARGET}')
    plt.ylabel(f'Predicted {TARGET}')
    plt.title(f'GP vs ExtraTrees Predictions for {TARGET}')
    plt.legend()
    plt.savefig('predictions_comparison.png')
    plt.close()