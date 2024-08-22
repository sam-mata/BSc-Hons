import numpy as np
from deap import algorithms, base, creator, tools, gp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import operator
import math
import random

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def rand101():
    return random.randint(-1, 1)

def setup_toolbox():
    pset = gp.PrimitiveSet("MAIN", 9)  # Explicitly set to 9 input features
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)

    pset.addEphemeralConstant("rand101", rand101)

    pset.renameArguments(ARG0='x0', ARG1='x1', ARG2='x2', ARG3='x3', ARG4='x4',
                         ARG5='x5', ARG6='x6', ARG7='x7', ARG8='x8')

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    return toolbox, pset

def evalSymbReg(individual, toolbox, X, y):
    func = toolbox.compile(expr=individual)
    try:
        pred = np.array([func(*xi) for xi in X])
        mse = mean_squared_error(y, pred)
        return mse,
    except Exception as e:
        print(f"Error in evalSymbReg: {e}")
        return float('inf'),

def create_gp_model(X, y, n_gen=50, pop_size=300, cxpb=0.5, mutpb=0.1):
    toolbox, pset = setup_toolbox()

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.register("evaluate", evalSymbReg, toolbox=toolbox, X=X, y=y)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb, mutpb, n_gen, stats=mstats, halloffame=hof, verbose=True)

    return hof[0], toolbox

def evaluate_gp_model(model, toolbox, X, y):
    func = toolbox.compile(expr=model)
    try:
        y_pred = np.array([func(*xi) for xi in X])
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    except Exception as e:
        print(f"Error in evaluate_gp_model: {e}")
        return {
            'MSE': float('inf'),
            'RMSE': float('inf'),
            'MAE': float('inf'),
            'R2': float('-inf')
        }

def apply_gp(X_train, X_test, y_train, y_test):
    best_model, toolbox = create_gp_model(X_train, y_train)
    
    train_metrics = evaluate_gp_model(best_model, toolbox, X_train, y_train)
    test_metrics = evaluate_gp_model(best_model, toolbox, X_test, y_test)
    
    return best_model, train_metrics, test_metrics