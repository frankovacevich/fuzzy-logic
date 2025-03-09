import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl


# Categories
POOR = "POOR"
GOOD = "GOOD"
EXCELLENT = "EXCELLENT"

BAD = "BAD"
DELICIOUS = "DELICIOUS"

CHEAP = "CHEAP"
AVERAGE = "AVERAGE"
GENEROUS = "GENEROUS"


# Define variables
service_level = ctrl.Antecedent(np.arange(0, 101, 1), "service_level")
food_quality = ctrl.Antecedent(np.arange(0, 101, 1), "food_quality")
tip_percentage = ctrl.Consequent(np.arange(0, 31, 1), "tip_percentage")


# Define membership functions
service_level[POOR] = fuzz.trimf(service_level.universe, [0, 0, 50])
service_level[GOOD] = fuzz.trimf(service_level.universe, [10, 50, 90])
service_level[EXCELLENT] = fuzz.trimf(service_level.universe, [75, 100, 100])

food_quality[BAD] = fuzz.trimf(food_quality.universe, [0, 0, 50])
food_quality[DELICIOUS] = fuzz.trimf(food_quality.universe, [50, 100, 100])

tip_percentage[CHEAP] = fuzz.trimf(tip_percentage.universe, [0, 6, 12])
tip_percentage[AVERAGE] = fuzz.trimf(tip_percentage.universe, [10, 15, 20])
tip_percentage[GENEROUS] = fuzz.trimf(tip_percentage.universe, [18, 24, 30])


# Rules
rule1 = ctrl.Rule(service_level[POOR], tip_percentage[CHEAP])
rule2 = ctrl.Rule(food_quality[BAD], tip_percentage[CHEAP])
rule3 = ctrl.Rule(service_level[GOOD], tip_percentage[AVERAGE])
rule4 = ctrl.Rule(service_level[EXCELLENT], tip_percentage[GENEROUS])
rule5 = ctrl.Rule(food_quality[DELICIOUS], tip_percentage[GENEROUS])


# Create control system
c = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
s = ctrl.ControlSystemSimulation(c)


s.input["service_level"] = 30
s.input["food_quality"] = 80
s.compute()
print(f"Tip Percentage: {s.output['tip_percentage']:.2f}%")
