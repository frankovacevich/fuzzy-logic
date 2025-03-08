from fuzzy_logic import FuzzyVariable, FuzzyValue, FuzzyLogic

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
service_level = FuzzyVariable("service_level", 0, 100)
food_quality = FuzzyVariable("food_quality", 0, 100)
tip_percentage = FuzzyVariable("tip_percentage", 0, 30)


# Define membership functions
service_level.add_triangular_membership_function(POOR, 0, 0, 50)
service_level.add_triangular_membership_function(GOOD, 10, 50, 90)
service_level.add_triangular_membership_function(EXCELLENT, 75, 100, 100)

food_quality.add_triangular_membership_function(BAD, 0, 0, 50)
food_quality.add_triangular_membership_function(DELICIOUS, 50, 100, 100)

tip_percentage.add_triangular_membership_function(CHEAP, 0, 6, 12)
tip_percentage.add_triangular_membership_function(AVERAGE, 10, 15, 20)
tip_percentage.add_triangular_membership_function(GENEROUS, 18, 24, 30)


# Rules
def rule1(
    service_level: FuzzyVariable, food_quality: FuzzyVariable
) -> tuple[str, FuzzyValue]:
    result = service_level.is_(POOR) | food_quality.is_(BAD)
    return CHEAP, result


def rule2(
    service_level: FuzzyVariable, food_quality: FuzzyVariable
) -> tuple[str, FuzzyValue]:
    result = service_level.is_(GOOD)
    return AVERAGE, result


def rule3(
    service_level: FuzzyVariable, food_quality: FuzzyVariable
) -> tuple[str, FuzzyValue]:
    result = service_level.is_(EXCELLENT) | food_quality.is_(DELICIOUS)
    return GENEROUS, result


# Create fuzzy logic system
fuzzy_logic = FuzzyLogic(
    inputs=[service_level, food_quality],
    output=tip_percentage,
    rules=[rule1, rule2, rule3],
)
result = fuzzy_logic.run(service_level=30, food_quality=80)
print(result)
