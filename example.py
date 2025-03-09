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
ServiceLevel = FuzzyVariable.create("service_level", 0, 100)
FoodQuality = FuzzyVariable.create("food_quality", 0, 100)
TipPercentage = FuzzyVariable.create("tip_percentage", 0, 30)


# Define membership functions
ServiceLevel.add_triangular_membership_function(POOR, 0, 0, 50)
ServiceLevel.add_triangular_membership_function(GOOD, 10, 50, 90)
ServiceLevel.add_triangular_membership_function(EXCELLENT, 75, 100, 100)

FoodQuality.add_triangular_membership_function(BAD, 0, 0, 50)
FoodQuality.add_triangular_membership_function(DELICIOUS, 50, 100, 100)

TipPercentage.add_triangular_membership_function(CHEAP, 0, 6, 12)
TipPercentage.add_triangular_membership_function(AVERAGE, 10, 15, 20)
TipPercentage.add_triangular_membership_function(GENEROUS, 18, 24, 30)


# Rules
def rule1(service_level: FuzzyVariable, food_quality: FuzzyVariable) -> tuple[str, FuzzyValue]:
    result = service_level.is_(POOR)
    return CHEAP, result


def rule2(service_level: FuzzyVariable, food_quality: FuzzyVariable) -> tuple[str, FuzzyValue]:
    result = food_quality.is_(BAD)
    return CHEAP, result


def rule3(service_level: FuzzyVariable, food_quality: FuzzyVariable) -> tuple[str, FuzzyValue]:
    result = service_level.is_(GOOD)
    return AVERAGE, result


def rule4(service_level: FuzzyVariable, food_quality: FuzzyVariable) -> tuple[str, FuzzyValue]:
    result = service_level.is_(EXCELLENT)
    return GENEROUS, result


def rule5(service_level: FuzzyVariable, food_quality: FuzzyVariable) -> tuple[str, FuzzyValue]:
    result = food_quality.is_(DELICIOUS)
    return GENEROUS, result


# Create fuzzy logic system
fuzzy_logic = FuzzyLogic(
    inputs=[ServiceLevel, FoodQuality],
    output=TipPercentage,
    rules=[rule1, rule2, rule3, rule4, rule5],
)

result = fuzzy_logic.predict(service_level=30, food_quality=80)
print(result)

result_cat = fuzzy_logic.predict_categorical(service_level=30, food_quality=80)
print(result_cat)
