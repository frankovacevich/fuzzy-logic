from fuzzy_logic import FuzzyVariable, FuzzyLogic, FuzzyTuple

# Categories
POOR = "POOR"
GOOD = "GOOD"
EXCELLENT = "EXCELLENT"

BAD = "BAD"
DELICIOUS = "DELICIOUS"

CHEAP = "CHEAP"
AVERAGE = "AVERAGE"
GENEROUS = "GENEROUS"


class TipPercentageCalculator:

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
    def _rule1(self, service_level: FuzzyVariable, food_quality: FuzzyVariable) -> FuzzyTuple:
        result = service_level.is_(POOR)
        if result.degree > 0.5:
            self._explanation += " - The service level is poor.\n"
        return CHEAP, result

    def _rule2(self, service_level: FuzzyVariable, food_quality: FuzzyVariable) -> FuzzyTuple:
        result = food_quality.is_(BAD)
        if result.degree > 0.5:
            self._explanation += " - The food quality is bad.\n"
        return CHEAP, result

    def _rule3(self, service_level: FuzzyVariable, food_quality: FuzzyVariable) -> FuzzyTuple:
        result = service_level.is_(GOOD)
        if result.degree > 0.5:
            self._explanation += " - The service level is good.\n"
        return AVERAGE, result

    def _rule4(self, service_level: FuzzyVariable, food_quality: FuzzyVariable) -> FuzzyTuple:
        result = service_level.is_(EXCELLENT)
        if result.degree > 0.5:
            self._explanation += " - The service level is excellent.\n"
        return GENEROUS, result

    def _rule5(self, service_level: FuzzyVariable, food_quality: FuzzyVariable) -> FuzzyTuple:
        result = food_quality.is_(DELICIOUS)
        if result.degree > 0.5:
            self._explanation += " - The food quality is delicious.\n"
        return GENEROUS, result

    def __init__(self) -> None:
        self._explanation = ""
        self._fuzzy_logic = FuzzyLogic(
            inputs=[self.ServiceLevel, self.FoodQuality],
            output=self.TipPercentage,
            rules=[self._rule1, self._rule2, self._rule3, self._rule4, self._rule5],
        )

    def predict(self, service_level: float, food_quality: float) -> tuple[float, str]:
        self._explanation = ""
        result = self._fuzzy_logic.predict(service_level=service_level, food_quality=food_quality)
        self._explanation = f"Tip percentage should be {result:.2f}% because:\n{self._explanation}"
        return result, self._explanation


calc = TipPercentageCalculator()
result, explanation = calc.predict(service_level=90, food_quality=80)
print(explanation)
