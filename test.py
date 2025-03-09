from unittest import TestCase
from fuzzy_logic import FuzzyVariable, FuzzyLogic


# Categories
POOR = "POOR"
GOOD = "GOOD"
EXCELLENT = "EXCELLENT"

BAD = "BAD"
DELICIOUS = "DELICIOUS"

CHEAP = "CHEAP"
AVERAGE = "AVERAGE"
GENEROUS = "GENEROUS"


class Test(TestCase):

    @staticmethod
    def rule1(service_level, food_quality):
        return CHEAP, service_level.is_(POOR)

    @staticmethod
    def rule2(service_level, food_quality):
        return CHEAP, food_quality.is_(BAD)

    @staticmethod
    def rule3(service_level, food_quality):
        return AVERAGE, service_level.is_(GOOD)

    @staticmethod
    def rule4(service_level, food_quality):
        return GENEROUS, service_level.is_(EXCELLENT)

    @staticmethod
    def rule5(service_level, food_quality):
        return GENEROUS, food_quality.is_(DELICIOUS)

    @classmethod
    def setUpClass(cls):
        # Define variables
        cls.ServiceLevel = FuzzyVariable.create("service_level", 0, 100)
        cls.FoodQuality = FuzzyVariable.create("food_quality", 0, 100)
        cls.TipPercentage = FuzzyVariable.create("tip_percentage", 0, 30)

        # Define membership functions
        cls.ServiceLevel.add_triangular_membership_function(POOR, 0, 0, 50)
        cls.ServiceLevel.add_triangular_membership_function(GOOD, 10, 50, 90)
        cls.ServiceLevel.add_triangular_membership_function(EXCELLENT, 75, 100, 100)

        cls.FoodQuality.add_triangular_membership_function(BAD, 0, 0, 50)
        cls.FoodQuality.add_triangular_membership_function(DELICIOUS, 50, 100, 100)

        cls.TipPercentage.add_triangular_membership_function(CHEAP, 0, 6, 12)
        cls.TipPercentage.add_triangular_membership_function(AVERAGE, 10, 15, 20)
        cls.TipPercentage.add_triangular_membership_function(GENEROUS, 18, 24, 30)

    def test_predict(self):
        # Arrange
        fuzzy_logic = FuzzyLogic(
            inputs=[self.ServiceLevel, self.FoodQuality],
            output=self.TipPercentage,
            rules=[self.rule1, self.rule2, self.rule3, self.rule4, self.rule5],
        )

        # Act
        result = fuzzy_logic.predict(service_level=30, food_quality=80)

        # Assert
        self.assertAlmostEqual(15.88, result, places=2)

    def test_predict_categorical(self):
        # Arrange
        fuzzy_logic = FuzzyLogic(
            inputs=[self.ServiceLevel, self.FoodQuality],
            output=self.TipPercentage,
            rules=[self.rule1, self.rule2, self.rule3, self.rule4, self.rule5],
        )

        # Act
        category = fuzzy_logic.predict_categorical(service_level=30, food_quality=80)

        # Assert
        self.assertEqual(GENEROUS, category)
