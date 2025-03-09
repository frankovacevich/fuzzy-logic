import numpy as np
from typing import Callable, Generator, Optional


MembershipFunction = Callable[[float | np.ndarray], "FuzzyValue"]
FuzzyTuple = tuple[str, "FuzzyValue"]


class FuzzyValue:
    def __init__(self, degree: Optional[np.ndarray] = None):
        self.degree = degree if degree is not None else np.array([0])

    def __str__(self) -> str:
        return f"FuzzyValue: {self.degree}"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self.degree)

    def __and__(self, other: "FuzzyValue") -> "FuzzyValue":
        """
        Implements the AND operator for fuzzy values by taking the minimum degree at each point.
        """
        return FuzzyValue(np.minimum(self.degree, other.degree))

    def __or__(self, other: "FuzzyValue") -> "FuzzyValue":
        """
        Implements the OR operator for fuzzy values by taking the maximum degree at each point.
        """
        return FuzzyValue(np.maximum(self.degree, other.degree))

    def __invert__(self) -> "FuzzyValue":
        """
        Implements the NOT operator for fuzzy values by taking the complement of the degree.
        """
        return FuzzyValue(1 - self.degree)


class FuzzyVariable:

    def __init__(self, name: str, min_val: float, max_val: float):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val

        self._membership_functions: dict[str, MembershipFunction] = {}
        self._fuzzy_values: dict[str, FuzzyValue] = {}

    @staticmethod
    def create(name: str, min_val: float, max_val: float) -> "FuzzyVariable":
        return FuzzyVariable(name, min_val, max_val)

    @property
    def categories(self) -> list[str]:
        return list(self._membership_functions.keys())

    def __str__(self) -> str:
        return f"FuzzyVariable<{self.name}, {self.min_val}, {self.max_val}>"

    def __repr__(self) -> str:
        return f"{self}: {self._fuzzy_values}"

    def __getitem__(self, category: str) -> FuzzyValue:
        return self.is_(category)

    def __setitem__(self, category: str, value: FuzzyValue) -> None:
        self._fuzzy_values[category] = value

    def __iter__(self) -> Generator[str, None, None]:
        yield from self._fuzzy_values

    def clone(self) -> "FuzzyVariable":
        """
        Create a copy of the variable with the same membership functions and empty fuzzy values.
        """
        new_var = FuzzyVariable(self.name, self.min_val, self.max_val)
        new_var._membership_functions = self._membership_functions.copy()
        new_var._fuzzy_values = {category: FuzzyValue() for category in self._membership_functions}
        return new_var

    def fuzzify(self, crisp_value: float | np.ndarray) -> "FuzzyVariable":
        """
        Transform a crisp value into a fuzzy set using the defined membership functions.
        """
        if len(self._membership_functions) == 0:
            raise ValueError("No membership functions defined.")

        new_var = self.clone()
        new_var._fuzzy_values = {
            category: func(np.atleast_1d(crisp_value))
            for category, func in self._membership_functions.items()
        }
        return new_var

    def is_(self, category: str) -> "FuzzyValue":
        """
        Return the degree of membership for a category.
        """
        if category not in self._fuzzy_values:
            raise ValueError(f"Category '{category}' not defined for variable '{self.name}'.")
        return self._fuzzy_values[category]

    def is_not(self, category: str) -> "FuzzyValue":
        """
        Return the degree of non-membership for a category.
        """
        return ~self.is_(category)

    def get_max_membership_category(self) -> str:
        """
        Return the category with the highest membership degree
        """
        return max(self._fuzzy_values, key=lambda v: self._fuzzy_values[v].degree)  # type: ignore

    def add_membership_function(self, category: str, func: MembershipFunction) -> None:
        """
        Add a membership function to the variable.
        """
        if category in self._membership_functions:
            raise ValueError(f"Membership function for '{category}' already defined.")
        self._fuzzy_values[category] = FuzzyValue()
        self._membership_functions[category] = func

    def add_triangular_membership_function(
        self,
        category: str,
        a: float,
        b: float,
        c: float,
    ) -> None:
        r"""
        Add a triangular membership function to the variable.
          ▲
        1-│          ^
          │         / \
          │        /   \
          │       /     \
        0-└──────|───|───|───────────────────►
                 a   b   c
        """

        def function(x: float | np.ndarray) -> FuzzyValue:
            x = np.atleast_1d(x)
            result = np.zeros_like(x)
            if a != b:
                result = np.where((a <= x) & (x <= b), (x - a) / (b - a), result)
            if b != c:
                result = np.where((b <= x) & (x <= c), (c - x) / (c - b), result)
            return FuzzyValue(result)

        self.add_membership_function(category, function)


class FuzzyLogic:

    def __init__(
        self, inputs: list[FuzzyVariable], output: FuzzyVariable, rules: list[Callable]
    ) -> None:
        self.inputs = inputs
        self.output = output
        self.rules = rules

    def _check_all_variables_provided(self, **crisp_values: float) -> None:
        missing = {var.name for var in self.inputs if var.name not in crisp_values}
        if missing:
            raise ValueError(f"Missing variable values in arguments: {missing}")

    def run_rules(self, **crisp_values: float) -> FuzzyVariable:
        """
        Run all rules and return the max membership degree for each category as a FuzzyVariable.
        """
        self._check_all_variables_provided(**crisp_values)
        inputs = {var.name: var.fuzzify(crisp_values[var.name]) for var in self.inputs}
        result = self.output.clone()

        # Execute each rule and keep the max membership degree for each category
        for rule in self.rules:
            category, fuzzy_value = rule(**inputs)
            result[category] = result[category] | fuzzy_value

        return result

    def predict(self, **crisp_values: float) -> float:
        """
        Use Mamdani inference and centroid deffuzification to predict the output value.
        """
        results = self.run_rules(**crisp_values)

        # Create fuzzy sets for all possible output values (universe)
        universe = np.linspace(self.output.min_val, self.output.max_val, 1000)
        output = self.output.fuzzify(universe)

        # Aggregate results
        aggregation = FuzzyValue(np.zeros_like(universe))
        for category in results:
            aggregation = aggregation | (output[category] & results[category])

        # Defuzzify
        centroid = np.sum(universe * aggregation.degree) / np.sum(aggregation.degree)
        return float(centroid)

    def predict_categorical(self, **crisp_values: float) -> str:
        """
        Use max membership degree to predict the output category.
        """
        results = self.run_rules(**crisp_values)
        return results.get_max_membership_category()
