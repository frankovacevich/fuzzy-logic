import numpy as np
from typing import Callable


MembershipFunction = Callable[[float], "FuzzyValue"]


class FuzzyValue:
    def __init__(self, degree: np.ndarray):
        self.degree = degree

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

    def clip(self, max_val: float) -> "FuzzyValue":
        """
        Clip the degree of the fuzzy value to a maximum value.
        """
        return FuzzyValue(np.minimum(self.degree, max_val))

    def maximum(self, other: "FuzzyValue") -> "FuzzyValue":
        """
        Combine two fuzzy values by taking the maximum degree at each point (same as OR).
        """
        return FuzzyValue(np.maximum(self.degree, other.degree))


class FuzzyVariable:

    def __init__(self, name: str, min_val: float, max_val: float):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val

        self._membership_functions: dict[str, MembershipFunction] = {}
        self._fuzzy_values: dict[str, FuzzyValue] = {}

    @property
    def categories(self) -> list[str]:
        return list(self._membership_functions.keys())

    def __str__(self) -> str:
        return f"FuzzyVariable<{self.name}, {self.min_val}, {self.max_val}>: {self._fuzzy_values}>"

    def __repr__(self) -> str:
        return str(self)

    def clone(self) -> "FuzzyVariable":
        new_var = FuzzyVariable(self.name, self.min_val, self.max_val)
        new_var._membership_functions = self._membership_functions.copy()
        return new_var

    def fuzzify(self, crisp_value: float | np.ndarray) -> "FuzzyVariable":
        if len(self._membership_functions) == 0:
            raise ValueError("No membership functions defined.")

        new_var = self.clone()
        new_var._fuzzy_values = {
            category: func(np.atleast_1d(crisp_value))
            for category, func in self._membership_functions.items()
        }
        return new_var

    def is_(self, category: str) -> "FuzzyValue":
        if len(self._fuzzy_values) == 0:
            raise ValueError("No fuzzy values set. Call .fuzzify() first.")
        if category not in self._fuzzy_values:
            raise ValueError(
                f"Category '{category}' not defined for variable '{self.name}'."
            )
        return self._fuzzy_values[category]

    def get_fuzzy_values(self) -> dict[str, FuzzyValue]:
        if len(self._fuzzy_values) == 0:
            raise ValueError("No fuzzy values set. Call .fuzzify() first.")
        return self._fuzzy_values

    def add_membership_function(self, category: str, func: MembershipFunction) -> None:
        if category in self._membership_functions:
            raise ValueError(f"Membership function for '{category}' already defined.")
        self._membership_functions[category] = func

    def add_triangular_membership_function(
        self,
        category: str,
        a: float,
        b: float,
        c: float,
    ) -> None:
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
        self,
        inputs: list[FuzzyVariable],
        output: FuzzyVariable,
        rules: list[Callable],
    ) -> None:
        self.inputs = inputs
        self.output = output
        self.rules = rules

    def _check_all_variables_provided(self, **crisp_values: float) -> None:
        missing = {var.name for var in self.inputs if var.name not in crisp_values}
        if missing:
            raise ValueError(f"Missing variable values in arguments: {missing}")

    def run(self, **crisp_values: float) -> float:
        self._check_all_variables_provided(**crisp_values)

        # Fuzzify inputs
        inputs = {var.name: var.fuzzify(crisp_values[var.name]) for var in self.inputs}

        # Create result arrays for each output category
        universe = np.linspace(self.output.min_val, self.output.max_val, 1000)
        results = self.output.fuzzify(universe).get_fuzzy_values()

        # Apply rules
        for rule in self.rules:
            category, output_value = rule(**inputs)
            if category not in results:
                raise ValueError(
                    f"Category '{category}' not defined for output variable."
                )

            results[category] = results[category].clip(output_value.degree)

        # maxim results
        aggregation = FuzzyValue(np.zeros_like(universe))
        for category in results:
            aggregation = aggregation | results[category]

        # Defuzzify
        centroid = np.sum(universe * aggregation.degree) / np.sum(aggregation.degree)
        return float(centroid)
