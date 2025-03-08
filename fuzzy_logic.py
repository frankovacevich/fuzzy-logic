import numpy as np
from typing import Callable, Optional


MembershipFunction = Callable[[float | np.ndarray], "FuzzyValue"]


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
            raise ValueError(f"Category '{category}' not defined for variable '{self.name}'.")
        return self._fuzzy_values[category]

    def is_not(self, category: str) -> "FuzzyValue":
        return ~self.is_(category)

    def get_fuzzy_set(self) -> dict[str, FuzzyValue]:
        if len(self._fuzzy_values) == 0:
            raise ValueError("No fuzzy values set. Call .fuzzify() first.")
        return self._fuzzy_values

    def get_maximum_membership(self) -> str:
        return max(self._fuzzy_values, key=lambda x: self._fuzzy_values[x].degree)  # type: ignore

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


class FuzzyRule:
    name: str
    description: str

    def __call__(self, **kwargs) -> tuple[str, FuzzyValue]:
        raise NotImplementedError


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

    def fuzzify_inputs(self, **crisp_values: float) -> dict[str, FuzzyVariable]:
        self._check_all_variables_provided(**crisp_values)
        return {var.name: var.fuzzify(crisp_values[var.name]) for var in self.inputs}

    def predict(self, **crisp_values: float) -> float:
        """
        Use Mamdani inference and centroid deffuzification to predict the output value.
        """
        inputs = self.fuzzify_inputs(**crisp_values)

        # Apply rules
        results: dict[str, FuzzyValue] = {}
        for rule in self.rules:
            category, output_value = rule(**inputs)
            if category not in self.output.categories:
                raise ValueError(f"Category '{category}' not defined for output variable.")
            results[category] = results.get(category, FuzzyValue()) | output_value

        # Create result arrays for each output category
        universe = np.linspace(self.output.min_val, self.output.max_val, 1000)
        fuzzy_output = self.output.fuzzify(universe).get_fuzzy_set()
        for category, value in results.items():
            fuzzy_output[category] = fuzzy_output[category] & value

        # Aggregate results
        aggregation = FuzzyValue(np.zeros_like(universe))
        for category, values in fuzzy_output.items():
            aggregation = aggregation | values

        # Defuzzify
        centroid = np.sum(universe * aggregation.degree) / np.sum(aggregation.degree)
        return float(centroid)

    def predict_categorical(self, **crisp_values: float) -> str:
        """
        Use max membership degree to predict the output category.
        """
        inputs = self.fuzzify_inputs(**crisp_values)

        results: dict[str, float] = {}
        for rule in self.rules:
            category, output_value = rule(**inputs)
            if output_value.degree > results.get(category, 0):
                results[category] = output_value.degree

        return max(results, key=results.get)  # type: ignore
