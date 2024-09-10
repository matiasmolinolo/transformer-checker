import json
import os
import random
from typing import List, Tuple

import typer
from tqdm import tqdm
from typing_extensions import Annotated

from .checker import is_dyck_word
from .constants import BRACKETS


def _generate_balanced_string(order: int, length: int, seed: int = 42) -> str:
    """
    Generate a string of length `length` from the Dyck language of order `order`.

    Args:
        order (int): The order of the Dyck language.
        length (int): The length of the string to generate.
        seed (int): The seed for the random number generator.
    Returns:
        str: A string of length `length` from the Dyck language of order `order`.
    """

    if length == 0:
        return ""

    length = length if length % 2 == 0 else length + 1

    stack = []
    word = ""

    brackets = [(k, v) for k, v in list(BRACKETS.items())[:order]]

    half_length = length // 2

    first_half = last_half = 0

    while first_half + last_half < length:
        if first_half < half_length and (len(stack) == 0 or random.random() < 0.5):
            opening_bracket, closing_bracket = random.choice(brackets)
            stack.append(closing_bracket)
            first_half += 1
            word += opening_bracket
        else:
            bracket = stack.pop()
            last_half += 1
            word += bracket

    return word

    # grammar
    # s -> eps
    # s -> (s)s


def _generate_unbalanced_string(order: int, length: int, seed: int = 42) -> str:
    """
    Generate a string of length `length` that is not necessarily from the Dyck language of order `order`.

    Args:
        order (int): The order of the Dyck language.
        length (int): The length of the string to generate.
        seed (int): The seed for the random number generator.
    Returns:
        str: A string of length `length` that is not necessarily from the Dyck language of order `order`.
    """
    random.seed(seed)

    word = ""

    opening_brackets = [k for k, _ in list(BRACKETS.items())[:order]]
    closing_brackets = [v for _, v in list(BRACKETS.items())[:order]]

    brackets = opening_brackets + closing_brackets

    first_char = random.choice(opening_brackets) if random.random() < 0.5 else random.choice(closing_brackets)
    word += first_char

    random_brackets = [random.choice(brackets) for _ in range(length - 1)]
    random.shuffle(random_brackets)
    unbalanced_str = word + "".join(random_brackets)

    if is_dyck_word(unbalanced_str, order):
        del unbalanced_str
        del brackets
        return _generate_unbalanced_string(order, length)

    return unbalanced_str


def _generate_samples(
    n: int, k: int, min_length: int = 0, max_length: int = 1024, balanced: float = 0.5, seed: int = 42
) -> List[str]:
    """
    Generate a list of 'n' strings of length at most 'max_length' from the Dyck language of order 'k'.
    These strings may or may not be members of the Dyck language of order 'k'.

    The distribution of balanced and unbalanced strings is controlled by the 'balanced' parameter.
    A value of 1.0 will generate only balanced strings, a value of 0.0 will generate only unbalanced strings
    and a value of 0.5 will generate an equal number of balanced and unbalanced strings.

    Args:
        n (int): The number of strings to generate.
        k (int): The order of the Dyck language.
        min_length (int): The minimum length of the strings to generate.
        max_length (int): The maximum length of the strings to generate.
        balanced (float): The proportion of balanced strings to generate.
        seed (int): The seed for the random number generator.
    Returns:
        List[str]: A list of 'n' strings of length at most 'max_length' from the Dyck language of order 'k'."""

    random.seed(seed)

    balanced_strings = [
        _generate_balanced_string(k, random.randint(min_length, max_length))
        for _ in tqdm(range(int(n * balanced)), desc="Generating balanced strings")
    ]
    unbalanced_strings = [
        _generate_unbalanced_string(k, random.randint(min_length, max_length))
        for _ in tqdm(range(n - len(balanced_strings)), desc="Generating unbalanced strings")
    ]

    assert (
        len(balanced_strings) + len(unbalanced_strings) == n
    ), "The number of generated strings does not match the expected number."
    assert all(
        len(s) <= max_length for s in balanced_strings + unbalanced_strings
    ), "Some strings exceed the maximum length."
    assert all(
        len(s) >= min_length for s in balanced_strings + unbalanced_strings
    ), "Some strings are shorter than the minimum length."
    assert all(
        is_dyck_word(s, k) for s in balanced_strings
    ), "Some balanced strings are not members of the Dyck language."
    assert all(
        not is_dyck_word(s, k) for s in unbalanced_strings
    ), "Some unbalanced strings are members of the Dyck language."

    samples = balanced_strings + unbalanced_strings
    random.shuffle(samples)

    return samples


def generate_dataset(
    n: Annotated[int, typer.Option(help="The number of strings to generate.")] = 500_000,
    k: Annotated[int, typer.Option(help="The order of the Dyck language.")] = 3,
    min_length: Annotated[int, typer.Option(help="The minimum length of the strings to generate.")] = 2,
    max_length: Annotated[int, typer.Option(help="The maximum length of the strings to generate.")] = 1024,
    balanced: Annotated[float, typer.Option(help="The proportion of balanced strings to generate.")] = 0.5,
    file: Annotated[
        bool,
        typer.Option(
            help="If present, the dataset will be saved to a file, otherwise it will be returned to a variable."
        ),
    ] = True,
) -> List[Tuple[str, bool]] | str:
    """
    Generate a list of 'n' strings of length at most 'max_length' from the Dyck language of order 'k'.
    These strings may or may not be members of the Dyck language of order 'k'.

    The distribution of balanced and unbalanced strings is controlled by the 'balanced' parameter.
    A value of 1.0 will generate only balanced strings, a value of 0.0 will generate only unbalanced strings
    and a value of 0.5 will generate an equal number of balanced and unbalanced strings.

    Args:
        n (int): The number of strings to generate.
        k (int): The order of the Dyck language.
        max_length (int): The maximum length of the strings to generate.
        balanced (float): The proportion of balanced strings to generate.
        path (str): The path to save the generated strings - if None, the list will be returned to a variable.

    Returns:
        List[Tuple[str, bool]]|str: A list of dictionaries that contain (str, bool), where string is the Dyck-k member string and class is its membership to the language or the path to the file.
    """
    path = f"data/dyck-{k}_{n}-samples_{max_length}-len_p{str(balanced).replace('.', '')}.jsonl"

    if file:
        if os.path.exists(path):
            print(f"File {path} already exists.")
            return path

    strings: List[str] = _generate_samples(n, k, min_length, max_length, balanced)
    dataset = [(s, is_dyck_word(s, k)) for s in strings]

    if file:
        if not os.path.exists(path.split("/")[0]):
            print(f"Creating directory: {path.split('/')[0]}")
            os.makedirs(path.split("/")[0])
        f = open(path, "w")
        for sample in tqdm(dataset, desc=f"Saving dataset to {path}"):
            json_record = json.dumps(sample)
            f.write(json_record + "\n")
        print(f"Dataset saved to {path}")
        f.close()
        return path
    else:
        return dataset


if __name__ == "__main__":
    typer.run(generate_dataset)
