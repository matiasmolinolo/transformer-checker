import json
import random
from typing import Dict, List, Tuple

import typer
from typing_extensions import Annotated
from dyck_k_generator import constants as c
from dyck_k_generator import checker
from tqdm import tqdm


def _generate_balanced_string(order: int, length: int) -> str:
    """
    Generate a string of length `length` from the Dyck language of order `order`.

    Args:
        order (int): The order of the Dyck language.
        length (int): The length of the string to generate.

    Returns:
        str: A string of length `length` from the Dyck language of order `order`.
    """
    length = length if length % 2 == 0 else length + 1

    brackets = [(k, v) for k, v in list(c.BRACKETS.items())[:order]]

    half_length = length // 2

    first_half = last_half = ''

    for _ in range(half_length):
        selected_brackets = random.choice(brackets)
        first_half += selected_brackets[0]
        last_half += selected_brackets[1]

    return first_half + last_half[::-1]


def _generate_unbalanced_string(order: int, length: int) -> str:
    """
    Generate a string of length `length` that is not necessarily from the Dyck language of order `order`.
    
    Args:
        order (int): The order of the Dyck language.
        length (int): The length of the string to generate.

    Returns:
        str: A string of length `length` that is not necessarily from the Dyck language of order `order`.
    """

    brackets = [(k, v) for k, v in list(c.BRACKETS.items())[:order]]
    brackets = [bracket for pair in brackets for bracket in pair]

    unbalanced_str = ''.join(random.choice(brackets) for _ in range(length))

    if checker.is_dyck_word(unbalanced_str, order):
        del unbalanced_str
        del brackets
        return _generate_unbalanced_string(order, length)
    
    return unbalanced_str


def _generate_samples(n: int, k: int, max_length: int = 1024, balanced: float = 0.5) -> List[str]:
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
        
    Returns:
        List[str]: A list of 'n' strings of length at most 'max_length' from the Dyck language of order 'k'."""
    
    balanced_strings = [_generate_balanced_string(k, random.randint(2, max_length)) for _ in tqdm(range(int(n * balanced)), desc='Generating balanced strings')]
    unbalanced_strings = [_generate_unbalanced_string(k, random.randint(2, max_length)) for _ in tqdm(range(n - len(balanced_strings)), desc='Generating unbalanced strings')]
    
    assert len(balanced_strings) + len(unbalanced_strings) == n, "The number of generated strings does not match the expected number."
    assert all(len(s) <= max_length for s in balanced_strings + unbalanced_strings), "Some strings exceed the maximum length."
    assert all(checker.is_dyck_word(s, k) for s in balanced_strings), "Some balanced strings are not members of the Dyck language."
    assert all(not checker.is_dyck_word(s, k) for s in unbalanced_strings), "Some unbalanced strings are members of the Dyck language."

    samples = balanced_strings + unbalanced_strings
    random.shuffle(samples)
    
    return samples
    

def generate_dataset(
    n: Annotated[int, typer.Option(help="The number of strings to generate.")] = 500_000, 
    k: Annotated[int, typer.Option(help="The order of the Dyck language.")] = 3,
    max_length: Annotated[int, typer.Option(help="The maximum length of the strings to generate.")] = 1024, 
    balanced: Annotated[float, typer.Option(help="The proportion of balanced strings to generate.")] = 0.5, 
    file: Annotated[bool, typer.Option(help="If present, the dataset will be saved to a file, otherwise it will be returned to a variable.")] = True
) -> List[Tuple[str, bool]]|None:
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
        List[Tuple[str, bool]]|None: A list of dictionaries that contain (str, bool), where string is the Dyck-k member string and class is its membership to the language.
    """
    
    strings = _generate_samples(n, k, max_length, balanced)
    dataset = [(s, checker.is_dyck_word(s, k)) for s in strings]

    if file:
        path = f"data/dyck-{k}_{n}-samples_{max_length}-len_p{str(balanced).replace('.', '')}.jsonl"
        with open(path, 'w') as f:
            for sample in tqdm(dataset, desc=f'Saving dataset to {path}'):
                json_record = json.dumps(sample)
                f.write(json_record + '\n')
            print(f'Dataset saved to {path}')
    else:
        return dataset

if __name__ == "__main__":
    typer.run(generate_dataset)
