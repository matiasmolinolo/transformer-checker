import typer
from typing_extensions import Annotated
from dyck_k_generator import constants as c
from tqdm import tqdm


def is_dyck_word(
        query: Annotated[str, typer.Argument()], 
        k: Annotated[int, typer.Argument()], 
        verbose: Annotated[bool, typer.Option()] = False
    ) -> bool|None:
    """
    Check if a word is a member of the Dyck language of order k.

    Args:
        query (str): The word to check.
        k (int): The order of the Dyck language.

    Returns:
        bool: True if the word is a member of the Dyck language of order k, False otherwise.
    """

    if len(query) % 2 != 0:
        return False
    
    bracket_types = {k: v for k, v in list(c.BRACKETS.items())[:k]}
    closing_brackets = {v: k for k, v in bracket_types.items()}

    stack = []

    for bracket in tqdm(query, desc="Checking Dyck word", disable=not verbose):
        if bracket in bracket_types:
            stack.append(bracket)
        elif bracket in closing_brackets:
            if not stack or closing_brackets[bracket] != stack.pop():
                return False
            
    if verbose:
        print(not stack)
    else:
        return not stack

if __name__ == "__main__":
    is_dyck_word()

    