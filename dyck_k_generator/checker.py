import typer
from tqdm import tqdm
from typing_extensions import Annotated

from dyck_k_generator import constants as c


def is_dyck_word(
    query: Annotated[str, typer.Argument()],
    k: Annotated[int, typer.Argument()],
    verbose: Annotated[bool, typer.Option()] = False,
) -> bool:
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

    bracket_types = list(c.BRACKETS.items())[:k]

    opening_brackets = {opening for opening, _ in bracket_types}
    closing_brackets = {closing: opening for opening, closing in bracket_types}

    stack = []

    for bracket in tqdm(query, desc="Checking Dyck word", disable=not verbose):
        if bracket in opening_brackets:
            stack.append(bracket)
        elif bracket in closing_brackets:
            if not stack or closing_brackets[bracket] != stack.pop():
                return False
        else:
            return False

    if verbose:
        print(not stack)
        return not stack
    else:
        return not stack


if __name__ == "__main__":
    typer.run(is_dyck_word)
