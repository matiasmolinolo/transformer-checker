import typer
from tqdm import tqdm
from typing_extensions import Annotated

from .constants import BRACKETS

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
        if verbose:
            print(False)
        return False

    bracket_types = list(BRACKETS.items())[:k]

    opening_brackets = {opening for opening, _ in bracket_types}
    closing_brackets = {closing: opening for opening, closing in bracket_types}

    stack = []

    with tqdm(total=len(query), desc="Checking Dyck word", disable=not verbose) as pbar:
        for bracket in query:
            if bracket in opening_brackets:
                stack.append(bracket)
            elif bracket in closing_brackets:
                if not stack or closing_brackets[bracket] != stack.pop():
                    pbar.update(len(query) - pbar.n)  # Complete the progress bar
                    if verbose:
                        print(False)
                    return False
            else:
                pbar.update(len(query) - pbar.n)  # Complete the progress bar
                if verbose:
                    print(False)
                return False
            pbar.update(1)

    result = not stack
    if verbose:
        print(result)
    return result


if __name__ == "__main__":
    typer.run(is_dyck_word)
