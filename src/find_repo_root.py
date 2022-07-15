## Function to get the repository path # Max Langer # 2022-07-07 ##

# import the needed modules
import os

import git


def get_repo_root() -> str:
    """Gets the absolute path of the Git repository.
        If not in a repository, throws an error.

    Returns:
        str: Root of the Git repository.
    """
    try:
        git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
        repo_root = git_repo.git.rev_parse("--show-toplevel")
        return repo_root
    except:
        raise Exception("Please use the scripts from within the repository.")


if __name__ == "__main__":
    get_repo_root()
