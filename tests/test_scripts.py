# %% IMPORTS

from cs231n_spr_2025 import scripts

# %% FUNCTIONS


def test_main() -> None:
    # given
    argv = ["x", "y", "z"]
    # when
    result = scripts.main(argv)
    # then
    assert result == 0, "Result should be 0!"
