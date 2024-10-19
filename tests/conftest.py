import pytest


@pytest.fixture
def unexpected_param_msg():
    return "Error: Unexpected parameter(s): "
