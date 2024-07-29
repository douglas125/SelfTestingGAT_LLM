""" Utility functions for the analysis of results
"""
import json


def remove_function_strings(x):
    """Some LLMs will produce function. or function: before
    the real function name. We remove them for fair comparison
    """
    x = x.replace("functions.", "")
    x = x.replace("functions:", "")
    return x


def detect_invented_tools(tool_list, all_tools):
    """Checks if a list has invented tools.
    Returns the names of the invented tools
    """
    invented_tools = []
    for x in tool_list:
        adjusted_name = remove_function_strings(x)
        if adjusted_name not in all_tools:
            invented_tools.append(x)
    return ",".join(invented_tools)


def belongs_to_col(tool_list, col_tool_names):
    """Checks if a given tool list belongs to a column
    Every tool in the tool list has to match a function in the column
    And they need to be different if more than one
    """
    if len(tool_list) > 2:
        return False

    if len(tool_list) == 2 and tool_list[0] == tool_list[1]:
        return False

    required_matches = len(col_tool_names)
    n_matches = 0
    for t in tool_list:
        delta = 1 if (remove_function_strings(t) in str(col_tool_names)) else 0
        n_matches += delta

    return n_matches == required_matches


def is_tool_selection_correct(expected_answer, selected_tools, all_tools):
    """Checks if the selected tools are correct.
    Attempts to unfold ':' into multiple tools.
    Returns:
        (True if the selection is correct, list of invented tools)
    """
    y_true = json.loads(expected_answer.replace("'", '"'))
    try:
        y_pred = json.loads(selected_tools.replace("'", '"'))
    except:
        print(selected_tools.replace("'", '"'))
        y_pred = ["Invalid JSON"] + selected_tools.replace("'", '"').split(",")
        # raise
    y_pred = [remove_function_strings(x) for x in y_pred]
    if len(y_pred) == 1 and ":" in y_pred[0]:
        y_pred = y_pred[0].split(":")

    # handle the case where LLMs output tool_name:arguments
    y_pred = [x.split(":")[0] for x in y_pred]

    assert isinstance(y_pred, list)
    assert isinstance(y_true, list)
    invented_tools = [x for x in y_pred if (x not in all_tools)]

    # make sure that the lists match
    score = len(set(y_true).intersection(set(y_pred))) / len(
        set(y_true).union(set(y_pred))
    )
    return set(y_true) == set(y_pred), score, invented_tools, len(y_pred) == 0
