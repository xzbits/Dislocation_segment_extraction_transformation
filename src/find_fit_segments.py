import numpy as np
import math


def rmse_score(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def score_function_2_model(inputs1, targets1, inputs2, targets2, slope1, slope2):
    """
    Scoring the fitting lines with Cartesian coordinate
    :param inputs1: Input for segment 1
    :param targets1: Target values for segment 1
    :param inputs2: Input for segment 2
    :param targets2: Target values for segment 2
    :param slope1: Slope and offset values for segment 1
    :param slope2: Slope and offset values for segment 2
    :return: The total RMSE-Score for 2 segments
    """
    predictions1 = inputs1 * slope1[0] + slope1[1]
    predictions2 = inputs2 * slope2[0] + slope2[1]
    return rmse_score(predictions1, targets1) + rmse_score(predictions2, targets2)


def score_function_2_model_loglog_coordinate(inputs1, targets1, inputs2, targets2, slope1, slope2):
    """
    Scoring the fitting lines with Log-log coordinate
    :param inputs1: Input for segment 1
    :param targets1: Target values for segment 1
    :param inputs2: Input for segment 2
    :param targets2: Target values for segment 2
    :param slope1: Slope and offset values for segment 1
    :param slope2: Slope and offset values for segment 2
    :return: The total RMSE-Score for 2 segments
    """
    predictions1 = np.log(inputs1**-slope1[0] * math.e**-slope1[1])
    predictions2 = np.log(inputs2**-slope2[0] * math.e**-slope2[1])
    return rmse_score(predictions1, np.log(targets1)) + rmse_score(predictions2, np.log(targets2))


def get_split_point(target_x, target_y):
    output_score = -1
    output_index = -1
    output_slope1 = None
    output_slope2 = None
    output_error_list = list()
    no_element = target_x.shape[0]
    for i in range(2, no_element-1):
        # Linear regression model
        segment_1_x = target_x[:i]
        segment_1_y = target_y[:i]
        segment_2_x = target_x[i:]
        segment_2_y = target_y[i:]
        d1 = -1 * np.polyfit(segment_1_x, segment_1_y, 1)
        d2 = -1 * np.polyfit(segment_2_x, segment_2_y, 1)

        fit_score = score_function_2_model_loglog_coordinate(segment_1_x, segment_1_y, segment_2_x, segment_2_y, d1, d2)
        output_error_list.append(fit_score)
        if output_score == -1:
            output_score = fit_score
            output_index = i
            output_slope1 = d1
            output_slope2 = d2
        else:
            if output_score > fit_score:
                output_score = fit_score
                output_index = i
                output_slope1 = d1
                output_slope2 = d2
            else:
                pass

    return output_index, output_score, output_slope1, output_slope2, output_error_list
