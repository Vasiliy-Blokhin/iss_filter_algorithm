from numpy import interp


def interp_4_dote(
        point_limits: list[int],  # [-4, 4]
        prcnt_limits: list[float],  # [-3, 5]
        dote_prcnt: float,  # x -- 0.7
        prcnt_start_limit: float,  # 0.2
        is_abs_limit=True
):
    # Реализует интерполяцию в диапозоне -4 -- 0 -- 4
    if dote_prcnt > prcnt_start_limit:
        return interp(
            dote_prcnt,
            xp=[prcnt_start_limit, prcnt_limits[1]],
            fp=[0, point_limits[1]]
        )

    if dote_prcnt < prcnt_start_limit:
        if is_abs_limit:
            prcnt_start_limit = -prcnt_start_limit
        return interp(
            dote_prcnt,
            xp=[prcnt_limits[0], prcnt_start_limit],
            fp=[point_limits[0], 0]
        )

    return 0


def interp_6_dote(
        point_limits: list[int],  # [-6, 6]
        prcnt_limits: list[float],  # [-2, -1, 1, 2]
        dote_prcnt: float,  # x -- 0.7
        prcnt_start_limit: float,  # 0.2
        is_abs_limit=True
):
    # Реализует интерполяцию в диапозоне -6 -- 6 -- 0 -- -6 -- 6
    if (
        dote_prcnt > prcnt_start_limit
        and dote_prcnt < prcnt_limits[2]
    ):
        return interp(
            dote_prcnt,
            xp=[prcnt_start_limit, prcnt_limits[2]],
            fp=[0, point_limits[1]]
        )
    elif dote_prcnt >= prcnt_limits[2]:
        return interp(
            dote_prcnt,
            xp=[prcnt_limits[2], prcnt_limits[3]],
            fp=[point_limits[1], point_limits[0]]
        )

    if is_abs_limit:
        prcnt_start_limit = -prcnt_start_limit
    if (
        dote_prcnt < prcnt_start_limit
        and dote_prcnt > prcnt_limits[1]
    ):
        return interp(
            dote_prcnt,
            xp=[prcnt_limits[1], prcnt_start_limit],
            fp=[point_limits[0], 0],
        )
    elif dote_prcnt < prcnt_limits[1]:
        return interp(
            dote_prcnt,
            xp=[prcnt_limits[0], prcnt_limits[1]],
            fp=[point_limits[1], point_limits[0]],
        )
    else:
        return 0
