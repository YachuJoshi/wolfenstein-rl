def get_env(level, mode):
    if level not in ("basic", "defend", "deadly"):
        raise ValueError("Need a valid level!")

    if level == "basic":
        from rl.env.basic import WolfensteinBasicEnv

        return WolfensteinBasicEnv(render_mode=mode)

    if level == "defend":
        from rl.env.defend import WolfensteinDefendTheCenterEnv

        return WolfensteinDefendTheCenterEnv(render_mode=mode)

    from rl.env.deadly import WolfensteinDeadlyCorridorEnv

    return WolfensteinDeadlyCorridorEnv(render_mode=mode)


def get_dir(level):
    if level not in ("basic", "defend", "deadly"):
        raise ValueError("Need a valid level!")

    return f"./logs/{level}", f"./models/{level}"


def get_model_dir(level, steps):
    if level not in ("basic", "defend", "deadly"):
        raise ValueError("Need a valid level!")

    return f"./models/{level}/model_{steps}"
