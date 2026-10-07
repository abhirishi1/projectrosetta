import re
from pathlib import Path

import numpy as np
import pandas as pd
from scenariogeneration import xosc

CAR_MODELS = ["car_white.osgb", "car_blue.osgb", "car_red.osgb", "car_yellow.osgb"]
XYT_COLUMNS = ["x", "y", "time"]
XYT_NUMBER = re.compile(r"[+-]?(\d+(,\d*)?|,\d+)([eE][+-]?\d+)?")


def read_xyt(path: Path) -> pd.DataFrame:
    """
    Read x y time points.

    Returns:
        DataFrame with x, y, time, and yaw.

    """
    _validate_xyt_rows(Path(path))
    df = pd.read_csv(Path(path), sep=r"\s+", header=None, names=XYT_COLUMNS, decimal=",")
    df["time"] -= df["time"].iloc[0]
    df["yaw"] = _yaw(df)
    return df


def _validate_xyt_rows(path: Path) -> None:
    """
    Check that every non-blank row holds exactly x, y, and time as comma-decimal numbers.

    Raises:
        ValueError: If a row has the wrong number of values or a non-numeric value.

    """
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            values = line.split()
            if not values:
                continue
            if len(values) != len(XYT_COLUMNS):
                raise ValueError(
                    f"{path}, line {line_number}: expected 3 numeric values (x y time), "
                    f"found {len(values)}"
                )
            for name, value in zip(XYT_COLUMNS, values):
                if not XYT_NUMBER.fullmatch(value):
                    raise ValueError(
                        f"{path}, line {line_number}: {name} value {value!r} is not a number "
                        "(expected comma decimals, e.g. 0,5)"
                    )


def xyt_files_to_xosc(xyt_paths: list[Path], output_path: Path) -> Path:
    """
    Write a minimal trajectory replay .xosc.

    Returns:
        Generated .xosc path.

    """
    actors = [(Path(path).stem, read_xyt(path)) for path in xyt_paths]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _scenario(actors).write_xml(str(output))
    return output


def _scenario(actors: list[tuple[str, pd.DataFrame]]) -> xosc.Scenario:
    entities = xosc.Entities()
    init = xosc.Init()
    act = xosc.Act("act", stoptrigger=_time_trigger("act_stop", _stop_time(actors), "stop"))

    for index, (name, df) in enumerate(actors):
        entities.add_scenario_object(name, _vehicle(name, CAR_MODELS[index % len(CAR_MODELS)]))
        init.add_init_action(name, xosc.TeleportAction(_position(df.iloc[0])))
        act.add_maneuver_group(_maneuver_group(name, df))

    storyboard = xosc.StoryBoard(init, _time_trigger("scenario_stop", _stop_time(actors), "stop"))
    storyboard.add_act(act)
    return xosc.Scenario(
        "xyt replay",
        "project-rosetta",
        xosc.ParameterDeclarations(),
        entities,
        storyboard,
        xosc.RoadNetwork(),
        xosc.Catalog(),
        osc_minor_version=1,
    )


def _vehicle(name: str, model: str) -> xosc.Vehicle:
    return xosc.Vehicle(
        f"{name}_vehicle",
        xosc.VehicleCategory.car,
        xosc.BoundingBox(1.8, 4.5, 1.8, 1.5, 0, 0.9),
        xosc.Axle(0.5, 0.6, 1.6, 3.1, 0.3),
        xosc.Axle(0, 0.6, 1.6, 0, 0.3),
        69,
        10,
        10,
        model3d=model,
    )


def _maneuver_group(name: str, df: pd.DataFrame) -> xosc.ManeuverGroup:
    trajectory = xosc.Trajectory(f"{name}_trajectory", closed=False)
    trajectory.add_shape(
        xosc.Polyline(
            time=df["time"].tolist(),
            positions=[_position(row) for row in df.itertuples()],
        )
    )
    event = xosc.Event(f"{name}_event", xosc.Priority.overwrite)
    event.add_action(
        f"{name}_action",
        xosc.FollowTrajectoryAction(
            trajectory,
            following_mode=xosc.FollowingMode.position,
            reference_domain=xosc.ReferenceContext.absolute,
            scale=1,
            offset=0,
        ),
    )
    event.add_trigger(_time_trigger(f"{name}_start", 0, "start"))
    maneuver = xosc.Maneuver(f"{name}_maneuver")
    maneuver.add_event(event)
    group = xosc.ManeuverGroup(f"{name}_group")
    group.add_actor(name)
    group.add_maneuver(maneuver)
    return group


def _position(row) -> xosc.WorldPosition:
    return xosc.WorldPosition(x=row.x, y=row.y, h=row.yaw)


def _time_trigger(name: str, time: float, point: str) -> xosc.ValueTrigger:
    return xosc.ValueTrigger(
        name,
        0,
        xosc.ConditionEdge.none,
        xosc.SimulationTimeCondition(time, xosc.Rule.greaterThan),
        triggeringpoint=point,
    )


def _stop_time(actors: list[tuple[str, pd.DataFrame]]) -> float:
    return max(df["time"].iloc[-1] for _, df in actors)


def _yaw(df: pd.DataFrame) -> np.ndarray:
    dx = df["x"].diff().fillna(0)
    dy = df["y"].diff().fillna(0)
    moving = np.hypot(dx, dy) > 0.001
    yaw = np.zeros(len(df))
    last_yaw = np.arctan2(dy[moving].iloc[0], dx[moving].iloc[0]) if moving.any() else 0.0
    for index in range(len(df)):
        if moving.iloc[index]:
            last_yaw = np.arctan2(dy.iloc[index], dx.iloc[index])
        yaw[index] = last_yaw
    return np.unwrap(yaw)
