from pydantic import BaseModel, Field
from .base import ToolSpec


class WeatherArgs(BaseModel):
    city: str = Field(min_length=1, max_length=30)


_WEATHER = {"上海": {"condition": "晴", "temperature_c": 26}, "北京": {"condition": "多云", "temperature_c": 22}, "广州": {"condition": "小雨", "temperature_c": 28}}


def _handle(args: WeatherArgs, *, session_state: dict) -> dict:
    return {"city": args.city, **_WEATHER.get(args.city, {"condition": "未知", "temperature_c": None})}


def weather_tool() -> ToolSpec:
    return ToolSpec("weather", "查询演示天气数据", WeatherArgs, _handle)

