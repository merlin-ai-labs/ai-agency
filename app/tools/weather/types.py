"""Type definitions for weather tool."""

from typing import TypedDict


class WeatherCondition(TypedDict):
    """Weather condition details."""

    main: str  # e.g., "Clouds", "Clear"
    description: str  # e.g., "scattered clouds"
    icon: str  # Icon code


class WeatherResponse(TypedDict):
    """Structured weather data from OpenWeatherMap API."""

    location: str
    temperature: float  # In specified units
    feels_like: float
    temp_min: float
    temp_max: float
    pressure: int  # hPa
    humidity: int  # percentage
    visibility: int  # meters
    wind_speed: float
    wind_deg: int | None
    clouds: int  # cloudiness percentage
    condition: str  # e.g., "Clear"
    description: str  # e.g., "clear sky"
    timestamp: int  # Unix timestamp


class WeatherResult(TypedDict):
    """Result returned by weather tool."""

    success: bool
    data: WeatherResponse | None
    error: str | None
    cached: bool  # Whether result was from cache
