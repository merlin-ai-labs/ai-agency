"""Weather tool v1 - Current weather lookup.

This tool demonstrates the standard pattern for external API tools:
1. Call external API via client
2. Store call history in database
3. Return structured result
"""

import logging
import time
from datetime import datetime

from sqlmodel import Session

from app.db.base import get_session
from app.db.repositories.weather_repository import WeatherRepository
from app.tools.weather.client import WeatherClient
from app.tools.weather.types import WeatherResult

logger = logging.getLogger(__name__)


async def get_weather(
    location: str,
    tenant_id: str,
    units: str = "metric",
    use_cache: bool = True,
    cache_hours: int = 1,
) -> WeatherResult:
    """Get current weather for a location.

    Fetches weather data from OpenWeatherMap API and stores the call history
    in the database. Supports optional caching to reduce API calls.

    Args:
        location: City name (e.g., "London", "Paris", "Tokyo")
        tenant_id: Tenant identifier for multi-tenancy
        units: Temperature units - "metric" (Celsius), "imperial" (Fahrenheit),
            or "standard" (Kelvin)
        use_cache: Whether to check cache before making API call
        cache_hours: How many hours to consider cached data valid

    Returns:
        Weather result with data or error

    Example:
        >>> result = await get_weather("London", tenant_id="acme", units="metric")
        >>> if result["success"]:
        ...     print(f"Temperature: {result['data']['temperature']}°C")
    """
    # Initialize client and repository
    client = WeatherClient()

    start_time = time.time()

    with Session(get_session()) as session:
        repository = WeatherRepository(session)

        # Check cache if enabled
        if use_cache:
            cached_calls = repository.get_calls_by_location(
                tenant_id=tenant_id,
                location=location,
                hours=cache_hours,
            )
            if cached_calls:
                cached = cached_calls[0]
                if cached.success and cached.response_data:
                    cache_age_seconds = (datetime.utcnow() - cached.created_at).seconds
                    logger.info(
                        "Returning cached weather data",
                        extra={
                            "location": location,
                            "tenant_id": tenant_id,
                            "cache_age_minutes": cache_age_seconds // 60,
                        },
                    )
                    return WeatherResult(
                        success=True,
                        data=cached.response_data,  # type: ignore
                        error=None,
                        cached=True,
                    )

        # Make API call
        try:
            weather_data = await client.get_current_weather(location, units=units)
            api_call_ms = int((time.time() - start_time) * 1000)

            # Store successful call in database
            repository.create_call(
                tenant_id=tenant_id,
                location=location,
                units=units,
                temperature=weather_data["temperature"],
                feels_like=weather_data["feels_like"],
                weather_condition=weather_data["condition"],
                weather_description=weather_data["description"],
                humidity=weather_data["humidity"],
                wind_speed=weather_data["wind_speed"],
                response_data=weather_data,  # type: ignore
                success=True,
                api_call_ms=api_call_ms,
            )
            session.commit()

            logger.info(
                "Weather data retrieved successfully",
                extra={
                    "location": location,
                    "tenant_id": tenant_id,
                    "api_call_ms": api_call_ms,
                },
            )

            return WeatherResult(
                success=True,
                data=weather_data,
                error=None,
                cached=False,
            )

        except Exception as e:
            api_call_ms = int((time.time() - start_time) * 1000)

            # Store failed call in database
            repository.create_call(
                tenant_id=tenant_id,
                location=location,
                units=units,
                temperature=None,
                feels_like=None,
                weather_condition=None,
                weather_description=None,
                humidity=None,
                wind_speed=None,
                response_data={},
                success=False,
                error_message=str(e),
                api_call_ms=api_call_ms,
            )
            session.commit()

            logger.error(
                "Weather data retrieval failed",
                extra={
                    "location": location,
                    "tenant_id": tenant_id,
                    "error": str(e),
                },
            )

            return WeatherResult(
                success=False,
                data=None,
                error=str(e),
                cached=False,
            )
