"""OpenWeatherMap API client with retry logic and error handling.

This client demonstrates best practices for external API integration:
- Retry logic with exponential backoff
- Timeout handling
- Comprehensive error messages
- Request/response logging
"""

import logging
from typing import Any

import httpx

from app.config import settings
from app.core.decorators import log_execution, retry, timeout
from app.core.exceptions import WeatherError
from app.tools.weather.types import WeatherResponse

logger = logging.getLogger(__name__)


class WeatherClient:
    """OpenWeatherMap API client.

    Provides methods to fetch current weather data with built-in retry logic,
    timeout handling, and structured error messages.

    Attributes:
        api_key: OpenWeatherMap API key
        base_url: API base URL
        timeout_seconds: Request timeout in seconds

    Example:
        >>> client = WeatherClient()
        >>> weather = await client.get_current_weather("London", units="metric")
        >>> print(f"Temperature: {weather['temperature']}°C")
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int | None = None,
    ):
        """Initialize weather client.

        Args:
            api_key: OpenWeatherMap API key. If None, uses settings.openweather_api_key
            base_url: API base URL. If None, uses settings.openweather_base_url
            timeout_seconds: Request timeout. If None, uses settings.openweather_timeout

        Raises:
            ValueError: If API key is not provided
        """
        self.api_key = api_key or settings.openweather_api_key
        if not self.api_key:
            msg = "OpenWeatherMap API key not provided"
            raise ValueError(msg)

        self.base_url = base_url or settings.openweather_base_url
        self.timeout_seconds = timeout_seconds or settings.openweather_timeout

        logger.info(
            "Initialized OpenWeatherMap client",
            extra={
                "base_url": self.base_url,
                "timeout": self.timeout_seconds,
            },
        )

    @log_execution
    @timeout(seconds=30.0)
    @retry(
        max_attempts=3,
        backoff_type="exponential",
        min_wait=1.0,
        max_wait=10.0,
        exceptions=(httpx.HTTPError,),
    )
    async def get_current_weather(
        self,
        location: str,
        units: str = "metric",
    ) -> WeatherResponse:
        """Fetch current weather for a location.

        Args:
            location: City name (e.g., "London", "New York") or coordinates
            units: Temperature units - "metric" (Celsius), "imperial" (Fahrenheit),
                or "standard" (Kelvin)

        Returns:
            Structured weather data

        Raises:
            WeatherError: If API call fails or returns invalid data

        Example:
            >>> weather = await client.get_current_weather("Paris", units="metric")
            >>> print(f"{weather['temperature']}°C, {weather['description']}")
        """
        # Build API URL
        url = f"{self.base_url}/weather"
        params = {
            "q": location,
            "appid": self.api_key,
            "units": units,
        }

        try:
            logger.debug(
                "Calling OpenWeatherMap API",
                extra={
                    "location": location,
                    "units": units,
                },
            )

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    params=params,
                    timeout=self.timeout_seconds,
                )

                # Handle HTTP errors
                if response.status_code == 404:
                    raise WeatherError(
                        f"Location not found: {location}",
                        details={"location": location, "status_code": 404},
                    )

                if response.status_code == 401:
                    raise WeatherError(
                        "Invalid API key",
                        details={"status_code": 401},
                    )

                response.raise_for_status()
                data = response.json()

                # Parse response into structured format
                weather_response = self._parse_response(data, location)

                logger.info(
                    "OpenWeatherMap API call successful",
                    extra={
                        "location": location,
                        "temperature": weather_response["temperature"],
                        "condition": weather_response["condition"],
                    },
                )

                return weather_response

        except httpx.HTTPError as e:
            logger.error(
                "OpenWeatherMap API call failed",
                extra={
                    "location": location,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise WeatherError(
                f"Weather API call failed: {str(e)}",
                details={
                    "location": location,
                    "error_type": type(e).__name__,
                },
                original_error=e,
            ) from e
        except Exception as e:
            logger.error(
                "Unexpected error in weather API call",
                extra={
                    "location": location,
                    "error": str(e),
                },
            )
            raise WeatherError(
                f"Unexpected error: {str(e)}",
                details={"location": location},
                original_error=e,
            ) from e

    def _parse_response(self, data: dict[str, Any], location: str) -> WeatherResponse:
        """Parse OpenWeatherMap API response into structured format.

        Args:
            data: Raw API response
            location: Location name (for the response)

        Returns:
            Structured weather response
        """
        try:
            main = data["main"]
            weather = data["weather"][0] if data["weather"] else {}
            wind = data.get("wind", {})
            clouds = data.get("clouds", {})

            return WeatherResponse(
                location=location,
                temperature=main["temp"],
                feels_like=main["feels_like"],
                temp_min=main["temp_min"],
                temp_max=main["temp_max"],
                pressure=main["pressure"],
                humidity=main["humidity"],
                visibility=data.get("visibility", 0),
                wind_speed=wind.get("speed", 0.0),
                wind_deg=wind.get("deg"),
                clouds=clouds.get("all", 0),
                condition=weather.get("main", "Unknown"),
                description=weather.get("description", ""),
                timestamp=data["dt"],
            )
        except (KeyError, IndexError, TypeError) as e:
            raise WeatherError(
                "Invalid API response format",
                details={"error": str(e)},
                original_error=e,
            ) from e
